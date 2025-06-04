from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import asyncio
import json
from anthropic import Anthropic
from anthropic.types import Message, MessageParam, ThinkingConfigParam, ToolUseBlock, TextBlock

from app.core.function import FunctionTool
from app.schemas.response import AgentResult, StreamChunk

class Agent:
    """An agent that can interact with Anthropic's Claude API."""
    
    def __init__(
        self,
        name: str,
        instructions: str,
        model: str = "claude-sonnet-4-20250514",
        tools: Optional[List[Union[FunctionTool, Dict[str, Any]]]] = None,
        handoffs: Optional[List['Agent']] = None,
        output_type: Optional[type] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        thinking: ThinkingConfigParam | None = None,
        client: Optional[Anthropic] = None
    ):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.tools = tools or []
        self.handoffs = handoffs or []
        self.output_type = output_type
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking = thinking
        self.client = client or Anthropic()

        if self.thinking and self.thinking["type"] == "enabled":
            self.temperature = 1.0
        
        # Convert function tools to Anthropic format
        self._invoke_tools = []
        self._tool_map = {}
                
        for tool in self.tools:
            if isinstance(tool, FunctionTool):
                anthropic_tool = tool.to_anthropic_tool()
                self._invoke_tools.append(anthropic_tool)
                self._tool_map[tool.name] = tool
            elif isinstance(tool, dict):
                self._invoke_tools.append(tool)
                
        # Add handoff tools
        for agent in self.handoffs:
            handoff_tool = {
                "name": f"handoff_to_{agent.name.lower().replace(' ', '_')}",
                "description": f"Handoff control to {agent.name}",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message to pass to the next agent"
                        }
                    },
                    "required": ["message"]
                }
            }
            self._invoke_tools.append(handoff_tool)
            
    async def _process_tool_calls(self, message: Message) -> tuple[List[MessageParam], Optional['Agent'], bool]:
        """Process tool calls and return tool results, handoff agent, and completion status."""
        tool_results = []
        handoff_agent = None
        is_complete = False
        
        if not message.content:
            return tool_results, handoff_agent, True
            
        for content_block in message.content:
            if hasattr(content_block, 'type') and content_block.type == "tool_use":
                tool_name = content_block.name
                tool_input = content_block.input
                tool_use_id = content_block.id
                
                # Check for handoff
                if tool_name.startswith("handoff_to_"):
                    agent_name = tool_name.replace("handoff_to_", "").replace("_", " ")
                    for agent in self.handoffs:
                        if agent.name.lower().replace(" ", "_") == agent_name:
                            handoff_agent = agent
                            break
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": f"Handed off to {agent_name}"
                    })
                    continue
                    
                # Execute function tool
                if tool_name in self._tool_map:
                    try:
                        result = await self._tool_map[tool_name].call(**tool_input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": str(result)
                        })
                    except Exception as e:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": f"Error: {str(e)}",
                            "is_error": True
                        })
                        
        return tool_results, handoff_agent, is_complete
    
    def zip_response(self, response: Message) -> str:
        return ""

    async def run(self, input_message: List[MessageParam] | str, max_turns: int = 10) -> AgentResult:
        """Run the agent with the given input."""
        if isinstance(input_message, str):
            messages: List[MessageParam] = [
                {"role": "user", "content": input_message}
            ]
        else:
            messages = input_message
        
        current_agent = self
        tool_blocks = []
        tool_result_blocks = []
        thinking_blocks = []
        
        for turn in range(max_turns):
            # Prepare system message
            system_message = current_agent.instructions

            if len(tool_blocks) > 0 or len(thinking_blocks) > 0:
                interleaved_blocks = []
                if len(thinking_blocks) > 0:
                    interleaved_blocks.append(thinking_blocks[-1])
                    interleaved_blocks.append(tool_blocks[-1])
                messages.append({
                    "role": "assistant",
                    "content": interleaved_blocks
                })

                thinking_blocks = []
                tool_blocks = []
            
            if len(tool_result_blocks) > 0:
                messages.append({
                    "role": "user",
                    "content": tool_result_blocks
                })
                tool_result_blocks = []

            kwargs = {
                "model": current_agent.model,
                "max_tokens": current_agent.max_tokens,
                "temperature": current_agent.temperature,
                "system": system_message,
                "messages": messages,
                "thinking": current_agent.thinking,
            }
            if current_agent._invoke_tools:
                kwargs["tools"] = current_agent._invoke_tools
            
            # Make API call
            response = await current_agent.client.messages.create(**kwargs)
            final_block = response.content[-1] if response.content else None

            if final_block and not isinstance(final_block, TextBlock):
                for block in response.content:
                    if block.type == "thinking" or block.type == "redacted_thinking":
                        thinking_blocks.append(block)
                    elif block.type == "tool_use" or block.type == "server_tool_use":
                        tool_blocks.append(block)
                    elif block.type == "web_search_tool_result":
                        tool_result_blocks.append(block)
            
            # Process tool calls
            tool_results, handoff_agent, is_complete = await current_agent._process_tool_calls(response)
            tool_result_blocks.extend(tool_results)

            final_block = response.content[-1] if response.content else None

            if final_block and isinstance(final_block, TextBlock):
                final_output = self.zip_response(response)
                return AgentResult(
                    final_output=final_output,
                    messages=messages,
                    agent_name=current_agent.name
                )
            return AgentResult(
                final_output="I'm sorry, I got an unexpected response.",
                messages=messages,
                agent_name=current_agent.name
            )
        
    async def run_stream(self, messages: List[MessageParam] | str, max_turns: int = 10) -> AsyncGenerator[StreamChunk, None]:
        """Run the agent with streaming output."""
        current_agent = self

        if isinstance(messages, str):
            messages = [
                {"role": "user", "content": messages}
            ]
        else:
            messages = messages
            
        for turn in range(max_turns):
            # Prepare system message
            system_message = current_agent.instructions

            kwargs = {
                "model": current_agent.model,
                "max_tokens": current_agent.max_tokens,
                "temperature": current_agent.temperature,
                "system": system_message,
                "messages": messages,
                "stream": True,
            }
            if current_agent._invoke_tools:
                kwargs["tools"] = current_agent._invoke_tools
            
            # Make streaming API call
            stream = await current_agent.client.messages.create(**kwargs)
            
            # Collect streamed content
            full_content = []
            current_text = ""
            
            async for chunk in stream:
                if chunk.type == "message_start":
                    continue
                elif chunk.type == "content_block_start":
                    if chunk.content_block.type == "text":
                        current_text = ""
                elif chunk.type == "content_block_delta":
                    if chunk.delta.type == "text_delta":
                        current_text += chunk.delta.text
                        yield StreamChunk(
                            content=chunk.delta.text,
                            is_complete=False,
                            agent_name=current_agent.name,
                            messages=messages
                        )
                elif chunk.type == "content_block_stop":
                    if current_text:
                        full_content.append({
                            "type": "text",
                            "text": current_text
                        })
                elif chunk.type == "message_delta":
                    continue
                elif chunk.type == "message_stop":
                    break
            
            # Add assistant message
            messages.append({
                "role": "assistant",
                "content": full_content
            })
            
            # Create a mock response object for tool processing
            class MockResponse:
                def __init__(self, content):
                    self.content = content
            
            mock_response = MockResponse(full_content)
            
            # Process tool calls
            tool_results, handoff_agent, is_complete = await current_agent._process_tool_calls(mock_response)
            
            # Handle handoff
            if handoff_agent:
                current_agent = handoff_agent
                yield StreamChunk(
                    content=f"\n[Handed off to {handoff_agent.name}]\n",
                    is_complete=False,
                    agent_name=current_agent.name,
                    messages=messages
                )
                continue
                
            # Add tool results if any
            if tool_results:
                messages.extend(tool_results)
                continue
                
            # Check for completion
            if full_content and len(full_content) > 0:
                final_content = ""
                for content_block in full_content:
                    if content_block.get("type") == "text":
                        final_content += content_block.get("text", "")
                        
                # If no tools were called and we have text content, we're done
                if final_content and not any(block.get("type") == "tool_use" for block in full_content):
                    yield StreamChunk(
                        content="",
                        is_complete=True,
                        agent_name=current_agent.name,
                        messages=messages
                    )
                    return
                    
        # If we've reached max turns, signal completion
        yield StreamChunk(
            content="",
            is_complete=True,
            agent_name=current_agent.name,
            messages=messages
        )


class Runner:
    """Runner for executing agents."""
    
    @staticmethod
    async def run(agent: Agent, input_message: List[MessageParam] | str, max_turns: int = 10) -> AgentResult:
        """Run an agent asynchronously."""
        return await agent.run(input_message, max_turns)
        
    @staticmethod
    def run_sync(agent: Agent, input_message: List[MessageParam] | str, max_turns: int = 10) -> AgentResult:
        """Run an agent synchronously."""
        return asyncio.run(agent.run(input_message, max_turns))
        
    @staticmethod
    async def run_stream(agent: Agent, input_message: List[MessageParam] | str, max_turns: int = 10) -> AsyncGenerator[StreamChunk, None]:
        """Run an agent with streaming output."""
        async for chunk in agent.run_stream(input_message, max_turns):
            yield chunk
            
    @staticmethod
    def run_stream_sync(agent: Agent, input_message: List[MessageParam] | str, max_turns: int = 10):
        """Run an agent with streaming output synchronously."""
        async def _run_stream():
            async for chunk in agent.run_stream(input_message, max_turns):
                yield chunk
                
        # For synchronous streaming, we need to handle the async generator
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async_gen = _run_stream()
            while True:
                try:
                    chunk = loop.run_until_complete(async_gen.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break
        finally:
            loop.close()