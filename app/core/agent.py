from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import asyncio
import json
from anthropic import Anthropic
from anthropic.types import Message, MessageParam, ThinkingConfigParam, ContentBlock, TextBlock

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
                anthropic_tool = tool.to_invoke_tool()
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
            
    async def _process_tool_calls(self, content_block: ContentBlock) -> tuple[List[MessageParam], Optional['Agent'], bool]:
        """Process tool calls and return tool results, handoff agent, and completion status."""
        tool_results = []
        handoff_agent = None
        is_complete = False
        
        if not content_block:
            return tool_results, handoff_agent, True
            
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
        block_caches: List[ContentBlock] = []
        tool_result_blocks = []
        
        for turn in range(max_turns):
            # Prepare system message
            system_message = current_agent.instructions

            if len(block_caches) > 0:
                messages.append({
                    "role": "assistant",
                    "content": block_caches
                })
                block_caches = []
            
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
                        block_caches.append(block)
                    elif block.type == "tool_use":
                        block_caches.append(block)
                        # Process tool calls
                        tool_results, handoff_agent, is_complete = await current_agent._process_tool_calls(block)
                        tool_result_blocks.extend(tool_results)
                continue

            if final_block and isinstance(final_block, TextBlock):
                final_output = final_block.text.strip()
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

        current_agent = self
        skip_tool_use = True
        block_caches: List[ContentBlock] = []
        tool_result_blocks = []
            
        for turn in range(max_turns):
            # Prepare system message
            system_message = current_agent.instructions

            if len(block_caches) > 0:
                messages.append({
                    "role": "assistant",
                    "content": block_caches
                })
                block_caches = []
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
                "stream": True,
            }

            if current_agent._invoke_tools:
                kwargs["tools"] = current_agent._invoke_tools
            
            async with current_agent.client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if event.type == "message_start":
                        skip_tool_use = True
                        continue
                    elif event.type == "content_block_start":
                        if event.content_block.type == "text":
                            yield StreamChunk(
                                type="text_start",
                                content="\n<text>\n",
                                is_complete=False,
                                agent_name=current_agent.name,
                            )
                        elif event.content_block.type == "thinking":
                            yield StreamChunk(
                                type="thinking_start",
                                content="\n<thinking>\n",
                                is_complete=False,
                                agent_name=current_agent.name,
                            )
                        elif event.content_block.type == "tool_use":
                            skip_tool_use = False
                            yield StreamChunk(
                                type="tool_use_start",
                                content="\n<tool_use>\n",
                                is_complete=False,
                                agent_name=current_agent.name,
                            )
                    elif event.type == "thinking":
                        yield StreamChunk(
                            type="thinking",
                            content=event.thinking,
                            is_complete=False,
                            agent_name=current_agent.name,
                        )
                    elif event.type == "text":
                        yield StreamChunk(
                            type="text",
                            content=event.text,
                            is_complete=False,
                            agent_name=current_agent.name,
                        )
                    elif event.type == "content_block_stop":
                        if event.content_block.type == "text":
                            block_caches.append(event.content_block)
                            yield StreamChunk(
                                type="text_stop",
                                content="\n</text>\n",
                                is_complete=False,
                                agent_name=current_agent.name,
                            )
                        elif event.content_block.type == "thinking":
                            block_caches.append(event.content_block)
                            yield StreamChunk(
                                type="thinking_stop",
                                content="\n</thinking>\n",
                                is_complete=False,
                                agent_name=current_agent.name,
                            )
                        elif event.content_block.type == "tool_use":
                            block_caches.append(event.content_block)
                            yield StreamChunk(
                                type="tool_use_stop",
                                content="\n</tool_use>\n",
                                is_complete=False,
                                agent_name=current_agent.name,
                            )
                    elif event.type == "message_stop":
                        break
            if skip_tool_use:
                yield StreamChunk(
                    content="",
                    is_complete=True,
                    agent_name=current_agent.name,
                    type="text"
                )
            
            yield StreamChunk(
                content="\n<tool_result>\n",
                is_complete=False,
                agent_name=current_agent.name,
                type="tool_result_start"
            )
            
            # Process tool calls
            tool_results, handoff_agent, is_complete = await current_agent._process_tool_calls(content_block=block_caches[-1] if block_caches else None)
            
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
                tool_result_blocks.extend(tool_results)
                continue
        # If we've reached max turns, signal completion
        yield StreamChunk(
            content="",
            type="error",
            is_complete=True,
            agent_name=current_agent.name
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