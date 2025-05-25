from typing import Any, Dict, List, Optional, Union, Callable, Awaitable, AsyncGenerator
from dataclasses import dataclass, field
import asyncio
import json
from anthropic import Anthropic
from anthropic.types import Message, MessageParam


@dataclass
class AgentResult:
    """Result of an agent run."""
    final_output: str
    messages: List[MessageParam] = field(default_factory=list)
    agent_name: str = ""


@dataclass
class StreamChunk:
    """A chunk of streamed content."""
    content: str
    is_complete: bool = False
    agent_name: str = ""
    messages: List[MessageParam] = field(default_factory=list)
    

class FunctionTool:
    """Wrapper for function tools."""
    
    def __init__(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None):
        self.func = func
        self.name = name or func.__name__
        self.description = description or func.__doc__ or ""
        
    def to_anthropic_tool(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format."""
        # Get function signature for parameters
        import inspect
        sig = inspect.signature(self.func)
        
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            param_type = "string"  # Default type
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                    
            properties[param_name] = {"type": param_type}
            
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
                
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
        
    async def call(self, **kwargs) -> Any:
        """Call the function with given arguments."""
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(**kwargs)
        else:
            return self.func(**kwargs)


def function_tool(func: Callable) -> FunctionTool:
    """Decorator to create a function tool."""
    return FunctionTool(func)


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
        self.client = client or Anthropic()
        
        # Convert function tools to Anthropic format
        self._anthropic_tools = []
        self._tool_map = {}
        
        for tool in self.tools:
            if isinstance(tool, FunctionTool):
                anthropic_tool = tool.to_anthropic_tool()
                self._anthropic_tools.append(anthropic_tool)
                self._tool_map[tool.name] = tool
            elif isinstance(tool, dict):
                self._anthropic_tools.append(tool)
                
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
            self._anthropic_tools.append(handoff_tool)
            
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
        
    async def run(self, input_message: List[MessageParam] | str, max_turns: int = 10) -> AgentResult:
        """Run the agent with the given input."""
        if isinstance(input_message, str):
            messages: List[MessageParam] = [
                {"role": "user", "content": input_message}
            ]
        else:
            messages = input_message
        
        current_agent = self
        
        for turn in range(max_turns):
            # Prepare system message
            system_message = current_agent.instructions
            
            # Make API call
            response = await current_agent.client.messages.create(
                model=current_agent.model,
                max_tokens=current_agent.max_tokens,
                temperature=current_agent.temperature,
                system=system_message,
                messages=messages,
                tools=current_agent._anthropic_tools if current_agent._anthropic_tools else None
            )
            
            # Add assistant message
            messages.append({
                "role": "assistant",
                "content": response.content
            })
            
            # Process tool calls
            tool_results, handoff_agent, is_complete = await current_agent._process_tool_calls(response)
            
            # Handle handoff
            if handoff_agent:
                current_agent = handoff_agent
                continue
                
            # Add tool results if any
            if tool_results:
                messages.extend(tool_results)
                continue
                
            # Check for completion
            if response.content and len(response.content) > 0:
                final_content = ""
                for content_block in response.content:
                    if content_block.type == "text":
                        final_content += content_block.text
                        
                # If no tools were called and we have text content, we're done
                if final_content and not any(block.type == "tool_use" for block in response.content):
                    return AgentResult(
                        final_output=final_content,
                        messages=messages,
                        agent_name=current_agent.name
                    )
                    
        # If we've reached max turns, return the last response
        final_content = ""
        if messages and messages[-1]["role"] == "assistant":
            content = messages[-1]["content"]
            if isinstance(content, list):
                for block in content:
                    if hasattr(block, 'text'):
                        final_content += block.text
                    elif isinstance(block, dict) and block.get('type') == 'text':
                        final_content += block.get('text', '')
            elif isinstance(content, str):
                final_content = content
                
        return AgentResult(
            final_output=final_content,
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
            
            # Make streaming API call
            stream = await current_agent.client.messages.create(
                model=current_agent.model,
                max_tokens=current_agent.max_tokens,
                temperature=current_agent.temperature,
                system=system_message,
                messages=messages,
                #tools=current_agent._anthropic_tools if current_agent._anthropic_tools else NotGiven,
                stream=True
            )
            
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