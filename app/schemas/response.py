from dataclasses import dataclass, field
from typing import List, Literal
from anthropic.types import  MessageParam


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
    type: Literal[
        "text_start",
        "text",
        "text_stop",
        "thinking_start",
        "thinking",
        "thinking_stop",
        "tool_use_start",
        "tool_use",
        "tool_use_stop",
        "tool_result_start",
        "tool_result",
        "tool_result_stop",
        "error"
    ] = "text"