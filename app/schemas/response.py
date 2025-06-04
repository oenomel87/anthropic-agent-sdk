from dataclasses import dataclass, field
from typing import List
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
    messages: List[MessageParam] = field(default_factory=list)