"""
Normalized message types for cross-framework history reconstruction.

These types represent the "common denominator" between the platform's
storage format and what various LLM frameworks need. The flow is:

    Platform History → Normalized Messages → Framework-Specific Messages
    (list[dict])       (list[NormalizedMessage])   (Anthropic/LangGraph/etc)

This separation allows:
- Single source of truth for tool call/result pairing logic
- Framework adapters that are simple mappers
- Easy testing of pairing logic without framework dependencies
- Clean extension to new frameworks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class NormalizedUserText:
    """A user text message."""

    type: Literal["user_text"] = field(default="user_text", init=False)
    sender_name: str
    content: str


@dataclass
class NormalizedToolExchange:
    """
    A paired tool call + result.

    Represents a complete tool invocation: the LLM called a tool,
    and we have the result. These are always paired because that's
    what LLM APIs expect in conversation history.
    """

    type: Literal["tool_exchange"] = field(default="tool_exchange", init=False)
    tool_name: str
    tool_id: str  # run_id or generated ID for pairing
    input_args: dict[str, Any]
    output: str
    is_error: bool = False


@dataclass
class NormalizedSystemMessage:
    """A system message (e.g., participant updates)."""

    type: Literal["system"] = field(default="system", init=False)
    content: str


# Union type for all normalized messages
NormalizedMessage = (
    NormalizedUserText | NormalizedToolExchange | NormalizedSystemMessage
)
