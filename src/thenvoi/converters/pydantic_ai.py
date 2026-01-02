"""Pydantic AI history converter."""

from __future__ import annotations

from typing import Any

try:
    from pydantic_ai.messages import (
        ModelRequest,
        UserPromptPart,
    )
except ImportError as e:
    raise ImportError(
        "Pydantic AI dependencies not installed. "
        "Install with: uv add thenvoi-sdk[pydantic_ai]"
    ) from e

from thenvoi.core.protocols import HistoryConverter

# Type alias for Pydantic AI messages
PydanticAIMessages = list[ModelRequest]


class PydanticAIHistoryConverter(HistoryConverter[PydanticAIMessages]):
    """
    Converts platform history to Pydantic AI message format.

    Output:
    - user messages → ModelRequest with UserPromptPart
    - other agents' messages → ModelRequest with UserPromptPart (with [name] prefix)
    - this agent's messages → skipped (redundant with tool results)

    Note:
    - Only converts text messages (tool_call/tool_result events are skipped)
    - User messages are prefixed with sender name: "[Alice]: Hello"
    """

    def __init__(self, agent_name: str = ""):
        """
        Initialize converter.

        Args:
            agent_name: Name of this agent. Messages from this agent are skipped
                       (they're redundant with tool results). Messages from other
                       agents are included as ModelRequest.
        """
        self._agent_name = agent_name

    def set_agent_name(self, name: str) -> None:
        """
        Set agent name so converter knows which messages to skip.

        Args:
            name: Name of this agent
        """
        self._agent_name = name

    def convert(self, raw: list[dict[str, Any]]) -> PydanticAIMessages:
        """Convert platform history to Pydantic AI format."""
        messages: PydanticAIMessages = []

        for hist in raw:
            message_type = hist.get("message_type", "text")

            # Only convert text messages
            if message_type != "text":
                continue

            role = hist.get("role", "user")
            content = hist.get("content", "")
            sender_name = hist.get("sender_name", "")

            if role == "assistant" and sender_name == self._agent_name:
                # Skip THIS agent's text (redundant with tool results)
                continue
            else:
                # User messages AND other agents' messages
                formatted_content = (
                    f"[{sender_name}]: {content}" if sender_name else content
                )
                messages.append(
                    ModelRequest(parts=[UserPromptPart(content=formatted_content)])
                )

        return messages
