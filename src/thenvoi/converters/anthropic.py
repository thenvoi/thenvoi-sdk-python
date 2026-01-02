"""Anthropic history converter."""

from __future__ import annotations

from typing import Any

from thenvoi.core.protocols import HistoryConverter

# Type alias for Anthropic messages
AnthropicMessages = list[dict[str, str]]


class AnthropicHistoryConverter(HistoryConverter[AnthropicMessages]):
    """
    Converts platform history to Anthropic message format.

    Output: [{"role": "user", "content": "..."}]

    Note:
    - Only converts text messages (tool_call/tool_result events are skipped)
    - User messages are prefixed with sender name: "[Alice]: Hello"
    - Other agents' messages are included as user messages with [name] prefix
    - This agent's messages are skipped (redundant with tool results)
    """

    def __init__(self, agent_name: str = ""):
        """
        Initialize converter.

        Args:
            agent_name: Name of this agent. Messages from this agent are skipped
                       (they're redundant with tool results). Messages from other
                       agents are included as user messages.
        """
        self._agent_name = agent_name

    def set_agent_name(self, name: str) -> None:
        """
        Set agent name so converter knows which messages to skip.

        Args:
            name: Name of this agent
        """
        self._agent_name = name

    def convert(self, raw: list[dict[str, Any]]) -> AnthropicMessages:
        """Convert platform history to Anthropic format."""
        messages: AnthropicMessages = []

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
                messages.append(
                    {
                        "role": "user",
                        "content": f"[{sender_name}]: {content}"
                        if sender_name
                        else content,
                    }
                )

        return messages
