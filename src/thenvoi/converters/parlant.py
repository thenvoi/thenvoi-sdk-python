"""Parlant history converter."""

from __future__ import annotations

from typing import Any

from thenvoi.core.protocols import HistoryConverter

# Type alias for Parlant messages (simple dict format)
ParlantMessages = list[dict[str, Any]]


class ParlantHistoryConverter(HistoryConverter[ParlantMessages]):
    """
    Converts platform history to Parlant message format.

    Output: [{"role": "user", "content": "...", "sender": "..."}]

    Note:
    - Only converts text messages (tool_call/tool_result events are skipped)
    - User messages are prefixed with sender name for context
    - Other agents' messages are included with role "assistant" and sender info
    - This agent's messages are skipped (redundant with Parlant's internal state)
    """

    def __init__(self, agent_name: str = ""):
        """
        Initialize converter.

        Args:
            agent_name: Name of this agent. Messages from this agent are skipped
                       (they're redundant with tool results). Messages from other
                       agents are included with their sender info.
        """
        self._agent_name = agent_name

    def set_agent_name(self, name: str) -> None:
        """
        Set agent name so converter knows which messages to skip.

        Args:
            name: Name of this agent
        """
        self._agent_name = name

    def convert(self, raw: list[dict[str, Any]]) -> ParlantMessages:
        """Convert platform history to Parlant format."""
        messages: ParlantMessages = []

        for hist in raw:
            message_type = hist.get("message_type", "text")

            # Only convert text messages
            if message_type != "text":
                continue

            role = hist.get("role", "user")
            content = hist.get("content", "")
            sender_name = hist.get("sender_name", "")
            sender_type = hist.get("sender_type", "User")

            if role == "assistant" and sender_name == self._agent_name:
                # Skip THIS agent's text (redundant with Parlant state)
                continue
            elif role == "assistant":
                # Other agents' messages
                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "sender": sender_name,
                        "sender_type": sender_type,
                    }
                )
            else:
                # User messages
                messages.append(
                    {
                        "role": "user",
                        "content": f"[{sender_name}]: {content}"
                        if sender_name
                        else content,
                        "sender": sender_name,
                        "sender_type": sender_type,
                    }
                )

        return messages
