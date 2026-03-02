"""CrewAI history converter."""

from __future__ import annotations

from typing import Any

from thenvoi.core.protocols import HistoryConverter
from thenvoi.converters.normalized_events import TextHistoryEvent, normalize_history_events

# Type alias for CrewAI messages (simple dict format)
CrewAIMessages = list[dict[str, Any]]


class CrewAIHistoryConverter(HistoryConverter[CrewAIMessages]):
    """
    Converts platform history to CrewAI-compatible message format.

    Output: [{"role": "user", "content": "...", "sender": "..."}]

    Note:
    - Only converts text messages (tool_call/tool_result events are skipped)
    - User messages include sender name for context
    - Other agents' messages are included with role "assistant"
    - This agent's messages are skipped (redundant with CrewAI's internal state)
    """

    def __init__(self, agent_name: str = ""):
        """
        Initialize converter.

        Args:
            agent_name: Name of this agent. Messages from this agent are skipped
                       (they're redundant with internal state). Messages from other
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

    def convert(self, raw: list[dict[str, Any]]) -> CrewAIMessages:
        """Convert platform history to CrewAI format."""
        messages: CrewAIMessages = []

        for event in normalize_history_events(raw):
            if not isinstance(event, TextHistoryEvent):
                continue

            if event.role == "assistant" and event.sender_name == self._agent_name:
                # Skip THIS agent's text (redundant with CrewAI state)
                continue
            elif event.role == "assistant":
                # Other agents' messages
                messages.append(
                    {
                        "role": "assistant",
                        "content": event.content,
                        "sender": event.sender_name,
                        "sender_type": event.sender_type,
                    }
                )
            else:
                # User messages
                messages.append(
                    {
                        "role": "user",
                        "content": f"[{event.sender_name}]: {event.content}"
                        if event.sender_name
                        else event.content,
                        "sender": event.sender_name,
                        "sender_type": event.sender_type,
                    }
                )

        return messages
