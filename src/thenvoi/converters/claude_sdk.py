"""Claude SDK history converter."""

from __future__ import annotations

import logging
from typing import Any

from thenvoi.core.protocols import HistoryConverter

logger = logging.getLogger(__name__)


class ClaudeSDKHistoryConverter(HistoryConverter[str]):
    """
    Converts platform history to Claude SDK text format.

    Claude SDK uses text context rather than structured messages.
    This converter formats history as a text block for context injection.

    Handles:
    - Skipping this agent's text messages (redundant with tool events)
    - Including other agents' text messages with [name]: prefix
    - Including tool_call events as raw JSON (as stored by platform)
    - Including tool_result events as raw JSON (as stored by platform)

    Output example:
        [Alice]: What's the weather?
        {"name": "get_weather", "args": {"location": "NYC"}, "tool_call_id": "toolu_123"}
        {"output": {"temperature": 72}, "tool_call_id": "toolu_123"}
        [Other Agent]: I can help too!
    """

    def __init__(self, agent_name: str = ""):
        """
        Initialize converter.

        Args:
            agent_name: Name of this agent. Text messages from this agent
                       are skipped (redundant with tool events).
        """
        self._agent_name = agent_name

    def set_agent_name(self, name: str) -> None:
        """
        Set agent name so converter knows which messages to skip.

        Args:
            name: Name of this agent
        """
        self._agent_name = name

    def convert(self, raw: list[dict[str, Any]]) -> str:
        """Convert platform history to text format for Claude SDK."""
        if not raw:
            return ""

        lines: list[str] = []

        for hist in raw:
            message_type = hist.get("message_type", "text")
            content = hist.get("content", "")
            role = hist.get("role", "user")
            sender_name = hist.get("sender_name", "Unknown")

            if message_type == "text":
                # Skip own text (redundant with tool results)
                if role == "assistant" and sender_name == self._agent_name:
                    logger.debug(f"Skipping own message: {content[:50]}...")
                    continue
                # Include user and other agents' messages
                if content:
                    lines.append(f"[{sender_name}]: {content}")

            elif message_type == "tool_call":
                # Include raw tool_call JSON as-is
                if content:
                    lines.append(content)

            elif message_type == "tool_result":
                # Include raw tool_result JSON as-is
                if content:
                    lines.append(content)

        return "\n".join(lines) if lines else ""
