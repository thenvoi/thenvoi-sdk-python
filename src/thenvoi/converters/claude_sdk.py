"""Claude SDK history converter."""

from __future__ import annotations

from typing import Any

from thenvoi.core.protocols import HistoryConverter


class ClaudeSDKHistoryConverter(HistoryConverter[str]):
    """
    Converts platform history to Claude SDK text format.

    Claude SDK uses text context rather than structured messages.
    This converter formats history as a human-readable text block
    with sender names and room_id context.

    Output: "Previous conversation history:\\n[room_id: xxx][Alice]: Hello\\n..."

    Args:
        room_id: Room ID to include in context prefix
    """

    def __init__(self, room_id: str = ""):
        self.room_id = room_id

    def convert(self, raw: list[dict[str, Any]]) -> str:
        """Convert platform history to text format for Claude SDK."""
        if not raw:
            return ""

        lines: list[str] = []
        room_context = f"[room_id: {self.room_id}]" if self.room_id else ""

        for hist in raw:
            message_type = hist.get("message_type", "text")

            # Only include text messages in context
            if message_type != "text":
                continue

            sender_name = hist.get("sender_name", "Unknown")
            content = hist.get("content", "")

            if content:
                lines.append(f"{room_context}[{sender_name}]: {content}")

        if not lines:
            return ""

        return "Previous conversation history:\n" + "\n".join(lines)
