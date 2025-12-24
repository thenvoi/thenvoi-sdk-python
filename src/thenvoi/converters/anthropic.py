"""Anthropic history converter."""

from __future__ import annotations

from typing import Any

from thenvoi.core.protocols import HistoryConverter

# Type alias for Anthropic messages
AnthropicMessages = list[dict[str, str]]


class AnthropicHistoryConverter(HistoryConverter[AnthropicMessages]):
    """
    Converts platform history to Anthropic message format.

    Output: [{"role": "user"|"assistant", "content": "..."}]

    Note:
    - Only converts text messages (tool_call/tool_result events are skipped)
    - User messages are prefixed with sender name: "[Alice]: Hello"
    - Assistant messages are passed through as-is
    """

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

            if role == "assistant":
                messages.append({"role": "assistant", "content": content})
            else:
                # Prefix user messages with sender name
                messages.append(
                    {
                        "role": "user",
                        "content": f"[{sender_name}]: {content}"
                        if sender_name
                        else content,
                    }
                )

        return messages
