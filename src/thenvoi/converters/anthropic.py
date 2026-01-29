"""Anthropic history converter."""

from __future__ import annotations

import json
import logging
from typing import Any

from thenvoi.core.protocols import HistoryConverter

logger = logging.getLogger(__name__)

# Type alias for Anthropic messages (can have structured content)
AnthropicMessages = list[dict[str, Any]]


def _flush_pending_tool_calls(
    messages: AnthropicMessages, pending_tool_calls: list[dict[str, Any]]
) -> None:
    """Flush pending tool calls into a single assistant message."""
    if pending_tool_calls:
        messages.append(
            {
                "role": "assistant",
                "content": list(pending_tool_calls),
            }
        )
        pending_tool_calls.clear()


class AnthropicHistoryConverter(HistoryConverter[AnthropicMessages]):
    """
    Converts platform history to Anthropic message format.

    Output: [{"role": "user"|"assistant", "content": "..." | [...]}]

    Handles:
    - text messages: User messages with [name] prefix, other agents as user messages
    - tool_call: Assistant message with tool_use content blocks
    - tool_result: User message with tool_result content blocks
    - This agent's text messages are skipped (redundant with tool results)

    Tool events are stored in platform as JSON:
    - tool_call: {"name": "...", "args": {...}, "tool_call_id": "..."}
    - tool_result: {"name": "...", "output": "...", "tool_call_id": "..."}
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
        # Collect tool calls to batch them into a single assistant message
        pending_tool_calls: list[dict[str, Any]] = []

        for hist in raw:
            message_type = hist.get("message_type", "text")
            content = hist.get("content", "")

            if message_type == "tool_call":
                # Parse tool call JSON and collect for batching
                try:
                    event = json.loads(content)
                    tool_call_id = event.get("tool_call_id")
                    tool_name = event.get("name")
                    # Skip events with missing required fields
                    if not tool_call_id:
                        logger.warning(
                            "Skipping tool_call with missing tool_call_id: %s",
                            content[:100],
                        )
                        continue
                    if not tool_name:
                        logger.warning(
                            "Skipping tool_call with missing name: %s", content[:100]
                        )
                        continue
                    tool_use_block = {
                        "type": "tool_use",
                        "id": tool_call_id,
                        "name": tool_name,
                        "input": event.get("args", {}),
                    }
                    pending_tool_calls.append(tool_use_block)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse tool_call: %s", content[:100])

            elif message_type == "tool_result":
                # Flush pending tool calls first
                _flush_pending_tool_calls(messages, pending_tool_calls)

                # Parse tool result JSON
                try:
                    event = json.loads(content)
                    tool_call_id = event.get("tool_call_id")
                    tool_name = event.get("name")
                    # Skip events with missing required fields
                    if not tool_call_id:
                        logger.warning(
                            "Skipping tool_result with missing tool_call_id: %s",
                            content[:100],
                        )
                        continue
                    if not tool_name:
                        logger.warning(
                            "Skipping tool_result with missing name: %s", content[:100]
                        )
                        continue
                    tool_result_block = {
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": str(event.get("output", "")),
                    }
                    messages.append(
                        {
                            "role": "user",
                            "content": [tool_result_block],
                        }
                    )
                except json.JSONDecodeError:
                    logger.warning("Failed to parse tool_result: %s", content[:100])

            elif message_type == "text":
                # Flush pending tool calls first
                _flush_pending_tool_calls(messages, pending_tool_calls)

                role = hist.get("role", "user")
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

        # Flush any remaining pending tool calls
        _flush_pending_tool_calls(messages, pending_tool_calls)

        return messages
