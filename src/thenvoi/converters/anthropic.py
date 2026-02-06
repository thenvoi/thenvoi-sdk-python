"""Anthropic history converter."""

from __future__ import annotations

from typing import Any

from thenvoi.core.protocols import HistoryConverter

from ._tool_parsing import parse_tool_call, parse_tool_result

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


def _flush_pending_tool_results(
    messages: AnthropicMessages, pending_tool_results: list[dict[str, Any]]
) -> None:
    """Flush pending tool results into a single user message.

    Per Anthropic docs, all tool results must be in a single user message
    to enable parallel tool use. Sending separate messages "teaches Claude
    to avoid parallel calls."
    """
    if pending_tool_results:
        messages.append(
            {
                "role": "user",
                "content": list(pending_tool_results),
            }
        )
        pending_tool_results.clear()


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
    - tool_result: {"name": "...", "output": "...", "tool_call_id": "...", "is_error": bool}
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
        # Collect tool results to batch them into a single user message
        pending_tool_results: list[dict[str, Any]] = []

        for hist in raw:
            message_type = hist.get("message_type", "text")
            content = hist.get("content", "")

            if message_type == "tool_call":
                # Flush pending tool results before starting new tool calls
                _flush_pending_tool_results(messages, pending_tool_results)

                # Parse tool call JSON and collect for batching
                parsed = parse_tool_call(content)
                if parsed:
                    tool_use_block = {
                        "type": "tool_use",
                        "id": parsed.tool_call_id,
                        "name": parsed.name,
                        "input": parsed.args,
                    }
                    pending_tool_calls.append(tool_use_block)

            elif message_type == "tool_result":
                # Flush pending tool calls first (tool results follow tool calls)
                _flush_pending_tool_calls(messages, pending_tool_calls)

                # Parse tool result JSON and collect for batching
                parsed = parse_tool_result(content)
                if parsed:
                    tool_result_block: dict[str, Any] = {
                        "type": "tool_result",
                        "tool_use_id": parsed.tool_call_id,
                        "content": parsed.output,
                    }
                    # Only include is_error if True (Anthropic API expects this)
                    if parsed.is_error:
                        tool_result_block["is_error"] = True
                    pending_tool_results.append(tool_result_block)

            elif message_type == "text":
                # Flush pending tool calls and results first
                _flush_pending_tool_calls(messages, pending_tool_calls)
                _flush_pending_tool_results(messages, pending_tool_results)

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

        # Flush any remaining pending tool calls and results
        _flush_pending_tool_calls(messages, pending_tool_calls)
        _flush_pending_tool_results(messages, pending_tool_results)

        return messages
