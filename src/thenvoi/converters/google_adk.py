"""Google ADK history converter."""

from __future__ import annotations

import logging
from typing import Any

from thenvoi.core.protocols import HistoryConverter

from ._tool_parsing import parse_tool_call, parse_tool_result

logger = logging.getLogger(__name__)

# Type alias for Google ADK messages (dict-based, converted to Content by adapter)
GoogleADKMessages = list[dict[str, Any]]


def _flush_pending_tool_calls(
    messages: GoogleADKMessages, pending_tool_calls: list[dict[str, Any]]
) -> None:
    """Flush pending tool calls into a single model message."""
    if pending_tool_calls:
        messages.append(
            {
                "role": "model",
                "content": list(pending_tool_calls),
            }
        )
        pending_tool_calls.clear()


def _flush_pending_tool_results(
    messages: GoogleADKMessages, pending_tool_results: list[dict[str, Any]]
) -> None:
    """Flush pending tool results into a single user message."""
    if pending_tool_results:
        messages.append(
            {
                "role": "user",
                "content": list(pending_tool_results),
            }
        )
        pending_tool_results.clear()


class GoogleADKHistoryConverter(HistoryConverter[GoogleADKMessages]):
    """
    Converts platform history to Google ADK message format.

    Output: [{"role": "user"|"model", "content": "..." | [...]}]

    Handles:
    - text messages: User messages with [name] prefix, other agents as user messages
    - tool_call: Model message with function_call content blocks
    - tool_result: User message with function_response content blocks
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

    def convert(self, raw: list[dict[str, Any]]) -> GoogleADKMessages:
        """Convert platform history to Google ADK format."""
        messages: GoogleADKMessages = []
        pending_tool_calls: list[dict[str, Any]] = []
        pending_tool_results: list[dict[str, Any]] = []

        for hist in raw:
            message_type = hist.get("message_type", "text")
            content = hist.get("content", "")

            if message_type == "tool_call":
                _flush_pending_tool_results(messages, pending_tool_results)

                parsed = parse_tool_call(content)
                if parsed:
                    tool_call_block = {
                        "type": "function_call",
                        "id": parsed.tool_call_id,
                        "name": parsed.name,
                        "args": parsed.args,
                    }
                    pending_tool_calls.append(tool_call_block)

            elif message_type == "tool_result":
                _flush_pending_tool_calls(messages, pending_tool_calls)

                parsed = parse_tool_result(content)
                if parsed:
                    tool_result_block: dict[str, Any] = {
                        "type": "function_response",
                        "tool_call_id": parsed.tool_call_id,
                        "name": parsed.name,
                        "output": parsed.output,
                    }
                    if parsed.is_error:
                        tool_result_block["is_error"] = True
                    pending_tool_results.append(tool_result_block)

            elif message_type == "text":
                _flush_pending_tool_calls(messages, pending_tool_calls)
                _flush_pending_tool_results(messages, pending_tool_results)

                role = hist.get("role", "user")
                sender_name = hist.get("sender_name", "")

                if role == "assistant" and sender_name == self._agent_name:
                    # Skip THIS agent's text (redundant with tool results)
                    continue
                else:
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
