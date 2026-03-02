"""Anthropic history converter."""

from __future__ import annotations

import logging
from typing import Any

from thenvoi.core.protocols import HistoryConverter
from thenvoi.converters.normalized_events import (
    ToolCallHistoryEvent,
    ToolResultHistoryEvent,
    TextHistoryEvent,
    normalize_history_events,
)

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


def _patch_orphaned_tool_uses(messages: AnthropicMessages) -> None:
    """Inject synthetic tool_result blocks for orphaned tool_use blocks.

    The Anthropic API requires every ``tool_use`` block in an assistant
    message to have a corresponding ``tool_result`` in the next user
    message.  When history is corrupted (e.g. a 403 killed tool
    execution), some tool_use blocks may lack results.  This function
    walks the converted messages and injects error tool_results so the
    history is valid.

    Mutations happen in-place via list insertion.
    """
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
            i += 1
            continue

        # Collect tool_use IDs in this assistant message
        tool_use_ids = {
            block["id"]
            for block in msg["content"]
            if isinstance(block, dict) and block.get("type") == "tool_use"
        }

        if not tool_use_ids:
            i += 1
            continue

        # Check the next message for matching tool_results
        next_msg = messages[i + 1] if i + 1 < len(messages) else None
        matched_ids: set[str] = set()

        if (
            next_msg
            and next_msg.get("role") == "user"
            and isinstance(next_msg.get("content"), list)
        ):
            matched_ids = {
                block["tool_use_id"]
                for block in next_msg["content"]
                if isinstance(block, dict)
                and block.get("type") == "tool_result"
                and block.get("tool_use_id") in tool_use_ids
            }

        orphaned_ids = tool_use_ids - matched_ids

        if orphaned_ids:
            sorted_ids = sorted(orphaned_ids)
            logger.warning(
                "Patching %d orphaned tool_use block(s): %s",
                len(sorted_ids),
                sorted_ids,
            )
            synthetic_results = [
                {
                    "type": "tool_result",
                    "tool_use_id": uid,
                    "content": "Error: tool execution was interrupted",
                    "is_error": True,
                }
                for uid in sorted_ids
            ]

            if next_msg is not None and next_msg.get("role") == "user":
                # Merge synthetics into the existing user message to
                # preserve the alternating-turns contract.
                if isinstance(next_msg["content"], str):
                    next_msg["content"] = [
                        {"type": "text", "text": next_msg["content"]}
                    ]
                next_msg["content"] = synthetic_results + next_msg["content"]
            else:
                # No user message follows — insert a new one.
                messages.insert(
                    i + 1,
                    {"role": "user", "content": synthetic_results},
                )

        i += 1


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

        for event in normalize_history_events(raw):
            if isinstance(event, ToolCallHistoryEvent):
                # Flush pending tool results before starting new tool calls
                _flush_pending_tool_results(messages, pending_tool_results)

                tool_use_block = {
                    "type": "tool_use",
                    "id": event.tool_call_id,
                    "name": event.name,
                    "input": event.args,
                }
                pending_tool_calls.append(tool_use_block)

            elif isinstance(event, ToolResultHistoryEvent):
                # Flush pending tool calls first (tool results follow tool calls)
                _flush_pending_tool_calls(messages, pending_tool_calls)

                tool_result_block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": event.tool_call_id,
                    "content": event.output,
                }
                if event.is_error:
                    tool_result_block["is_error"] = True
                pending_tool_results.append(tool_result_block)

            elif isinstance(event, TextHistoryEvent):
                # Flush pending tool calls and results first
                _flush_pending_tool_calls(messages, pending_tool_calls)
                _flush_pending_tool_results(messages, pending_tool_results)

                if event.role == "assistant" and event.sender_name == self._agent_name:
                    # Skip THIS agent's text (redundant with tool results)
                    continue
                else:
                    # User messages AND other agents' messages
                    messages.append(
                        {
                            "role": "user",
                            "content": f"[{event.sender_name}]: {event.content}"
                            if event.sender_name
                            else event.content,
                        }
                    )

        # Flush any remaining pending tool calls and results
        _flush_pending_tool_calls(messages, pending_tool_calls)
        _flush_pending_tool_results(messages, pending_tool_results)

        # Sanitize orphaned tool_use blocks: every assistant message with
        # tool_use blocks must be followed by a user message containing
        # matching tool_result blocks.  Orphans arise when tool execution
        # was interrupted (e.g. a 403 on the events endpoint killed the
        # processing loop before results were recorded).
        _patch_orphaned_tool_uses(messages)

        return messages
