"""
Platform history parser - converts raw platform history to normalized messages.

This is the single source of truth for:
- Parsing tool call/result JSON events
- Pairing tool calls with their results (run_id primary, name-LIFO fallback)
- Filtering message types (skip assistant text, internal types)

The pairing strategy handles:
- Out-of-order results
- Back-to-back calls to the same tool
- Parallel tool calls
- Missing matches (warns but doesn't fabricate)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .normalized import (
    NormalizedMessage,
    NormalizedToolExchange,
    NormalizedUserText,
)

logger = logging.getLogger(__name__)


def parse_platform_history(
    history: list[dict[str, Any]],
) -> list[NormalizedMessage]:
    """
    Parse platform history into normalized messages.

    Args:
        history: Raw platform history from API. Each item has:
            - message_type: "text" | "tool_call" | "tool_result" | ...
            - content: Message content (JSON for tool events)
            - role: "user" | "assistant"
            - sender_name: Display name

    Returns:
        List of normalized messages ready for framework adapters.

    Example:
        >>> history = [
        ...     {"message_type": "text", "role": "user", "sender_name": "Alice", "content": "Hi"},
        ...     {"message_type": "tool_call", "content": '{"run_id": "abc", "name": "greet", "data": {"input": {}}}'},
        ...     {"message_type": "tool_result", "content": '{"run_id": "abc", "name": "greet", "data": {"output": "Hello!"}}'},
        ... ]
        >>> messages = parse_platform_history(history)
        >>> [m.type for m in messages]
        ['user_text', 'tool_exchange']
    """
    messages: list[NormalizedMessage] = []

    # Pending tool calls for pairing
    # Primary: keyed by run_id for O(1) lookup
    pending_by_run_id: dict[str, _PendingToolCall] = {}
    # Fallback: stack per tool name for LIFO matching
    pending_by_name: dict[str, list[_PendingToolCall]] = {}

    for item in history:
        message_type = item.get("message_type", "text")
        content = item.get("content", "")
        role = item.get("role")
        sender_name = item.get("sender_name", "Unknown")

        if message_type == "tool_call":
            _handle_tool_call(content, pending_by_run_id, pending_by_name)

        elif message_type == "tool_result":
            exchange = _handle_tool_result(content, pending_by_run_id, pending_by_name)
            if exchange:
                messages.append(exchange)

        elif message_type == "text":
            if role == "assistant":
                # Skip assistant text - redundant with tool_call/tool_result
                # Including it would teach LLM to respond with text instead of tools
                logger.debug(f"Skipping redundant assistant text: {content[:50]}")
            else:
                messages.append(
                    NormalizedUserText(sender_name=sender_name, content=content)
                )

        # Skip other message types (thought, error, task, etc.)

    # Warn about unmatched tool calls
    _warn_unmatched(pending_by_run_id, pending_by_name)

    return messages


class _PendingToolCall:
    """Internal: holds parsed tool call waiting for its result."""

    __slots__ = ("run_id", "tool_name", "input_args")

    def __init__(self, run_id: str | None, tool_name: str, input_args: dict[str, Any]):
        self.run_id = run_id
        self.tool_name = tool_name
        self.input_args = input_args


def _handle_tool_call(
    content: str,
    pending_by_run_id: dict[str, _PendingToolCall],
    pending_by_name: dict[str, list[_PendingToolCall]],
) -> None:
    """Parse and store a pending tool call."""
    try:
        event = json.loads(content)
        run_id = event.get("run_id")
        tool_name = event.get("name", "unknown")
        input_args = event.get("data", {}).get("input", {})

        pending = _PendingToolCall(run_id, tool_name, input_args)

        if run_id:
            pending_by_run_id[run_id] = pending
        else:
            # Fallback: store by name as stack for LIFO matching
            if tool_name not in pending_by_name:
                pending_by_name[tool_name] = []
            pending_by_name[tool_name].append(pending)

    except json.JSONDecodeError:
        logger.warning(f"Failed to parse tool_call: {content[:100]}")


def _handle_tool_result(
    content: str,
    pending_by_run_id: dict[str, _PendingToolCall],
    pending_by_name: dict[str, list[_PendingToolCall]],
) -> NormalizedToolExchange | None:
    """
    Parse tool result and pair with pending call.

    Returns NormalizedToolExchange if matched, None if no match found.
    """
    try:
        event = json.loads(content)
        tool_name = event.get("name", "unknown")
        run_id = event.get("run_id")
        output = event.get("data", {}).get("output", "")
        is_error = event.get("data", {}).get("is_error", False)

        # Find matching pending tool call
        matching: _PendingToolCall | None = None

        # Primary: match by run_id (most reliable)
        if run_id and run_id in pending_by_run_id:
            matching = pending_by_run_id.pop(run_id)

        # Fallback: match by name (LIFO - pop from end of stack)
        if not matching and tool_name in pending_by_name:
            stack = pending_by_name[tool_name]
            if stack:
                matching = stack.pop()
                if not stack:
                    del pending_by_name[tool_name]

        if matching:
            # Generate tool_id: prefer run_id, fallback to name-based
            tool_id = run_id or f"tool_{tool_name}"

            return NormalizedToolExchange(
                tool_name=tool_name,
                tool_id=tool_id,
                input_args=matching.input_args,
                output=str(output),
                is_error=is_error,
            )
        else:
            logger.warning(
                f"tool_result without matching tool_call: name={tool_name}, run_id={run_id}"
            )
            return None

    except json.JSONDecodeError:
        logger.warning(f"Failed to parse tool_result: {content[:100]}")
        return None


def _warn_unmatched(
    pending_by_run_id: dict[str, _PendingToolCall],
    pending_by_name: dict[str, list[_PendingToolCall]],
) -> None:
    """Log warnings for any unmatched tool calls."""
    unmatched_count = len(pending_by_run_id) + sum(
        len(stack) for stack in pending_by_name.values()
    )

    if not unmatched_count:
        return

    logger.warning(f"Found {unmatched_count} tool_calls without matching tool_results")

    for run_id, pending in pending_by_run_id.items():
        logger.warning(
            f"Unmatched tool_call: name={pending.tool_name}, run_id={run_id}"
        )

    for name, stack in pending_by_name.items():
        for pending in stack:
            logger.warning(
                f"Unmatched tool_call: name={name}, input={pending.input_args}"
            )
