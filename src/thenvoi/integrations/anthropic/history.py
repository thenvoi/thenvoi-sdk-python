"""
Anthropic conversation history management.

Handles per-room message history storage, platform history conversion,
and message batching for Anthropic API requirements.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any, Literal

from anthropic.types import (
    ContentBlock,
    MessageParam,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)

logger = logging.getLogger(__name__)

# Type aliases
Role = Literal["user", "assistant"]
# ContentBlockParam types are for building requests
ContentBlockParam = TextBlockParam | ToolUseBlockParam | ToolResultBlockParam
# ContentBlock is the response type - MessageParam.content accepts both
ContentBlockTypes = ContentBlockParam | ContentBlock


class AnthropicHistoryManager:
    """
    Manages per-room conversation history for Anthropic SDK.

    Responsibilities:
    - Store and retrieve message history per room
    - Convert platform history to Anthropic message format
    - Batch consecutive same-role messages (Anthropic requires alternating roles)
    """

    def __init__(self) -> None:
        self._history: dict[str, list[MessageParam]] = {}

    def get_history(self, room_id: str) -> list[MessageParam]:
        """Get message history for a room."""
        return self._history.get(room_id, [])

    def add_message(
        self,
        room_id: str,
        role: Role,
        content: str | Sequence[ContentBlockTypes],
    ) -> None:
        """
        Add a message to room history.

        Args:
            room_id: Room identifier
            role: Message role ("user" or "assistant")
            content: Message content (str or list of content blocks)
        """
        if room_id not in self._history:
            self._history[room_id] = []
        self._history[room_id].append({"role": role, "content": content})

    def initialize_room(
        self, room_id: str, platform_history: list[dict[str, Any]] | None
    ) -> None:
        """
        Initialize room history, optionally loading platform history.

        Args:
            room_id: Room identifier
            platform_history: Optional platform history to convert and load
        """
        if platform_history:
            self._history[room_id] = self._convert_platform_history(platform_history)
            logger.info(
                f"Room {room_id}: Loaded {len(platform_history)} historical messages"
            )
        else:
            self._history[room_id] = []
            logger.info(f"Room {room_id}: No historical messages found")

    def ensure_room_exists(self, room_id: str) -> None:
        """Ensure room history exists (safety for non-first messages)."""
        if room_id not in self._history:
            self._history[room_id] = []

    def clear_room(self, room_id: str) -> None:
        """Clear message history for a room."""
        if room_id in self._history:
            del self._history[room_id]
            logger.debug(f"Room {room_id}: Cleaned up message history")

    def _convert_platform_history(
        self, platform_history: list[dict[str, Any]]
    ) -> list[MessageParam]:
        """
        Convert platform history to Anthropic message format.

        Handles message_type: "text", "tool_call", "tool_result"
        Pairs tool_calls with tool_results using run_id (primary) or name (LIFO fallback).

        Platform history format:
            {"message_type": "text"|"tool_call"|"tool_result", "content": str, ...}

        Anthropic format:
            {"role": "user"|"assistant", "content": str | list[ContentBlock]}
        """
        messages: list[MessageParam] = []

        # Pending tool calls for pairing (matching LangGraph strategy)
        pending_by_run_id: dict[str, dict[str, Any]] = {}
        pending_by_name: dict[str, list[dict[str, Any]]] = {}

        for hist in platform_history:
            message_type = hist.get("message_type", "text")
            content = hist.get("content", "")
            role = hist.get("role")
            sender_name = hist.get("sender_name", "Unknown")

            if message_type == "tool_call":
                # Parse and store pending tool call
                try:
                    event = json.loads(content)
                    run_id = event.get("run_id")
                    tool_name = event.get("name", "unknown")

                    if run_id:
                        pending_by_run_id[run_id] = event
                    else:
                        # Fallback: store by name (LIFO stack)
                        if tool_name not in pending_by_name:
                            pending_by_name[tool_name] = []
                        pending_by_name[tool_name].append(event)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tool_call: {content[:100]}")

            elif message_type == "tool_result":
                # Match to pending tool_call, emit assistant + user messages
                try:
                    event = json.loads(content)
                    tool_name = event.get("name", "unknown")
                    run_id = event.get("run_id")
                    output = event.get("data", {}).get("output", "")
                    is_error = event.get("data", {}).get("is_error", False)

                    # Use run_id as tool_use_id (matches how we store it)
                    tool_use_id = run_id or f"tool_{tool_name}"

                    # Find matching pending tool_call
                    matching_call = None

                    # Primary: match by run_id
                    if run_id and run_id in pending_by_run_id:
                        matching_call = pending_by_run_id.pop(run_id)

                    # Fallback: match by name (LIFO)
                    if not matching_call and tool_name in pending_by_name:
                        if pending_by_name[tool_name]:
                            matching_call = pending_by_name[tool_name].pop()

                    if matching_call:
                        tool_input = matching_call.get("data", {}).get("input", {})

                        # Emit assistant message with tool_use block
                        tool_use_block: ToolUseBlockParam = {
                            "type": "tool_use",
                            "id": tool_use_id,
                            "name": tool_name,
                            "input": tool_input,
                        }
                        messages.append(
                            {"role": "assistant", "content": [tool_use_block]}
                        )
                    else:
                        logger.warning(
                            f"tool_result without matching tool_call: "
                            f"name={tool_name}, run_id={run_id}"
                        )

                    # Emit user message with tool_result block
                    tool_result_block: ToolResultBlockParam = {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": str(output),
                        "is_error": is_error,
                    }
                    messages.append({"role": "user", "content": [tool_result_block]})

                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tool_result: {content[:100]}")

            elif message_type == "text":
                if role == "assistant":
                    # Skip assistant text - redundant with tool_call/tool_result
                    logger.debug(f"Skipping redundant assistant text: {content[:50]}")
                else:
                    # User text message
                    formatted = f"[{sender_name}]: {content}"
                    messages.append({"role": "user", "content": formatted})

            # Skip other message types (thought, error, task, etc.)

        # Warn about unmatched tool calls
        unmatched = len(pending_by_run_id) + sum(
            len(v) for v in pending_by_name.values()
        )
        if unmatched:
            logger.warning(
                f"Found {unmatched} tool_calls without matching tool_results"
            )

        return self._batch_consecutive_messages(messages)

    def _batch_consecutive_messages(
        self, messages: list[MessageParam]
    ) -> list[MessageParam]:
        """
        Batch consecutive same-role messages into single messages.

        Anthropic requires alternating user/assistant roles. This merges
        consecutive messages with the same role into a single message
        with multiple content blocks.
        """
        if not messages:
            return messages

        batched: list[MessageParam] = []
        current: MessageParam | None = None

        for msg in messages:
            if current and current["role"] == msg["role"]:
                # Same role - merge content blocks
                current_content = current["content"]
                msg_content = msg["content"]

                # Normalize to lists
                if isinstance(current_content, str):
                    text_block: TextBlockParam = {
                        "type": "text",
                        "text": current_content,
                    }
                    current = {"role": current["role"], "content": [text_block]}
                if isinstance(msg_content, str):
                    msg_text_block: TextBlockParam = {
                        "type": "text",
                        "text": msg_content,
                    }
                    msg_content = [msg_text_block]

                # Now current["content"] is a list, extend it
                current_content_list = current["content"]
                if isinstance(current_content_list, list):
                    current_content_list.extend(msg_content)
            else:
                # Different role - start new message
                if current:
                    batched.append(current)
                current = {"role": msg["role"], "content": msg["content"]}

        if current:
            batched.append(current)

        return batched
