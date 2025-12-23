"""
Anthropic conversation history management.

Handles per-room message history storage and platform history conversion.
Uses the shared history parser for tool call/result pairing.
"""

from __future__ import annotations

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

from thenvoi.integrations.history import parse_platform_history
from thenvoi.integrations.history.adapters.anthropic import to_anthropic_messages

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
    - Convert platform history to Anthropic message format (via shared parser)
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

        Uses the shared history parser for tool call/result pairing,
        then the Anthropic adapter for format conversion.
        """
        normalized = parse_platform_history(platform_history)
        return to_anthropic_messages(normalized)
