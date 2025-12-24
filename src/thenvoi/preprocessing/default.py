"""Default preprocessor - handles common preprocessing logic."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from thenvoi.core.protocols import Preprocessor
from thenvoi.core.types import AgentInput, HistoryProvider, PlatformMessage
from thenvoi.platform.event import MessageEvent, PlatformEvent
from thenvoi.runtime.execution import ExecutionContext
from thenvoi.runtime.tools import AgentTools
from thenvoi.runtime.formatters import format_history_for_llm
from thenvoi.integrations.base import check_and_format_participants

logger = logging.getLogger(__name__)


class DefaultPreprocessor(Preprocessor):
    """
    Default message preprocessor.

    Handles:
    - Self-message filtering
    - Event to PlatformMessage conversion (using tagged union pattern matching)
    - Session bootstrap detection + history loading (respects enable_context_hydration)
    - Participant change detection
    - AgentTools creation
    """

    async def process(
        self,
        ctx: ExecutionContext,
        event: PlatformEvent,
        agent_id: str,
    ) -> AgentInput | None:
        """Process platform event into AgentInput."""
        # Pattern match on tagged union - only handle MessageEvent
        match event:
            case MessageEvent(room_id=room_id, payload=msg_data):
                pass  # Continue processing
            case _:
                return None  # Skip non-message events

        # msg_data is now MessageCreatedPayload (fully typed)
        if msg_data is None:
            return None

        # Validate room_id is present (narrows type from str | None to str)
        if not room_id:
            logger.error("MessageEvent has no room_id - cannot process")
            return None

        # Skip messages from self
        if msg_data.sender_type == "Agent" and msg_data.sender_id == agent_id:
            logger.debug(f"Room {room_id}: Skipping own message {msg_data.id}")
            return None

        # Convert to PlatformMessage (typed attribute access, no dict lookups)
        msg = PlatformMessage(
            id=msg_data.id,
            room_id=msg_data.chat_room_id,
            content=msg_data.content,
            sender_id=msg_data.sender_id,
            sender_type=msg_data.sender_type,
            sender_name=None,
            message_type=msg_data.message_type,
            metadata=msg_data.metadata,  # Pass through as-is (Any type)
            created_at=datetime.fromisoformat(
                msg_data.inserted_at.replace("Z", "+00:00")
            ),
        )

        is_bootstrap = not ctx.is_llm_initialized

        # Load history on session bootstrap (if hydration enabled)
        raw_history: list[dict[str, Any]] = []
        if is_bootstrap:
            if ctx.config.enable_context_hydration:
                raw_history = await self._load_history(ctx, msg)
            ctx.mark_llm_initialized()

        # Check participants
        participants_msg = check_and_format_participants(ctx)

        # Create tools
        tools = AgentTools.from_context(ctx)

        return AgentInput(
            msg=msg,
            tools=tools,
            history=HistoryProvider(raw=raw_history),
            participants_msg=participants_msg,
            is_session_bootstrap=is_bootstrap,
            room_id=room_id,
        )

    async def _load_history(
        self,
        ctx: ExecutionContext,
        msg: PlatformMessage,
    ) -> list[dict[str, Any]]:
        """Load platform history for session bootstrap."""
        try:
            logger.info(f"Room {ctx.room_id}: Loading history...")
            context = await ctx.get_context()
            history = format_history_for_llm(context.messages, exclude_id=msg.id)
            logger.info(
                f"Room {ctx.room_id}: Got {len(history) if history else 0} messages"
            )
            return history or []
        except Exception as e:
            logger.warning(f"Room {ctx.room_id}: Failed to load history: {e}")
            return []
