"""Platform message normalization helpers for bridge routing."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from thenvoi.runtime.types import PlatformMessage

if TYPE_CHECKING:
    from thenvoi.client.streaming import MessageCreatedPayload

logger = logging.getLogger(__name__)


def coerce_datetime(value: Any) -> datetime:
    """Convert platform timestamp values into datetime objects."""
    if isinstance(value, datetime):
        return value

    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            logger.debug("Failed to parse timestamp: %s", value)
            return datetime.now(timezone.utc)

    return datetime.now(timezone.utc)


def metadata_to_dict(value: Any) -> dict[str, Any]:
    """Normalize payload metadata objects to plain dictionaries."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()  # type: ignore[no-any-return]
    return {}


def build_platform_message(
    payload: "MessageCreatedPayload",
    room_id: str,
    *,
    sender_name: str | None,
) -> PlatformMessage:
    """Build a normalized PlatformMessage from streaming payload data."""
    thread_id = payload.thread_id or room_id
    return PlatformMessage(
        id=payload.id,
        room_id=room_id,
        content=payload.content,
        sender_id=payload.sender_id,
        sender_type=payload.sender_type,
        sender_name=sender_name,
        message_type=payload.message_type or "text",
        metadata=metadata_to_dict(payload.metadata),
        created_at=coerce_datetime(payload.inserted_at),
        thread_id=thread_id,
    )


__all__ = ["build_platform_message", "coerce_datetime", "metadata_to_dict"]
