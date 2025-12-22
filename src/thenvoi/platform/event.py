"""
Platform events using tagged union pattern.

Events are strongly typed using discriminated unions, enabling type-safe
pattern matching and automatic type narrowing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

# Import payload models from streaming client
from thenvoi.client.streaming import (
    MessageCreatedPayload,
    RoomAddedPayload,
    RoomRemovedPayload,
    ParticipantAddedPayload,
    ParticipantRemovedPayload,
)


@dataclass(kw_only=True)
class MessageEvent:
    """Message created event."""

    type: Literal["message_created"] = "message_created"
    room_id: str
    payload: MessageCreatedPayload
    raw: dict[str, Any] | None = None


@dataclass(kw_only=True)
class RoomAddedEvent:
    """Room added event."""

    type: Literal["room_added"] = "room_added"
    room_id: str
    payload: RoomAddedPayload
    raw: dict[str, Any] | None = None


@dataclass(kw_only=True)
class RoomRemovedEvent:
    """Room removed event."""

    type: Literal["room_removed"] = "room_removed"
    room_id: str
    payload: RoomRemovedPayload
    raw: dict[str, Any] | None = None


@dataclass(kw_only=True)
class ParticipantAddedEvent:
    """Participant added event."""

    type: Literal["participant_added"] = "participant_added"
    room_id: str
    payload: ParticipantAddedPayload
    raw: dict[str, Any] | None = None


@dataclass(kw_only=True)
class ParticipantRemovedEvent:
    """Participant removed event."""

    type: Literal["participant_removed"] = "participant_removed"
    room_id: str
    payload: ParticipantRemovedPayload
    raw: dict[str, Any] | None = None


# Union type for all platform events
PlatformEvent = (
    MessageEvent
    | RoomAddedEvent
    | RoomRemovedEvent
    | ParticipantAddedEvent
    | ParticipantRemovedEvent
)
