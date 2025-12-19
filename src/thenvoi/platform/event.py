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


@dataclass
class MessageEvent:
    """Message created event."""

    type: Literal["message_created"] = "message_created"
    room_id: str | None = None
    payload: MessageCreatedPayload | None = None
    raw: dict[str, Any] | None = None


@dataclass
class RoomAddedEvent:
    """Room added event."""

    type: Literal["room_added"] = "room_added"
    room_id: str | None = None
    payload: RoomAddedPayload | None = None
    raw: dict[str, Any] | None = None


@dataclass
class RoomRemovedEvent:
    """Room removed event."""

    type: Literal["room_removed"] = "room_removed"
    room_id: str | None = None
    payload: RoomRemovedPayload | None = None
    raw: dict[str, Any] | None = None


@dataclass
class ParticipantAddedEvent:
    """Participant added event."""

    type: Literal["participant_added"] = "participant_added"
    room_id: str | None = None
    payload: ParticipantAddedPayload | None = None
    raw: dict[str, Any] | None = None


@dataclass
class ParticipantRemovedEvent:
    """Participant removed event."""

    type: Literal["participant_removed"] = "participant_removed"
    room_id: str | None = None
    payload: ParticipantRemovedPayload | None = None
    raw: dict[str, Any] | None = None


# Union type for all platform events
PlatformEvent = (
    MessageEvent
    | RoomAddedEvent
    | RoomRemovedEvent
    | ParticipantAddedEvent
    | ParticipantRemovedEvent
)
