"""
PlatformEvent - Single event type for all platform events.

Reuses existing payload models from streaming/client.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# Reuse existing payload models - don't recreate
from thenvoi.client.streaming import (
    MessageCreatedPayload,
    RoomAddedPayload,
    RoomRemovedPayload,
)


@dataclass
class PlatformEvent:
    """
    Single event type for all platform events.

    Wraps WebSocket events with consistent interface.
    Typed accessors use existing payload models from streaming/client.py.
    """

    type: str
    room_id: Optional[str]
    payload: dict[str, Any]
    raw: Optional[dict[str, Any]] = field(default=None, repr=False)

    # --- Typed accessors (reuse existing models) ---

    def as_message(self) -> MessageCreatedPayload:
        """Get payload as MessageCreatedPayload."""
        if self.type != "message_created":
            raise TypeError(f"Expected message_created, got {self.type}")
        return MessageCreatedPayload(**self.payload)

    def as_room_added(self) -> RoomAddedPayload:
        """Get payload as RoomAddedPayload."""
        if self.type != "room_added":
            raise TypeError(f"Expected room_added, got {self.type}")
        return RoomAddedPayload(**self.payload)

    def as_room_removed(self) -> RoomRemovedPayload:
        """Get payload as RoomRemovedPayload."""
        if self.type != "room_removed":
            raise TypeError(f"Expected room_removed, got {self.type}")
        return RoomRemovedPayload(**self.payload)

    # --- Convenience checks ---

    @property
    def is_message(self) -> bool:
        return self.type == "message_created"

    @property
    def is_room_added(self) -> bool:
        return self.type == "room_added"

    @property
    def is_room_removed(self) -> bool:
        return self.type == "room_removed"

    @property
    def is_participant_added(self) -> bool:
        return self.type == "participant_added"

    @property
    def is_participant_removed(self) -> bool:
        return self.type == "participant_removed"
