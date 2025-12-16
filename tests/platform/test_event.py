"""Tests for PlatformEvent."""

import pytest

from thenvoi.platform.event import PlatformEvent
from thenvoi.client.streaming import (
    MessageCreatedPayload,
    RoomAddedPayload,
    RoomRemovedPayload,
)


class TestPlatformEventConstruction:
    """Test PlatformEvent construction and basic fields."""

    def test_construct_message_event(self):
        """Construct a message_created event."""
        event = PlatformEvent(
            type="message_created",
            room_id="room-123",
            payload={
                "id": "msg-456",
                "content": "Hello",
                "message_type": "text",
                "sender_id": "user-789",
                "sender_type": "User",
                "chat_room_id": "room-123",
                "inserted_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
                "metadata": {"mentions": [], "status": "sent"},
            },
        )

        assert event.type == "message_created"
        assert event.room_id == "room-123"
        assert event.payload["content"] == "Hello"
        assert event.raw is None

    def test_construct_room_added_event(self):
        """Construct a room_added event."""
        event = PlatformEvent(
            type="room_added",
            room_id="room-123",
            payload={
                "id": "room-123",
                "title": "Test Room",
                "inserted_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            },
        )

        assert event.type == "room_added"
        assert event.room_id == "room-123"
        assert event.payload["title"] == "Test Room"

    def test_construct_with_raw_payload(self):
        """Event can store raw WebSocket payload for debugging."""
        raw = {
            "channel": "chat_room:room-123",
            "event": "message_created",
            "payload": {},
        }
        event = PlatformEvent(
            type="message_created",
            room_id="room-123",
            payload={"id": "msg-1"},
            raw=raw,
        )

        assert event.raw == raw


class TestPlatformEventTypeChecks:
    """Test is_* property helpers."""

    def test_is_message(self):
        event = PlatformEvent(type="message_created", room_id="r1", payload={})
        assert event.is_message is True
        assert event.is_room_added is False
        assert event.is_room_removed is False
        assert event.is_participant_added is False
        assert event.is_participant_removed is False

    def test_is_room_added(self):
        event = PlatformEvent(type="room_added", room_id="r1", payload={})
        assert event.is_room_added is True
        assert event.is_message is False

    def test_is_room_removed(self):
        event = PlatformEvent(type="room_removed", room_id="r1", payload={})
        assert event.is_room_removed is True
        assert event.is_message is False

    def test_is_participant_added(self):
        event = PlatformEvent(type="participant_added", room_id="r1", payload={})
        assert event.is_participant_added is True
        assert event.is_message is False

    def test_is_participant_removed(self):
        event = PlatformEvent(type="participant_removed", room_id="r1", payload={})
        assert event.is_participant_removed is True
        assert event.is_message is False


class TestPlatformEventTypedAccessors:
    """Test as_* methods that convert to typed payload models."""

    def test_as_message_success(self):
        """as_message() returns MessageCreatedPayload when type matches."""
        event = PlatformEvent(
            type="message_created",
            room_id="room-123",
            payload={
                "id": "msg-456",
                "content": "Hello world",
                "message_type": "text",
                "sender_id": "user-789",
                "sender_type": "User",
                "chat_room_id": "room-123",
                "inserted_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
                "metadata": {"mentions": [], "status": "sent"},
            },
        )

        msg = event.as_message()

        assert isinstance(msg, MessageCreatedPayload)
        assert msg.id == "msg-456"
        assert msg.content == "Hello world"
        assert msg.sender_id == "user-789"
        assert msg.sender_type == "User"

    def test_as_message_wrong_type_raises(self):
        """as_message() raises TypeError for non-message events."""
        event = PlatformEvent(type="room_added", room_id="r1", payload={})

        with pytest.raises(TypeError) as exc_info:
            event.as_message()

        assert "Expected message_created, got room_added" in str(exc_info.value)

    def test_as_room_added_success(self):
        """as_room_added() returns RoomAddedPayload when type matches."""
        event = PlatformEvent(
            type="room_added",
            room_id="room-123",
            payload={
                "id": "room-123",
                "title": "Test Room",
                "owner": {"id": "user-1", "name": "Test User", "type": "User"},
                "status": "active",
                "type": "direct",
                "created_at": "2024-01-01T00:00:00Z",
                "participant_role": "member",
            },
        )

        room = event.as_room_added()

        assert isinstance(room, RoomAddedPayload)
        assert room.id == "room-123"
        assert room.title == "Test Room"
        assert room.status == "active"

    def test_as_room_added_wrong_type_raises(self):
        """as_room_added() raises TypeError for non-room_added events."""
        event = PlatformEvent(type="message_created", room_id="r1", payload={})

        with pytest.raises(TypeError) as exc_info:
            event.as_room_added()

        assert "Expected room_added, got message_created" in str(exc_info.value)

    def test_as_room_removed_success(self):
        """as_room_removed() returns RoomRemovedPayload when type matches."""
        event = PlatformEvent(
            type="room_removed",
            room_id="room-123",
            payload={
                "id": "room-123",
                "status": "removed",
                "type": "direct",
                "title": "Test Room",
                "removed_at": "2024-01-01T00:00:00Z",
            },
        )

        room = event.as_room_removed()

        assert isinstance(room, RoomRemovedPayload)
        assert room.id == "room-123"
        assert room.status == "removed"

    def test_as_room_removed_wrong_type_raises(self):
        """as_room_removed() raises TypeError for non-room_removed events."""
        event = PlatformEvent(type="message_created", room_id="r1", payload={})

        with pytest.raises(TypeError) as exc_info:
            event.as_room_removed()

        assert "Expected room_removed, got message_created" in str(exc_info.value)


class TestPlatformEventEdgeCases:
    """Edge cases and special scenarios."""

    def test_event_with_none_room_id(self):
        """Some events may have no room_id."""
        event = PlatformEvent(
            type="system_event", room_id=None, payload={"status": "ok"}
        )

        assert event.room_id is None
        assert event.payload["status"] == "ok"

    def test_event_payload_is_dict_copy_safe(self):
        """Payload should be safely accessible as dict."""
        payload = {"id": "123", "nested": {"key": "value"}}
        event = PlatformEvent(type="test", room_id="r1", payload=payload)

        # Mutating original doesn't affect event (if caller passes copy)
        # Note: PlatformEvent stores reference, caller should pass copy if needed
        assert event.payload["nested"]["key"] == "value"
