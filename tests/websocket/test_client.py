"""
Tests for WebSocket payload validation.

These tests ensure the SDK handles invalid payloads gracefully by logging
errors and skipping malformed events, rather than crashing the connection.
"""

from __future__ import annotations

import logging

from thenvoi.client.streaming import (
    MessageCreatedPayload,
    ParticipantAddedPayload,
    ParticipantRemovedPayload,
    RoomAddedPayload,
    RoomRemovedPayload,
    WebSocketClient,
)


# --- Invalid payload tests: verify graceful handling (log + skip) ---


async def test_skips_invalid_message_created_payload(caplog):
    """Should log error and skip when message_created payload is missing required fields."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")
    callback_called = False

    class MockMessage:
        event = "message_created"
        payload = {
            "id": "msg-123",
            # Missing: content, sender_id, sender_type, etc.
        }

    async def dummy_callback(payload):
        nonlocal callback_called
        callback_called = True

    with caplog.at_level(logging.ERROR):
        await client._handle_events(MockMessage(), {"message_created": dummy_callback})

    assert not callback_called, "Callback should not be called for invalid payload"
    assert "Invalid message_created payload" in caplog.text


async def test_skips_invalid_room_added_payload(caplog):
    """Should log error and skip when room_added payload is missing required fields."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")
    callback_called = False

    class MockMessage:
        event = "room_added"
        payload = {
            "id": "room-123",
            # Missing: owner, status, type, title, etc.
        }

    async def dummy_callback(payload):
        nonlocal callback_called
        callback_called = True

    with caplog.at_level(logging.ERROR):
        await client._handle_events(MockMessage(), {"room_added": dummy_callback})

    assert not callback_called, "Callback should not be called for invalid payload"
    assert "Invalid room_added payload" in caplog.text


async def test_skips_invalid_room_removed_payload(caplog):
    """Should log error and skip when room_removed payload is missing required fields."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")
    callback_called = False

    class MockMessage:
        event = "room_removed"
        payload = {
            "id": "room-123",
            # Missing: status, type, title, removed_at
        }

    async def dummy_callback(payload):
        nonlocal callback_called
        callback_called = True

    with caplog.at_level(logging.ERROR):
        await client._handle_events(MockMessage(), {"room_removed": dummy_callback})

    assert not callback_called, "Callback should not be called for invalid payload"
    assert "Invalid room_removed payload" in caplog.text


async def test_skips_invalid_participant_added_payload(caplog):
    """Should log error and skip when participant_added payload is missing required fields."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")
    callback_called = False

    class MockMessage:
        event = "participant_added"
        payload = {
            "id": "p-123",
            # Missing: name, type
        }

    async def dummy_callback(payload):
        nonlocal callback_called
        callback_called = True

    with caplog.at_level(logging.ERROR):
        await client._handle_events(
            MockMessage(), {"participant_added": dummy_callback}
        )

    assert not callback_called, "Callback should not be called for invalid payload"
    assert "Invalid participant_added payload" in caplog.text


async def test_skips_invalid_participant_removed_payload(caplog):
    """Should log error and skip when participant_removed payload is missing required fields."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")
    callback_called = False

    class MockMessage:
        event = "participant_removed"
        payload = {
            # Missing: id
        }

    async def dummy_callback(payload):
        nonlocal callback_called
        callback_called = True

    with caplog.at_level(logging.ERROR):
        await client._handle_events(
            MockMessage(), {"participant_removed": dummy_callback}
        )

    assert not callback_called, "Callback should not be called for invalid payload"
    assert "Invalid participant_removed payload" in caplog.text


# --- Valid payload tests ---


async def test_accepts_valid_message_created_payload():
    """Should accept valid message_created payload without raising."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")
    received_payload = None

    async def test_callback(payload):
        nonlocal received_payload
        received_payload = payload

    class MockMessage:
        event = "message_created"
        payload = {
            "id": "msg-123",
            "content": "@TestBot hi",
            "message_type": "text",
            "metadata": {
                "mentions": [{"id": "agent-123", "username": "TestBot"}],
                "status": "sent",
            },
            "sender_id": "user-456",
            "sender_type": "User",
            "chat_room_id": "room-123",
            "thread_id": None,
            "inserted_at": "2025-11-17T11:20:10.284136Z",
            "updated_at": "2025-11-17T11:20:10.284136Z",
        }

    await client._handle_events(MockMessage(), {"message_created": test_callback})
    assert isinstance(received_payload, MessageCreatedPayload)
    assert received_payload.id == "msg-123"


async def test_accepts_valid_room_added_payload():
    """Should accept valid room_added payload without raising."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")
    received_payload = None

    async def test_callback(payload):
        nonlocal received_payload
        received_payload = payload

    class MockMessage:
        event = "room_added"
        payload = {
            "id": "room-123",
            "owner": {"id": "user-456", "name": "Test User", "type": "User"},
            "status": "active",
            "type": "direct",
            "title": "Test Room",
            "created_at": "2025-11-17T09:05:35.642172Z",
            "participant_role": "member",
        }

    await client._handle_events(MockMessage(), {"room_added": test_callback})
    assert isinstance(received_payload, RoomAddedPayload)
    assert received_payload.id == "room-123"


async def test_accepts_valid_room_removed_payload():
    """Should accept valid room_removed payload without raising."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")
    received_payload = None

    async def test_callback(payload):
        nonlocal received_payload
        received_payload = payload

    class MockMessage:
        event = "room_removed"
        payload = {
            "id": "room-123",
            "status": "active",
            "type": "direct",
            "title": "Test Room",
            "removed_at": "2025-11-17T11:26:59.925707",
        }

    await client._handle_events(MockMessage(), {"room_removed": test_callback})
    assert isinstance(received_payload, RoomRemovedPayload)
    assert received_payload.id == "room-123"


async def test_accepts_valid_participant_added_payload():
    """Should accept valid participant_added payload and pass typed model to callback."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")
    received_payload = None

    async def test_callback(payload):
        nonlocal received_payload
        received_payload = payload

    class MockMessage:
        event = "participant_added"
        payload = {
            "id": "p-123",
            "name": "Test Agent",
            "type": "Agent",
        }

    await client._handle_events(MockMessage(), {"participant_added": test_callback})
    assert isinstance(received_payload, ParticipantAddedPayload)
    assert received_payload.id == "p-123"
    assert received_payload.name == "Test Agent"


async def test_accepts_valid_participant_removed_payload():
    """Should accept valid participant_removed payload and pass typed model to callback."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")
    received_payload = None

    async def test_callback(payload):
        nonlocal received_payload
        received_payload = payload

    class MockMessage:
        event = "participant_removed"
        payload = {
            "id": "p-123",
        }

    await client._handle_events(MockMessage(), {"participant_removed": test_callback})
    assert isinstance(received_payload, ParticipantRemovedPayload)
    assert received_payload.id == "p-123"


async def test_allows_extra_fields_in_payload():
    """Should accept payloads with extra fields (forward compatibility)."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")
    received_payload = None

    async def test_callback(payload):
        nonlocal received_payload
        received_payload = payload

    class MockMessage:
        event = "room_removed"
        payload = {
            "id": "room-123",
            "status": "active",
            "type": "direct",
            "title": "Test Room",
            "removed_at": "2025-11-17T11:26:59.925707",
            # Extra fields backend might add in the future
            "extra_field_1": "some value",
            "extra_field_2": 42,
        }

    await client._handle_events(MockMessage(), {"room_removed": test_callback})
    assert isinstance(received_payload, RoomRemovedPayload)


async def test_skips_unknown_event_without_handler(caplog):
    """Should warn when receiving an event with no registered handler."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")

    class MockMessage:
        event = "unknown_event"
        payload = {"data": "test"}

    with caplog.at_level(logging.WARNING):
        await client._handle_events(MockMessage(), {})

    assert "no handler registered" in caplog.text


async def test_passes_raw_dict_for_unknown_event_types():
    """Should pass raw payload dict for event types without Pydantic models."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")
    received_payload = None

    async def test_callback(payload):
        nonlocal received_payload
        received_payload = payload

    class MockMessage:
        event = "task_created"
        payload = {"task_id": "t-123", "status": "pending"}

    await client._handle_events(MockMessage(), {"task_created": test_callback})
    assert received_payload == {"task_id": "t-123", "status": "pending"}
