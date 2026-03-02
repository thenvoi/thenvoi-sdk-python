"""
Tests for WebSocket payload validation.

These tests ensure the SDK handles invalid payloads gracefully by logging
errors and skipping malformed events, rather than crashing the connection.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from thenvoi.client.streaming import (
    MessageCreatedPayload,
    ParticipantAddedPayload,
    ParticipantRemovedPayload,
    RoomAddedPayload,
    RoomRemovedPayload,
    WebSocketClient,
)

# Shared valid payload used by multiple tests
VALID_MESSAGE_CREATED_PAYLOAD: dict = {
    "id": "msg-123",
    "content": "@TestBot hi",
    "message_type": "text",
    "metadata": {
        "mentions": [{"id": "agent-123", "handle": "testbot", "name": "TestBot"}],
        "status": "sent",
    },
    "sender_id": "user-456",
    "sender_type": "User",
    "chat_room_id": "room-123",
    "thread_id": None,
    "inserted_at": "2025-11-17T11:20:10.284136Z",
    "updated_at": "2025-11-17T11:20:10.284136Z",
}


def _mock_message(event: str, payload: dict) -> SimpleNamespace:
    return SimpleNamespace(event=event, payload=payload)


# --- Invalid payload tests: verify graceful handling (log + skip) ---


async def test_skips_invalid_message_created_payload(caplog):
    """Should log error and skip when message_created payload is missing required fields."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")
    callback_called = False

    message = _mock_message(
        "message_created",
        {
            "id": "msg-123",
            # Missing: content, sender_id, sender_type, etc.
        },
    )

    async def dummy_callback(payload):
        nonlocal callback_called
        callback_called = True

    with caplog.at_level(logging.ERROR):
        await client._handle_events(message, {"message_created": dummy_callback})

    assert not callback_called, "Callback should not be called for invalid payload"
    assert "Invalid message_created payload" in caplog.text


async def test_skips_invalid_room_added_payload(caplog):
    """Should log error and skip when room_added payload is missing required fields."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")
    callback_called = False

    message = _mock_message(
        "room_added",
        {
            # Missing: id (the only required field)
            "title": "Test Room",
        },
    )

    async def dummy_callback(payload):
        nonlocal callback_called
        callback_called = True

    with caplog.at_level(logging.ERROR):
        await client._handle_events(message, {"room_added": dummy_callback})

    assert not callback_called, "Callback should not be called for invalid payload"
    assert "Invalid room_added payload" in caplog.text


async def test_skips_invalid_room_removed_payload(caplog):
    """Should log error and skip when room_removed payload is missing required fields."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")
    callback_called = False

    message = _mock_message(
        "room_removed",
        {
            "id": "room-123",
            # Missing: status, type, title, removed_at
        },
    )

    async def dummy_callback(payload):
        nonlocal callback_called
        callback_called = True

    with caplog.at_level(logging.ERROR):
        await client._handle_events(message, {"room_removed": dummy_callback})

    assert not callback_called, "Callback should not be called for invalid payload"
    assert "Invalid room_removed payload" in caplog.text


async def test_skips_invalid_participant_added_payload(caplog):
    """Should log error and skip when participant_added payload is missing required fields."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")
    callback_called = False

    message = _mock_message(
        "participant_added",
        {
            "id": "p-123",
            # Missing required fields: name, type (only id is provided)
        },
    )

    async def dummy_callback(payload):
        nonlocal callback_called
        callback_called = True

    with caplog.at_level(logging.ERROR):
        await client._handle_events(message, {"participant_added": dummy_callback})

    assert not callback_called, "Callback should not be called for invalid payload"
    assert "Invalid participant_added payload" in caplog.text


async def test_skips_invalid_participant_removed_payload(caplog):
    """Should log error and skip when participant_removed payload is missing required fields."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")
    callback_called = False

    message = _mock_message(
        "participant_removed",
        {
            # Missing: id
        },
    )

    async def dummy_callback(payload):
        nonlocal callback_called
        callback_called = True

    with caplog.at_level(logging.ERROR):
        await client._handle_events(message, {"participant_removed": dummy_callback})

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

    message = _mock_message("message_created", VALID_MESSAGE_CREATED_PAYLOAD)

    await client._handle_events(message, {"message_created": test_callback})
    assert isinstance(received_payload, MessageCreatedPayload)
    assert received_payload.id == "msg-123"


async def test_accepts_valid_room_added_payload():
    """Should accept valid room_added payload without raising."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")
    received_payload = None

    async def test_callback(payload):
        nonlocal received_payload
        received_payload = payload

    message = _mock_message(
        "room_added",
        {
            "id": "room-123",
            "title": "Test Room",
            "task_id": None,
            "inserted_at": "2025-11-17T09:05:35.642172Z",
            "updated_at": "2025-11-17T09:05:35.642172Z",
        },
    )

    await client._handle_events(message, {"room_added": test_callback})
    assert isinstance(received_payload, RoomAddedPayload)
    assert received_payload.id == "room-123"


async def test_accepts_valid_room_removed_payload():
    """Should accept valid room_removed payload without raising."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")
    received_payload = None

    async def test_callback(payload):
        nonlocal received_payload
        received_payload = payload

    message = _mock_message(
        "room_removed",
        {
            "id": "room-123",
            "status": "active",
            "type": "direct",
            "title": "Test Room",
            "removed_at": "2025-11-17T11:26:59.925707",
        },
    )

    await client._handle_events(message, {"room_removed": test_callback})
    assert isinstance(received_payload, RoomRemovedPayload)
    assert received_payload.id == "room-123"


async def test_accepts_valid_participant_added_payload():
    """Should accept valid participant_added payload and pass typed model to callback."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")
    received_payload = None

    async def test_callback(payload):
        nonlocal received_payload
        received_payload = payload

    message = _mock_message(
        "participant_added",
        {
            "id": "p-123",
            "name": "Test Agent",
            "type": "Agent",
        },
    )

    await client._handle_events(message, {"participant_added": test_callback})
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

    message = _mock_message(
        "participant_removed",
        {
            "id": "p-123",
        },
    )

    await client._handle_events(message, {"participant_removed": test_callback})
    assert isinstance(received_payload, ParticipantRemovedPayload)
    assert received_payload.id == "p-123"


@pytest.mark.parametrize(
    ("event_name", "base_payload", "expected_type"),
    [
        pytest.param(
            "message_created",
            {
                "id": "msg-123",
                "content": "hi",
                "message_type": "text",
                "metadata": {
                    "mentions": [{"id": "a-1", "handle": "bot", "name": "Bot"}],
                    "status": "sent",
                },
                "sender_id": "u-1",
                "sender_type": "User",
                "chat_room_id": "r-1",
                "thread_id": None,
                "inserted_at": "2025-11-17T11:20:10Z",
                "updated_at": "2025-11-17T11:20:10Z",
            },
            MessageCreatedPayload,
            id="message_created",
        ),
        pytest.param(
            "room_added",
            {
                "id": "room-123",
                "owner": {"id": "u-1", "name": "User", "type": "User"},
                "status": "active",
                "type": "direct",
                "title": "Room",
                "inserted_at": "2025-11-17T09:05:35Z",
                "updated_at": "2025-11-17T09:05:35Z",
                "participant_role": "member",
            },
            RoomAddedPayload,
            id="room_added",
        ),
        pytest.param(
            "room_removed",
            {
                "id": "room-123",
                "status": "active",
                "type": "direct",
                "title": "Room",
                "removed_at": "2025-11-17T11:26:59Z",
            },
            RoomRemovedPayload,
            id="room_removed",
        ),
        pytest.param(
            "participant_added",
            {"id": "p-123", "name": "Agent", "type": "Agent"},
            ParticipantAddedPayload,
            id="participant_added",
        ),
        pytest.param(
            "participant_removed",
            {"id": "p-123"},
            ParticipantRemovedPayload,
            id="participant_removed",
        ),
    ],
)
async def test_allows_extra_fields_in_payload(event_name, base_payload, expected_type):
    """Should accept payloads with extra fields (forward compatibility)."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")
    received_payload = None

    async def test_callback(payload):
        nonlocal received_payload
        received_payload = payload

    extra_fields = {"extra_field_1": "some value", "extra_field_2": 42}

    message = _mock_message(event_name, {**base_payload, **extra_fields})

    await client._handle_events(message, {event_name: test_callback})
    assert isinstance(received_payload, expected_type)


async def test_skips_unknown_event_without_handler(caplog):
    """Should warn when receiving an event with no registered handler."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")

    message = _mock_message("unknown_event", {"data": "test"})

    with caplog.at_level(logging.WARNING):
        await client._handle_events(message, {})

    assert "no handler registered" in caplog.text


async def test_passes_raw_dict_for_unknown_event_types():
    """Should pass raw payload dict for event types without Pydantic models."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")
    received_payload = None

    async def test_callback(payload):
        nonlocal received_payload
        received_payload = payload

    message = _mock_message("task_created", {"task_id": "t-123", "status": "pending"})

    await client._handle_events(message, {"task_created": test_callback})
    assert received_payload == {"task_id": "t-123", "status": "pending"}


# --- Validation error counter tests ---


async def test_validation_error_count_increments_on_invalid_payload(caplog):
    """Should increment validation_error_count when a payload fails validation."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")
    assert client.validation_error_count == 0

    message = _mock_message(
        "message_created",
        {"id": "msg-123"},  # Missing required fields
    )

    async def dummy_callback(payload):
        pass

    with caplog.at_level(logging.ERROR):
        await client._handle_events(message, {"message_created": dummy_callback})

    assert client.validation_error_count == 1

    # Send another invalid payload to verify it keeps incrementing
    with caplog.at_level(logging.ERROR):
        await client._handle_events(message, {"message_created": dummy_callback})

    assert client.validation_error_count == 2


async def test_validation_error_count_stays_zero_on_valid_payload():
    """Should not increment validation_error_count for valid payloads."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")

    message = _mock_message("message_created", VALID_MESSAGE_CREATED_PAYLOAD)

    async def dummy_callback(payload):
        pass

    await client._handle_events(message, {"message_created": dummy_callback})
    assert client.validation_error_count == 0


async def test_reset_validation_error_count_returns_previous_value():
    """Should reset validation_error_count back to zero and return old value."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")

    message = _mock_message(
        "message_created",
        {"id": "msg-123"},  # Missing required fields
    )

    async def dummy_callback(payload):
        pass

    # Drive the counter up
    await client._handle_events(message, {"message_created": dummy_callback})
    assert client.validation_error_count == 1

    old_count = client.reset_validation_error_count()
    assert old_count == 1
    assert client.validation_error_count == 0


async def test_callback_exception_does_not_crash_handler(caplog):
    """Should log exception and not propagate when callback raises."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")

    message = _mock_message("message_created", VALID_MESSAGE_CREATED_PAYLOAD)

    async def failing_callback(payload):
        raise RuntimeError("callback boom")

    with caplog.at_level(logging.ERROR):
        await client._handle_events(message, {"message_created": failing_callback})

    assert "Callback error for message_created event" in caplog.text
    assert client.validation_error_count == 0


async def test_cancelled_error_propagates_through_callback():
    """CancelledError raised in callback must propagate (not be swallowed)."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")

    message = _mock_message("message_created", VALID_MESSAGE_CREATED_PAYLOAD)

    async def cancelling_callback(payload):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await client._handle_events(message, {"message_created": cancelling_callback})


async def test_join_agent_rooms_channel_uses_declarative_topic_spec():
    """join_agent_rooms_channel should subscribe using channel registry topic."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")
    client.client = AsyncMock()
    client.client.subscribe_to_topic = AsyncMock(return_value={"ok": True})

    async def _on_room_added(_payload: RoomAddedPayload) -> None:
        return None

    async def _on_room_removed(_payload: RoomRemovedPayload) -> None:
        return None

    result = await client.join_agent_rooms_channel(
        "bridge-agent",
        _on_room_added,
        _on_room_removed,
    )

    assert result == {"ok": True}
    subscribe_call = client.client.subscribe_to_topic.call_args
    assert subscribe_call.args[0] == "agent_rooms:bridge-agent"
    message_handler = subscribe_call.args[1]
    assert callable(message_handler)


def test_channel_event_handlers_validate_required_events():
    """Declarative channel spec should reject missing required callbacks."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")

    with pytest.raises(ValueError, match="Missing handlers"):
        client._channel_event_handlers(
            "agent_contacts",
            {
                "contact_request_received": AsyncMock(),
                "contact_request_updated": AsyncMock(),
                "contact_added": AsyncMock(),
            },
        )
