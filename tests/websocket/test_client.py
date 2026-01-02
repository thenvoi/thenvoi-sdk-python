"""
Tests for WebSocket payload validation.

These tests ensure the SDK fails fast when backend sends invalid payloads,
similar to how the REST API SDK raises errors for spec violations.
"""

import pytest
from pydantic import ValidationError
from thenvoi.client.streaming import WebSocketClient


async def test_validates_message_created_payload():
    """Should raise ValidationError when message_created payload is missing required fields."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")

    # Create a mock PHXMessage with invalid payload
    class MockMessage:
        event = "message_created"
        payload = {
            "id": "msg-123",
            # Missing: content, sender_id, sender_type, etc.
        }

    # Must provide callback for validation to occur (code returns early without callback)
    async def dummy_callback(payload):
        pass

    with pytest.raises(ValidationError) as exc_info:
        await client._handle_events(MockMessage(), {"message_created": dummy_callback})

    # Verify ValidationError mentions missing fields
    error_str = str(exc_info.value)
    assert "content" in error_str or "required" in error_str.lower()


async def test_validates_room_added_payload():
    """Should raise ValidationError when room_added payload is missing required fields."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")

    class MockMessage:
        event = "room_added"
        payload = {
            "id": "room-123",
            # Missing: owner, status, type, title, etc.
        }

    # Must provide callback for validation to occur (code returns early without callback)
    async def dummy_callback(payload):
        pass

    with pytest.raises(ValidationError) as exc_info:
        await client._handle_events(MockMessage(), {"room_added": dummy_callback})

    error_str = str(exc_info.value)
    assert "owner" in error_str or "required" in error_str.lower()


async def test_validates_room_removed_payload():
    """Should raise ValidationError when room_removed payload is missing required fields."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")

    class MockMessage:
        event = "room_removed"
        payload = {
            "id": "room-123",
            # Missing: status, type, title, removed_at
        }

    # Must provide callback for validation to occur (code returns early without callback)
    async def dummy_callback(payload):
        pass

    with pytest.raises(ValidationError) as exc_info:
        await client._handle_events(MockMessage(), {"room_removed": dummy_callback})

    error_str = str(exc_info.value)
    assert "required" in error_str.lower()


async def test_accepts_valid_message_created_payload():
    """Should accept valid message_created payload without raising."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")
    callback_called = False

    async def test_callback(payload):
        nonlocal callback_called
        callback_called = True

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

    # Should not raise
    await client._handle_events(MockMessage(), {"message_created": test_callback})
    assert callback_called


async def test_accepts_valid_room_added_payload():
    """Should accept valid room_added payload without raising."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")
    callback_called = False

    async def test_callback(payload):
        nonlocal callback_called
        callback_called = True

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

    # Should not raise
    await client._handle_events(MockMessage(), {"room_added": test_callback})
    assert callback_called


async def test_accepts_valid_room_removed_payload():
    """Should accept valid room_removed payload without raising."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")
    callback_called = False

    async def test_callback(payload):
        nonlocal callback_called
        callback_called = True

    class MockMessage:
        event = "room_removed"
        payload = {
            "id": "room-123",
            "status": "active",
            "type": "direct",
            "title": "Test Room",
            "removed_at": "2025-11-17T11:26:59.925707",
        }

    # Should not raise
    await client._handle_events(MockMessage(), {"room_removed": test_callback})
    assert callback_called


async def test_allows_extra_fields_in_payload():
    """Should accept payloads with extra fields (forward compatibility)."""
    client = WebSocketClient("ws://localhost", "test-key", "agent-123")

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

    # Should not raise - extra fields are allowed
    await client._handle_events(MockMessage(), {})
