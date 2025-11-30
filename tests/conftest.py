"""
Pytest fixtures for thenvoi SDK tests.

Provides mock objects for fast testing without real API calls.
"""

import pytest
from unittest.mock import AsyncMock
from thenvoi.client.streaming import MessageCreatedPayload, MessageMetadata, Mention


@pytest.fixture
def mock_api_client():
    """Mock AsyncRestClient for testing without real API calls."""
    client = AsyncMock()

    # Mock agent validation
    client.agents.get_agent.return_value = AsyncMock(
        data=AsyncMock(
            id="agent-123", name="TestBot", description="Test agent for unit tests"
        )
    )

    # Mock room listing
    client.chat_rooms.list_chats.return_value = AsyncMock(
        data=[
            AsyncMock(id="room-1"),
            AsyncMock(id="room-2"),
        ]
    )

    # Mock participant listing
    client.chat_participants.list_chat_participants.return_value = AsyncMock(
        data=[AsyncMock(id="agent-123", type="Agent", agent_name="TestBot")]
    )

    return client


@pytest.fixture
def mock_websocket():
    """Mock WebSocket client for testing subscriptions."""
    ws = AsyncMock()

    # Make it work as async context manager
    ws.__aenter__ = AsyncMock(return_value=ws)
    ws.__aexit__ = AsyncMock(return_value=None)

    # Mock channel operations
    ws.join_chat_room_channel = AsyncMock()
    ws.leave_chat_room_channel = AsyncMock()
    ws.join_agent_rooms_channel = AsyncMock()

    return ws


@pytest.fixture
def sample_room_message():
    """Standard test message from a user."""
    return MessageCreatedPayload(
        id="msg-789",
        content="@TestBot hello",
        message_type="text",
        metadata=MessageMetadata(
            mentions=[Mention(id="agent-123", username="TestBot")], status="sent"
        ),
        sender_id="user-456",
        sender_type="User",
        chat_room_id="room-123",
        inserted_at="test-timestamp",
        updated_at="test-timestamp",
    )


@pytest.fixture
def sample_agent_message():
    """Message from an agent (for filtering tests)."""
    return MessageCreatedPayload(
        id="msg-999",
        content="@TestBot hi",
        message_type="text",
        metadata=MessageMetadata(
            mentions=[Mention(id="agent-123", username="TestBot")], status="sent"
        ),
        sender_id="agent-123",
        sender_type="Agent",
        chat_room_id="room-123",
        inserted_at="test-timestamp",
        updated_at="test-timestamp",
    )


@pytest.fixture
def dummy_message_handler():
    """Dummy message handler for tests that don't need handler logic."""

    async def handler(msg: MessageCreatedPayload) -> None:
        pass

    return handler
