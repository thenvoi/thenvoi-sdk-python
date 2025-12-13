"""
Pytest fixtures for thenvoi SDK tests.

Provides mock objects for fast testing without real API calls.

Key fixture pattern:
- mock_agent_api: Explicit MagicMock of the agent_api namespace
- mock_api_client: AsyncMock with agent_api attached
- Tests should verify API calls using assert_called_once() and call_args
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from thenvoi.client.streaming import MessageCreatedPayload, MessageMetadata, Mention
from tests.fixtures import factory


@pytest.fixture
def mock_agent_api() -> MagicMock:
    """Create a mocked agent_api with all methods stubbed.

    This is an explicit MagicMock - it will NOT auto-create any attributes.
    Tests must set up return values for methods they want to call:

        mock_agent_api.get_agent_me.return_value = factory.response(factory.agent_me())
        mock_agent_api.list_agent_chats.return_value = factory.list_response([...])

    This ensures tests verify the correct API methods are called.
    """
    agent_api = MagicMock()

    # Pre-configure common methods with default responses
    # Tests can override these as needed
    agent_api.get_agent_me.return_value = factory.response(
        factory.agent_me(id="agent-123", name="TestBot", description="Test agent")
    )

    agent_api.list_agent_chats.return_value = factory.list_response([
        factory.chat_room(id="room-1"),
        factory.chat_room(id="room-2"),
    ])

    agent_api.list_agent_chat_participants.return_value = factory.list_response([
        factory.chat_participant(id="agent-123", name="TestBot", type="Agent"),
    ])

    agent_api.create_agent_chat_event.return_value = factory.response(
        factory.chat_event()
    )

    agent_api.create_agent_chat_message.return_value = factory.response(
        factory.chat_message()
    )

    return agent_api


@pytest.fixture
def mock_api_client(mock_agent_api: MagicMock) -> AsyncMock:
    """Create a mocked AsyncRestClient with agent_api attached.

    Uses the explicit mock_agent_api fixture to ensure API calls are verified.

    Usage:
        async def test_something(mock_api_client, mock_agent_api):
            # Set up specific return value
            mock_agent_api.get_agent_me.return_value = factory.response(...)

            # Call your code
            await some_function(mock_api_client)

            # Verify the correct API was called
            mock_agent_api.get_agent_me.assert_called_once()
    """
    client = AsyncMock()
    client.agent_api = mock_agent_api
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
