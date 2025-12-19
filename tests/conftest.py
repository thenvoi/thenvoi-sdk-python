"""
Pytest fixtures for thenvoi SDK tests.

Provides mock objects for fast testing without real API calls.

Key fixture pattern:
- mock_agent_api: Explicit MagicMock of the agent_api namespace
- mock_api_client: AsyncMock with agent_api attached
- Tests should verify API calls using assert_called_once() and call_args
"""

import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.fixtures import factory
from thenvoi.client.streaming import (
    MessageCreatedPayload,
    MessageMetadata,
    Mention,
    RoomAddedPayload,
    RoomRemovedPayload,
    RoomOwner,
    ParticipantAddedPayload,
    ParticipantRemovedPayload,
)
from thenvoi.platform.event import (
    MessageEvent,
    RoomAddedEvent,
    RoomRemovedEvent,
    ParticipantAddedEvent,
    ParticipantRemovedEvent,
)
from thenvoi.runtime.types import PlatformMessage


def pytest_ignore_collect(collection_path):
    """Skip integration tests in CI environment.

    GitHub Actions sets CI=true automatically.
    This allows CI to run unit tests while skipping integration tests
    that require API credentials.
    """
    if os.environ.get("CI") == "true":
        if "integration" in str(collection_path):
            return True
    return False


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

    agent_api.list_agent_chats.return_value = factory.list_response(
        [
            factory.chat_room(id="room-1"),
            factory.chat_room(id="room-2"),
        ]
    )

    agent_api.list_agent_chat_participants.return_value = factory.list_response(
        [
            factory.chat_participant(id="agent-123", name="TestBot", type="Agent"),
        ]
    )

    agent_api.create_agent_chat_event.return_value = factory.response(
        factory.chat_event()
    )

    agent_api.create_agent_chat_message.return_value = factory.response(
        factory.chat_message()
    )

    return agent_api


@pytest.fixture
def mock_human_api() -> MagicMock:
    """Create a mocked human_api with all methods stubbed.

    This is an explicit MagicMock for the User API (human_api namespace).
    Tests must set up return values for methods they want to call.

    Available methods:
    - get_my_profile() - User's profile
    - list_my_agents() - List owned agents
    - register_my_agent() - Register new agent (returns API key)
    - delete_my_agent() - Delete an agent
    - list_my_chats() - List user's chat rooms
    - create_my_chat_room() - Create new chat room
    - etc.
    """
    human_api = MagicMock()

    # Profile
    human_api.get_my_profile.return_value = factory.response(factory.user_profile())

    # Agents
    human_api.list_my_agents.return_value = factory.list_response(
        [factory.owned_agent()]
    )
    human_api.register_my_agent.return_value = factory.response(
        factory.registered_agent()
    )
    human_api.delete_my_agent.return_value = factory.response(factory.deleted_agent())

    # Chats (similar to agent_api)
    human_api.list_my_chats.return_value = factory.list_response(
        [factory.chat_room(id="room-1")]
    )
    human_api.create_my_chat_room.return_value = factory.response(factory.chat_room())
    human_api.get_my_chat_room.return_value = factory.response(factory.chat_room())
    human_api.list_my_chat_participants.return_value = factory.list_response(
        [factory.chat_participant()]
    )
    human_api.list_my_peers.return_value = factory.list_response([factory.peer()])

    return human_api


@pytest.fixture
def mock_api_client(mock_agent_api: MagicMock, mock_human_api: MagicMock) -> AsyncMock:
    """Create a mocked AsyncRestClient with both agent_api and human_api attached.

    Uses explicit mock fixtures to ensure API calls are verified.

    Usage:
        async def test_something(mock_api_client, mock_agent_api):
            # Set up specific return value
            mock_agent_api.get_agent_me.return_value = factory.response(...)

            # Call your code
            await some_function(mock_api_client)

            # Verify the correct API was called
            mock_agent_api.get_agent_me.assert_called_once()

        async def test_user_api(mock_api_client, mock_human_api):
            # Set up user API return value
            mock_human_api.list_my_agents.return_value = factory.list_response([...])

            # Verify user API calls
            mock_human_api.register_my_agent.assert_called_once()
    """
    client = AsyncMock()
    client.agent_api = mock_agent_api
    client.human_api = mock_human_api
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


# --- New Architecture Fixtures ---


@pytest.fixture
def mock_thenvoi_agent(mock_api_client, mock_websocket):
    """Mock ThenvoiAgent for session/adapter tests.

    Provides a mock coordinator with:
    - agent_id and agent_name set
    - Mock API client and WebSocket attached
    - Empty active_sessions dict
    - Mock internal methods for tools
    """
    agent = AsyncMock()
    agent.agent_id = "agent-123"
    agent.agent_name = "TestBot"
    agent._api_client = mock_api_client
    agent._ws_client = mock_websocket
    agent.active_sessions = {}

    # Mock internal methods used by AgentTools
    agent._send_message_internal = AsyncMock(
        return_value={"id": "msg-123", "status": "sent"}
    )
    agent._send_event_internal = AsyncMock(
        return_value={"id": "evt-123", "status": "sent"}
    )
    agent._add_participant_internal = AsyncMock(
        return_value={"id": "user-456", "name": "Test User", "role": "member"}
    )
    agent._remove_participant_internal = AsyncMock(
        return_value={"id": "user-456", "name": "Test User", "status": "removed"}
    )
    agent._lookup_peers_internal = AsyncMock(
        return_value={
            "peers": [{"id": "peer-1", "name": "Peer One", "type": "Agent"}],
            "metadata": {
                "page": 1,
                "page_size": 50,
                "total_count": 1,
                "total_pages": 1,
            },
        }
    )
    agent._get_participants_internal = AsyncMock(
        return_value=[{"id": "agent-123", "name": "TestBot", "type": "Agent"}]
    )
    agent._create_chatroom_internal = AsyncMock(return_value="new-room-123")
    agent.get_context = AsyncMock()

    return agent


@pytest.fixture
def mock_agent_session():
    """Mock AgentSession for isolated tests.

    Provides a mock session with:
    - room_id set
    - is_llm_initialized tracking
    - Empty participants list
    """
    session = AsyncMock()
    session.room_id = "room-123"
    session.is_llm_initialized = False
    session.participants = []
    session._last_participants_hash = None
    return session


@pytest.fixture
def sample_platform_message():
    """PlatformMessage fixture for new architecture.

    A standard user message mentioning the TestBot agent.
    """
    return PlatformMessage(
        id="msg-123",
        room_id="room-123",
        content="@TestBot hello",
        sender_id="user-456",
        sender_type="User",
        sender_name="Test User",
        message_type="text",
        metadata={"mentions": [{"id": "agent-123", "name": "TestBot"}]},
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_agent_platform_message():
    """PlatformMessage from the agent itself (for filtering tests)."""
    return PlatformMessage(
        id="msg-456",
        room_id="room-123",
        content="Hello there!",
        sender_id="agent-123",
        sender_type="Agent",
        sender_name="TestBot",
        message_type="text",
        metadata={},
        created_at=datetime.now(timezone.utc),
    )


# --- Helper Functions for Creating Test Events ---


def make_message_event(
    room_id: str = "room-123",
    msg_id: str = "msg-123",
    content: str = "Test message",
    sender_id: str = "user-456",
    sender_type: str = "User",
    **kwargs
) -> MessageEvent:
    """Helper to create MessageEvent for tests."""
    payload = MessageCreatedPayload(
        id=msg_id,
        content=content,
        message_type=kwargs.get("message_type", "text"),
        sender_id=sender_id,
        sender_type=sender_type,
        chat_room_id=room_id,
        inserted_at=kwargs.get("inserted_at", "2024-01-01T00:00:00Z"),
        updated_at=kwargs.get("updated_at", "2024-01-01T00:00:00Z"),
        metadata=kwargs.get("metadata", MessageMetadata(mentions=[], status="sent")),
    )
    return MessageEvent(room_id=room_id, payload=payload)


def make_room_added_event(
    room_id: str = "room-123",
    title: str = "Test Room",
    **kwargs
) -> RoomAddedEvent:
    """Helper to create RoomAddedEvent for tests."""
    payload = RoomAddedPayload(
        id=room_id,
        title=title,
        owner=kwargs.get(
            "owner", RoomOwner(id="user-1", name="Test User", type="User")
        ),
        status=kwargs.get("status", "active"),
        type=kwargs.get("type", "direct"),
        created_at=kwargs.get("created_at", "2024-01-01T00:00:00Z"),
        participant_role=kwargs.get("participant_role", "member"),
    )
    return RoomAddedEvent(room_id=room_id, payload=payload)


def make_room_removed_event(
    room_id: str = "room-123",
    title: str = "Test Room",
    **kwargs
) -> RoomRemovedEvent:
    """Helper to create RoomRemovedEvent for tests."""
    payload = RoomRemovedPayload(
        id=room_id,
        status=kwargs.get("status", "removed"),
        type=kwargs.get("type", "direct"),
        title=title,
        removed_at=kwargs.get("removed_at", "2024-01-01T00:00:00Z"),
    )
    return RoomRemovedEvent(room_id=room_id, payload=payload)


def make_participant_added_event(
    room_id: str = "room-123",
    participant_id: str = "user-456",
    name: str = "Test User",
    type: str = "User",
) -> ParticipantAddedEvent:
    """Helper to create ParticipantAddedEvent for tests."""
    payload = ParticipantAddedPayload(id=participant_id, name=name, type=type)
    return ParticipantAddedEvent(room_id=room_id, payload=payload)


def make_participant_removed_event(
    room_id: str = "room-123",
    participant_id: str = "user-456",
) -> ParticipantRemovedEvent:
    """Helper to create ParticipantRemovedEvent for tests."""
    payload = ParticipantRemovedPayload(id=participant_id)
    return ParticipantRemovedEvent(room_id=room_id, payload=payload)
