"""
Unit tests for ThenvoiAgent - coordinator and callback behavior.

Tests critical flows by mocking WebSocket and capturing registered callbacks:
1. Message filtering (SAFETY: don't respond to own messages)
2. Session lifecycle (room_added → create, room_removed → destroy + cleanup)
3. Participant updates (only for OTHER participants, not agent itself)
4. Cleanup callback invocation (for checkpointer clearing)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from thenvoi.core.agent import ThenvoiAgent
from thenvoi.client.streaming import (
    MessageCreatedPayload,
    MessageMetadata,
    RoomAddedPayload,
    RoomRemovedPayload,
    RoomOwner,
)


@pytest.fixture
def mock_rest_client():
    """Mock REST client with agent_api."""
    client = MagicMock()
    client.agent_api = MagicMock()

    # Mock get_agent_me
    agent_me = MagicMock()
    agent_me.name = "TestBot"
    agent_me.description = "A test agent"
    client.agent_api.get_agent_me = AsyncMock(return_value=MagicMock(data=agent_me))

    # Mock list_agent_chats (no existing rooms)
    client.agent_api.list_agent_chats = AsyncMock(return_value=MagicMock(data=[]))

    # Mock list_agent_chat_participants
    client.agent_api.list_agent_chat_participants = AsyncMock(
        return_value=MagicMock(data=[])
    )

    return client


@pytest.fixture
def mock_ws_client():
    """Mock WebSocket client that captures registered callbacks."""
    ws = AsyncMock()
    ws.__aenter__ = AsyncMock(return_value=ws)
    ws.__aexit__ = AsyncMock(return_value=None)

    # These will capture the callbacks
    ws.join_agent_rooms_channel = AsyncMock()
    ws.join_chat_room_channel = AsyncMock()
    ws.join_room_participants_channel = AsyncMock()
    ws.leave_chat_room_channel = AsyncMock()
    ws.leave_room_participants_channel = AsyncMock()

    return ws


@pytest.fixture
def agent(mock_rest_client, mock_ws_client):
    """Create ThenvoiAgent with mocked dependencies."""
    # Patch at the module level where it's imported
    with patch("thenvoi.core.agent.AsyncRestClient") as mock_rest_cls:
        mock_rest_cls.return_value = mock_rest_client
        agent = ThenvoiAgent(
            agent_id="agent-123",
            api_key="test-key",
            ws_url="ws://localhost/ws",
            rest_url="http://localhost",
        )
        # Store mocks for later injection in start()
        agent._mock_ws_client = mock_ws_client
        agent._mock_rest_client = mock_rest_client
        return agent


async def start_agent_with_mock_ws(agent, mock_ws_client, on_message):
    """Helper to start agent with mocked WebSocket."""
    with patch("thenvoi.core.agent.WebSocketClient") as mock_ws_cls:
        mock_ws_cls.return_value = mock_ws_client
        await agent.start(on_message=on_message)


class TestMessageFiltering:
    """CRITICAL SAFETY: Agent must not respond to its own messages."""

    async def test_filters_own_messages(self, agent, mock_ws_client):
        """Agent should ignore messages from itself to prevent infinite loops."""
        messages_received = []

        async def handler(msg, tools):
            messages_received.append(msg)

        await start_agent_with_mock_ws(agent, mock_ws_client, handler)

        # Get the room_added callback and simulate adding agent to a room
        room_added_call = mock_ws_client.join_agent_rooms_channel.call_args
        on_room_added = room_added_call.kwargs["on_room_added"]

        await on_room_added(
            RoomAddedPayload(
                id="room-123",
                owner=RoomOwner(id="owner-1", name="Owner", type="User"),
                status="active",
                type="direct",
                title="Test Room",
                created_at="2025-01-01T00:00:00Z",
                participant_role="member",
            )
        )

        # Now get the message callback for that room
        message_call = mock_ws_client.join_chat_room_channel.call_args
        on_message_created = message_call.kwargs["on_message_created"]

        # Simulate receiving OUR OWN message
        own_message = MessageCreatedPayload(
            id="msg-123",
            content="Hello from myself",
            message_type="text",
            metadata=MessageMetadata(mentions=[], status="sent"),
            sender_id="agent-123",  # Same as agent_id!
            sender_type="Agent",
            chat_room_id="room-123",
            inserted_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
        )
        await on_message_created(own_message)

        # Message should be filtered - queue should be empty
        session = agent._sessions.get("room-123")
        assert session is not None
        assert session.queue.qsize() == 0

    async def test_processes_other_messages(self, agent, mock_ws_client):
        """Agent should process messages from other users."""
        await start_agent_with_mock_ws(agent, mock_ws_client, AsyncMock())

        # Setup room
        on_room_added = mock_ws_client.join_agent_rooms_channel.call_args.kwargs[
            "on_room_added"
        ]
        await on_room_added(
            RoomAddedPayload(
                id="room-123",
                owner=RoomOwner(id="owner-1", name="Owner", type="User"),
                status="active",
                type="direct",
                title="Test Room",
                created_at="2025-01-01T00:00:00Z",
                participant_role="member",
            )
        )

        on_message_created = mock_ws_client.join_chat_room_channel.call_args.kwargs[
            "on_message_created"
        ]

        # Simulate receiving message from ANOTHER user
        user_message = MessageCreatedPayload(
            id="msg-456",
            content="Hello from user",
            message_type="text",
            metadata=MessageMetadata(mentions=[], status="sent"),
            sender_id="user-456",  # Different from agent_id
            sender_type="User",
            chat_room_id="room-123",
            inserted_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
        )
        await on_message_created(user_message)

        # Message should be enqueued
        session = agent._sessions.get("room-123")
        assert session.queue.qsize() == 1


class TestSessionLifecycle:
    """Tests for session creation and destruction."""

    async def test_room_added_creates_session(self, agent, mock_ws_client):
        """room_added event should create a new session."""
        await start_agent_with_mock_ws(agent, mock_ws_client, AsyncMock())

        on_room_added = mock_ws_client.join_agent_rooms_channel.call_args.kwargs[
            "on_room_added"
        ]

        await on_room_added(
            RoomAddedPayload(
                id="room-123",
                owner=RoomOwner(id="owner-1", name="Owner", type="User"),
                status="active",
                type="direct",
                title="Test Room",
                created_at="2025-01-01T00:00:00Z",
                participant_role="member",
            )
        )

        assert "room-123" in agent._sessions
        assert agent._sessions["room-123"].room_id == "room-123"

    async def test_room_added_subscribes_to_channels(self, agent, mock_ws_client):
        """room_added should subscribe to room messages and participants."""
        await start_agent_with_mock_ws(agent, mock_ws_client, AsyncMock())

        on_room_added = mock_ws_client.join_agent_rooms_channel.call_args.kwargs[
            "on_room_added"
        ]

        await on_room_added(
            RoomAddedPayload(
                id="room-123",
                owner=RoomOwner(id="owner-1", name="Owner", type="User"),
                status="active",
                type="direct",
                title="Test Room",
                created_at="2025-01-01T00:00:00Z",
                participant_role="member",
            )
        )

        # Should subscribe to both channels
        mock_ws_client.join_chat_room_channel.assert_called_once()
        mock_ws_client.join_room_participants_channel.assert_called_once()

    async def test_room_removed_destroys_session(self, agent, mock_ws_client):
        """room_removed event should destroy the session."""
        await start_agent_with_mock_ws(agent, mock_ws_client, AsyncMock())

        on_room_added = mock_ws_client.join_agent_rooms_channel.call_args.kwargs[
            "on_room_added"
        ]
        on_room_removed = mock_ws_client.join_agent_rooms_channel.call_args.kwargs[
            "on_room_removed"
        ]

        # First add the room
        await on_room_added(
            RoomAddedPayload(
                id="room-123",
                owner=RoomOwner(id="owner-1", name="Owner", type="User"),
                status="active",
                type="direct",
                title="Test Room",
                created_at="2025-01-01T00:00:00Z",
                participant_role="member",
            )
        )
        assert "room-123" in agent._sessions

        # Now remove it
        await on_room_removed(
            RoomRemovedPayload(
                id="room-123",
                status="removed",
                type="direct",
                title="Test Room",
                removed_at="2025-01-01T00:00:00Z",
            )
        )

        assert "room-123" not in agent._sessions

    async def test_room_removed_unsubscribes_from_channels(self, agent, mock_ws_client):
        """room_removed should unsubscribe from room channels."""
        await start_agent_with_mock_ws(agent, mock_ws_client, AsyncMock())

        on_room_added = mock_ws_client.join_agent_rooms_channel.call_args.kwargs[
            "on_room_added"
        ]
        on_room_removed = mock_ws_client.join_agent_rooms_channel.call_args.kwargs[
            "on_room_removed"
        ]

        await on_room_added(
            RoomAddedPayload(
                id="room-123",
                owner=RoomOwner(id="owner-1", name="Owner", type="User"),
                status="active",
                type="direct",
                title="Test Room",
                created_at="2025-01-01T00:00:00Z",
                participant_role="member",
            )
        )

        await on_room_removed(
            RoomRemovedPayload(
                id="room-123",
                status="removed",
                type="direct",
                title="Test Room",
                removed_at="2025-01-01T00:00:00Z",
            )
        )

        mock_ws_client.leave_chat_room_channel.assert_called_with("room-123")
        mock_ws_client.leave_room_participants_channel.assert_called_with("room-123")


class TestCleanupCallback:
    """Tests for session cleanup callback (for checkpointer clearing)."""

    async def test_cleanup_callback_invoked_on_room_removed(
        self, mock_rest_client, mock_ws_client
    ):
        """Cleanup callback should be called when session is destroyed."""
        cleanup_called_for = []

        async def cleanup_callback(room_id: str):
            cleanup_called_for.append(room_id)

        with patch("thenvoi.core.agent.AsyncRestClient", return_value=mock_rest_client):
            agent = ThenvoiAgent(
                agent_id="agent-123",
                api_key="test-key",
                on_session_cleanup=cleanup_callback,
            )

        await start_agent_with_mock_ws(agent, mock_ws_client, AsyncMock())

        on_room_added = mock_ws_client.join_agent_rooms_channel.call_args.kwargs[
            "on_room_added"
        ]
        on_room_removed = mock_ws_client.join_agent_rooms_channel.call_args.kwargs[
            "on_room_removed"
        ]

        # Add then remove room
        await on_room_added(
            RoomAddedPayload(
                id="room-123",
                owner=RoomOwner(id="owner-1", name="Owner", type="User"),
                status="active",
                type="direct",
                title="Test Room",
                created_at="2025-01-01T00:00:00Z",
                participant_role="member",
            )
        )

        await on_room_removed(
            RoomRemovedPayload(
                id="room-123",
                status="removed",
                type="direct",
                title="Test Room",
                removed_at="2025-01-01T00:00:00Z",
            )
        )

        # Cleanup callback should have been called
        assert cleanup_called_for == ["room-123"]

    async def test_cleanup_callback_invoked_on_stop(
        self, mock_rest_client, mock_ws_client
    ):
        """Cleanup callback should be called for all sessions when agent stops."""
        cleanup_called_for = []

        async def cleanup_callback(room_id: str):
            cleanup_called_for.append(room_id)

        with patch("thenvoi.core.agent.AsyncRestClient", return_value=mock_rest_client):
            agent = ThenvoiAgent(
                agent_id="agent-123",
                api_key="test-key",
                on_session_cleanup=cleanup_callback,
            )

        await start_agent_with_mock_ws(agent, mock_ws_client, AsyncMock())

        on_room_added = mock_ws_client.join_agent_rooms_channel.call_args.kwargs[
            "on_room_added"
        ]

        # Add multiple rooms
        for room_id in ["room-1", "room-2", "room-3"]:
            await on_room_added(
                RoomAddedPayload(
                    id=room_id,
                    owner=RoomOwner(id="owner-1", name="Owner", type="User"),
                    status="active",
                    type="direct",
                    title=f"Room {room_id}",
                    created_at="2025-01-01T00:00:00Z",
                    participant_role="member",
                )
            )

        # Stop agent
        await agent.stop()

        # All sessions should be cleaned up
        assert set(cleanup_called_for) == {"room-1", "room-2", "room-3"}


class TestParticipantUpdates:
    """Tests for participant_added/removed events."""

    async def test_participant_added_updates_session(self, agent, mock_ws_client):
        """participant_added should update session's participant list."""
        await start_agent_with_mock_ws(agent, mock_ws_client, AsyncMock())

        on_room_added = mock_ws_client.join_agent_rooms_channel.call_args.kwargs[
            "on_room_added"
        ]

        await on_room_added(
            RoomAddedPayload(
                id="room-123",
                owner=RoomOwner(id="owner-1", name="Owner", type="User"),
                status="active",
                type="direct",
                title="Test Room",
                created_at="2025-01-01T00:00:00Z",
                participant_role="member",
            )
        )

        # Get the participant callback
        participant_call = mock_ws_client.join_room_participants_channel.call_args
        on_participant_added = participant_call.kwargs["on_participant_added"]

        # Simulate participant added
        await on_participant_added(
            {"id": "user-456", "name": "New User", "type": "User"}
        )

        session = agent._sessions["room-123"]
        assert any(p["id"] == "user-456" for p in session.participants)

    async def test_participant_added_ignores_self(self, agent, mock_ws_client):
        """participant_added should ignore the agent itself."""
        await start_agent_with_mock_ws(agent, mock_ws_client, AsyncMock())

        on_room_added = mock_ws_client.join_agent_rooms_channel.call_args.kwargs[
            "on_room_added"
        ]

        await on_room_added(
            RoomAddedPayload(
                id="room-123",
                owner=RoomOwner(id="owner-1", name="Owner", type="User"),
                status="active",
                type="direct",
                title="Test Room",
                created_at="2025-01-01T00:00:00Z",
                participant_role="member",
            )
        )

        participant_call = mock_ws_client.join_room_participants_channel.call_args
        on_participant_added = participant_call.kwargs["on_participant_added"]

        session = agent._sessions["room-123"]
        initial_count = len(session.participants)

        # Simulate agent itself being added (should be ignored)
        await on_participant_added(
            {"id": "agent-123", "name": "TestBot", "type": "Agent"}
        )

        # Count should not change
        assert len(session.participants) == initial_count

    async def test_participant_removed_updates_session(self, agent, mock_ws_client):
        """participant_removed should update session's participant list."""
        await start_agent_with_mock_ws(agent, mock_ws_client, AsyncMock())

        on_room_added = mock_ws_client.join_agent_rooms_channel.call_args.kwargs[
            "on_room_added"
        ]

        await on_room_added(
            RoomAddedPayload(
                id="room-123",
                owner=RoomOwner(id="owner-1", name="Owner", type="User"),
                status="active",
                type="direct",
                title="Test Room",
                created_at="2025-01-01T00:00:00Z",
                participant_role="member",
            )
        )

        participant_call = mock_ws_client.join_room_participants_channel.call_args
        on_participant_added = participant_call.kwargs["on_participant_added"]
        on_participant_removed = participant_call.kwargs["on_participant_removed"]

        # Add then remove a participant
        await on_participant_added(
            {"id": "user-456", "name": "Test User", "type": "User"}
        )
        await on_participant_removed({"id": "user-456", "name": "Test User"})

        session = agent._sessions["room-123"]
        assert not any(p["id"] == "user-456" for p in session.participants)

    async def test_participant_removed_ignores_self(self, agent, mock_ws_client):
        """participant_removed for agent itself should be ignored (handled by room_removed)."""
        await start_agent_with_mock_ws(agent, mock_ws_client, AsyncMock())

        on_room_added = mock_ws_client.join_agent_rooms_channel.call_args.kwargs[
            "on_room_added"
        ]

        await on_room_added(
            RoomAddedPayload(
                id="room-123",
                owner=RoomOwner(id="owner-1", name="Owner", type="User"),
                status="active",
                type="direct",
                title="Test Room",
                created_at="2025-01-01T00:00:00Z",
                participant_role="member",
            )
        )

        participant_call = mock_ws_client.join_room_participants_channel.call_args
        on_participant_removed = participant_call.kwargs["on_participant_removed"]

        # Session should still exist after "removing" the agent via participant_removed
        # (real removal is via room_removed)
        await on_participant_removed({"id": "agent-123", "name": "TestBot"})

        assert "room-123" in agent._sessions  # Session NOT destroyed


class TestGetNextMessage:
    """Tests for _get_next_message handling of /next API responses."""

    @pytest.fixture
    def agent_with_client(self, mock_rest_client, mock_ws_client):
        """Create ThenvoiAgent with accessible mock client for testing."""
        with patch("thenvoi.core.agent.AsyncRestClient") as mock_rest_cls:
            mock_rest_cls.return_value = mock_rest_client
            agent = ThenvoiAgent(
                agent_id="agent-123",
                api_key="test-key",
                ws_url="ws://localhost/ws",
                rest_url="http://localhost",
            )
            agent._api_client = mock_rest_client
            return agent

    async def test_returns_none_on_204_no_content(
        self, agent_with_client, mock_rest_client
    ):
        """204 No Content means no unprocessed messages - return None silently."""
        from thenvoi_rest.core.api_error import ApiError

        # SDK raises ApiError for 204 No Content
        mock_rest_client.agent_api.get_agent_next_message = AsyncMock(
            side_effect=ApiError(status_code=204, body="")
        )

        result = await agent_with_client._get_next_message("room-123")

        assert result is None
        mock_rest_client.agent_api.get_agent_next_message.assert_called_once_with(
            chat_id="room-123"
        )

    async def test_returns_none_on_204_without_warning(
        self, agent_with_client, mock_rest_client, caplog
    ):
        """204 should NOT log a warning (it's expected behavior)."""
        from thenvoi_rest.core.api_error import ApiError
        import logging

        mock_rest_client.agent_api.get_agent_next_message = AsyncMock(
            side_effect=ApiError(status_code=204, body="")
        )

        with caplog.at_level(logging.WARNING):
            await agent_with_client._get_next_message("room-123")

        # No warning should be logged for 204
        assert "Failed to get next message" not in caplog.text

    async def test_returns_platform_message_on_200(
        self, agent_with_client, mock_rest_client
    ):
        """200 with message data should return PlatformMessage."""
        from datetime import datetime, timezone

        msg_data = MagicMock()
        msg_data.id = "msg-123"
        msg_data.chat_room_id = "room-123"
        msg_data.content = "Hello"
        msg_data.sender_id = "user-456"
        msg_data.sender_type = "User"
        msg_data.sender_name = "Test User"
        msg_data.message_type = "text"
        msg_data.metadata = {"key": "value"}
        msg_data.inserted_at = datetime.now(timezone.utc)

        mock_rest_client.agent_api.get_agent_next_message = AsyncMock(
            return_value=MagicMock(data=msg_data)
        )

        result = await agent_with_client._get_next_message("room-123")

        assert result is not None
        assert result.id == "msg-123"
        assert result.room_id == "room-123"
        assert result.content == "Hello"
        assert result.sender_id == "user-456"

    async def test_returns_none_when_data_is_none(
        self, agent_with_client, mock_rest_client
    ):
        """Response with data=None should return None."""
        mock_rest_client.agent_api.get_agent_next_message = AsyncMock(
            return_value=MagicMock(data=None)
        )

        result = await agent_with_client._get_next_message("room-123")

        assert result is None

    async def test_logs_warning_on_other_errors(
        self, agent_with_client, mock_rest_client, caplog
    ):
        """Other errors (not 204) should log a warning."""
        from thenvoi_rest.core.api_error import ApiError
        import logging

        mock_rest_client.agent_api.get_agent_next_message = AsyncMock(
            side_effect=ApiError(status_code=500, body="Internal Server Error")
        )

        with caplog.at_level(logging.WARNING):
            result = await agent_with_client._get_next_message("room-123")

        assert result is None
        assert "Failed to get next message" in caplog.text
