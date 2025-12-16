"""Tests for ThenvoiLink."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from thenvoi.platform.link import ThenvoiLink
from thenvoi.platform.event import PlatformEvent


@pytest.fixture
def mock_ws_client():
    """Mock WebSocketClient for testing ThenvoiLink."""
    ws = AsyncMock()

    # Async context manager support
    ws.__aenter__ = AsyncMock(return_value=ws)
    ws.__aexit__ = AsyncMock(return_value=None)

    # Mock channel operations
    ws.join_chat_room_channel = AsyncMock()
    ws.leave_chat_room_channel = AsyncMock()
    ws.join_agent_rooms_channel = AsyncMock()
    ws.join_room_participants_channel = AsyncMock()
    ws.leave_room_participants_channel = AsyncMock()
    ws.run_forever = AsyncMock()

    return ws


@pytest.fixture
def mock_rest_client():
    """Mock AsyncRestClient for testing ThenvoiLink."""
    client = AsyncMock()
    client.agent_api = MagicMock()
    return client


class TestThenvoiLinkConstruction:
    """Test ThenvoiLink initialization."""

    def test_init_stores_credentials(self):
        """Should store agent_id, api_key, and URLs."""
        link = ThenvoiLink(
            agent_id="agent-123",
            api_key="test-key",
            ws_url="wss://test.com/ws",
            rest_url="https://test.com",
        )

        assert link.agent_id == "agent-123"
        assert link.api_key == "test-key"
        assert link.ws_url == "wss://test.com/ws"
        assert link.rest_url == "https://test.com"

    def test_init_creates_rest_client(self):
        """Should create AsyncRestClient exposed as .rest."""
        link = ThenvoiLink(
            agent_id="agent-123",
            api_key="test-key",
        )

        assert link.rest is not None

    def test_init_starts_disconnected(self):
        """Should start in disconnected state."""
        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")

        assert link.is_connected is False
        assert link._ws is None
        assert link._subscribed_rooms == set()

    def test_init_no_event_handler(self):
        """Should start with no event handler."""
        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")

        assert link.on_event is None


class TestThenvoiLinkConnection:
    """Test connection lifecycle."""

    @patch("thenvoi.platform.link.WebSocketClient")
    async def test_connect_creates_websocket(self, mock_ws_class, mock_ws_client):
        """connect() should create WebSocketClient and enter context."""
        mock_ws_class.return_value = mock_ws_client

        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")
        await link.connect()

        mock_ws_class.assert_called_once_with(link.ws_url, link.api_key, link.agent_id)
        mock_ws_client.__aenter__.assert_called_once()
        assert link.is_connected is True

    @patch("thenvoi.platform.link.WebSocketClient")
    async def test_connect_when_already_connected_logs_warning(
        self, mock_ws_class, mock_ws_client
    ):
        """connect() when already connected should log warning and return."""
        mock_ws_class.return_value = mock_ws_client

        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")
        await link.connect()
        await link.connect()  # Second call

        # Should only create WS once
        assert mock_ws_class.call_count == 1

    @patch("thenvoi.platform.link.WebSocketClient")
    async def test_disconnect_exits_websocket_context(
        self, mock_ws_class, mock_ws_client
    ):
        """disconnect() should exit WebSocket context."""
        mock_ws_class.return_value = mock_ws_client

        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")
        await link.connect()
        await link.disconnect()

        mock_ws_client.__aexit__.assert_called_once_with(None, None, None)
        assert link.is_connected is False
        assert link._ws is None

    @patch("thenvoi.platform.link.WebSocketClient")
    async def test_disconnect_clears_subscribed_rooms(
        self, mock_ws_class, mock_ws_client
    ):
        """disconnect() should clear tracked subscriptions."""
        mock_ws_class.return_value = mock_ws_client

        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")
        await link.connect()
        link._subscribed_rooms.add("room-1")
        link._subscribed_rooms.add("room-2")

        await link.disconnect()

        assert link._subscribed_rooms == set()

    async def test_disconnect_when_not_connected_is_noop(self):
        """disconnect() when not connected should be a no-op."""
        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")
        await link.disconnect()  # Should not raise

        assert link.is_connected is False

    @patch("thenvoi.platform.link.WebSocketClient")
    async def test_run_forever_delegates_to_websocket(
        self, mock_ws_class, mock_ws_client
    ):
        """run_forever() should delegate to WebSocket."""
        mock_ws_class.return_value = mock_ws_client

        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")
        await link.connect()
        await link.run_forever()

        mock_ws_client.run_forever.assert_called_once()

    async def test_run_forever_raises_when_not_connected(self):
        """run_forever() should raise RuntimeError when not connected."""
        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")

        with pytest.raises(RuntimeError, match="Not connected"):
            await link.run_forever()


class TestThenvoiLinkSubscriptions:
    """Test subscription management."""

    @patch("thenvoi.platform.link.WebSocketClient")
    async def test_subscribe_agent_rooms_joins_channel(
        self, mock_ws_class, mock_ws_client
    ):
        """subscribe_agent_rooms() should join agent rooms channel."""
        mock_ws_class.return_value = mock_ws_client

        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")
        await link.connect()
        await link.subscribe_agent_rooms("agent-123")

        mock_ws_client.join_agent_rooms_channel.assert_called_once()
        # Verify callbacks were passed
        call_kwargs = mock_ws_client.join_agent_rooms_channel.call_args[1]
        assert "on_room_added" in call_kwargs
        assert "on_room_removed" in call_kwargs

    async def test_subscribe_agent_rooms_raises_when_not_connected(self):
        """subscribe_agent_rooms() should raise when not connected."""
        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")

        with pytest.raises(RuntimeError, match="Not connected"):
            await link.subscribe_agent_rooms("agent-123")

    @patch("thenvoi.platform.link.WebSocketClient")
    async def test_subscribe_room_joins_channels(self, mock_ws_class, mock_ws_client):
        """subscribe_room() should join chat room and participants channels."""
        mock_ws_class.return_value = mock_ws_client

        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")
        await link.connect()
        await link.subscribe_room("room-123")

        mock_ws_client.join_chat_room_channel.assert_called_once()
        mock_ws_client.join_room_participants_channel.assert_called_once()

    @patch("thenvoi.platform.link.WebSocketClient")
    async def test_subscribe_room_tracks_subscription(
        self, mock_ws_class, mock_ws_client
    ):
        """subscribe_room() should track room in _subscribed_rooms."""
        mock_ws_class.return_value = mock_ws_client

        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")
        await link.connect()
        await link.subscribe_room("room-123")

        assert "room-123" in link._subscribed_rooms

    @patch("thenvoi.platform.link.WebSocketClient")
    async def test_subscribe_room_idempotent(self, mock_ws_class, mock_ws_client):
        """subscribe_room() twice should not re-subscribe."""
        mock_ws_class.return_value = mock_ws_client

        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")
        await link.connect()
        await link.subscribe_room("room-123")
        await link.subscribe_room("room-123")  # Second call

        # Should only join once
        assert mock_ws_client.join_chat_room_channel.call_count == 1

    async def test_subscribe_room_raises_when_not_connected(self):
        """subscribe_room() should raise when not connected."""
        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")

        with pytest.raises(RuntimeError, match="Not connected"):
            await link.subscribe_room("room-123")

    @patch("thenvoi.platform.link.WebSocketClient")
    async def test_unsubscribe_room_leaves_channels(
        self, mock_ws_class, mock_ws_client
    ):
        """unsubscribe_room() should leave both channels."""
        mock_ws_class.return_value = mock_ws_client

        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")
        await link.connect()
        await link.subscribe_room("room-123")
        await link.unsubscribe_room("room-123")

        mock_ws_client.leave_chat_room_channel.assert_called_once_with("room-123")
        mock_ws_client.leave_room_participants_channel.assert_called_once_with(
            "room-123"
        )

    @patch("thenvoi.platform.link.WebSocketClient")
    async def test_unsubscribe_room_removes_from_tracking(
        self, mock_ws_class, mock_ws_client
    ):
        """unsubscribe_room() should remove room from _subscribed_rooms."""
        mock_ws_class.return_value = mock_ws_client

        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")
        await link.connect()
        await link.subscribe_room("room-123")
        await link.unsubscribe_room("room-123")

        assert "room-123" not in link._subscribed_rooms

    @patch("thenvoi.platform.link.WebSocketClient")
    async def test_unsubscribe_room_handles_leave_errors(
        self, mock_ws_class, mock_ws_client
    ):
        """unsubscribe_room() should handle errors gracefully."""
        mock_ws_class.return_value = mock_ws_client
        mock_ws_client.leave_chat_room_channel.side_effect = Exception("Leave failed")

        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")
        await link.connect()
        link._subscribed_rooms.add("room-123")

        # Should not raise, just log warning
        await link.unsubscribe_room("room-123")

        # Room should still be removed from tracking
        assert "room-123" not in link._subscribed_rooms

    async def test_unsubscribe_room_noop_when_not_subscribed(self):
        """unsubscribe_room() should be no-op for unsubscribed room."""
        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")

        # Should not raise
        await link.unsubscribe_room("room-123")


class TestThenvoiLinkEventDispatch:
    """Test event dispatch mechanism."""

    @patch("thenvoi.platform.link.WebSocketClient")
    async def test_dispatch_calls_on_event_handler(self, mock_ws_class, mock_ws_client):
        """_dispatch() should call on_event callback."""
        mock_ws_class.return_value = mock_ws_client

        handler_called = asyncio.Event()
        received_event = None

        async def handler(event):
            nonlocal received_event
            received_event = event
            handler_called.set()

        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")
        link.on_event = handler

        event = PlatformEvent(type="test", room_id="room-123", payload={"data": "test"})
        await link._dispatch(event)

        # Wait for async dispatch
        await asyncio.wait_for(handler_called.wait(), timeout=1.0)

        assert received_event is event

    @patch("thenvoi.platform.link.WebSocketClient")
    async def test_dispatch_without_handler_is_noop(
        self, mock_ws_class, mock_ws_client
    ):
        """_dispatch() without on_event handler should be no-op."""
        mock_ws_class.return_value = mock_ws_client

        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")
        link.on_event = None

        event = PlatformEvent(type="test", room_id="room-123", payload={})
        # Should not raise
        await link._dispatch(event)

    @patch("thenvoi.platform.link.WebSocketClient")
    async def test_dispatch_handles_handler_errors(self, mock_ws_class, mock_ws_client):
        """_dispatch() should catch handler exceptions."""
        mock_ws_class.return_value = mock_ws_client

        error_logged = asyncio.Event()

        async def failing_handler(event):
            error_logged.set()
            raise ValueError("Handler error")

        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")
        link.on_event = failing_handler

        event = PlatformEvent(type="test", room_id="room-123", payload={})
        # Should not raise
        await link._dispatch(event)

        # Wait for handler to be called
        await asyncio.wait_for(error_logged.wait(), timeout=1.0)


class TestThenvoiLinkEventHandlers:
    """Test internal event handlers that create PlatformEvent."""

    @patch("thenvoi.platform.link.WebSocketClient")
    async def test_on_room_added_creates_platform_event(
        self, mock_ws_class, mock_ws_client
    ):
        """_on_room_added() should create PlatformEvent with type room_added."""
        mock_ws_class.return_value = mock_ws_client

        received_event = None
        event_received = asyncio.Event()

        async def handler(event):
            nonlocal received_event
            received_event = event
            event_received.set()

        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")
        link.on_event = handler

        # Create mock payload
        payload = MagicMock()
        payload.id = "room-123"
        payload.model_dump.return_value = {
            "id": "room-123",
            "title": "Test Room",
            "owner": {"id": "u1", "name": "User", "type": "User"},
            "status": "active",
            "type": "direct",
            "created_at": "2024-01-01T00:00:00Z",
            "participant_role": "member",
        }

        await link._on_room_added(payload)
        await asyncio.wait_for(event_received.wait(), timeout=1.0)

        assert received_event is not None
        assert received_event.type == "room_added"
        assert received_event.room_id == "room-123"

    @patch("thenvoi.platform.link.WebSocketClient")
    async def test_on_room_removed_creates_platform_event(
        self, mock_ws_class, mock_ws_client
    ):
        """_on_room_removed() should create PlatformEvent with type room_removed."""
        mock_ws_class.return_value = mock_ws_client

        received_event = None
        event_received = asyncio.Event()

        async def handler(event):
            nonlocal received_event
            received_event = event
            event_received.set()

        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")
        link.on_event = handler

        payload = MagicMock()
        payload.id = "room-123"
        payload.model_dump.return_value = {"id": "room-123"}

        await link._on_room_removed(payload)
        await asyncio.wait_for(event_received.wait(), timeout=1.0)

        assert received_event is not None
        assert received_event.type == "room_removed"
        assert received_event.room_id == "room-123"

    @patch("thenvoi.platform.link.WebSocketClient")
    async def test_on_message_created_creates_platform_event(
        self, mock_ws_class, mock_ws_client
    ):
        """_on_message_created() should create PlatformEvent with type message_created."""
        mock_ws_class.return_value = mock_ws_client

        received_event = None
        event_received = asyncio.Event()

        async def handler(event):
            nonlocal received_event
            received_event = event
            event_received.set()

        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")
        link.on_event = handler

        payload = MagicMock()
        payload.model_dump.return_value = {
            "id": "msg-123",
            "content": "Hello",
            "sender_id": "user-456",
        }

        await link._on_message_created("room-123", payload)
        await asyncio.wait_for(event_received.wait(), timeout=1.0)

        assert received_event is not None
        assert received_event.type == "message_created"
        assert received_event.room_id == "room-123"
        assert received_event.payload["content"] == "Hello"

    @patch("thenvoi.platform.link.WebSocketClient")
    async def test_on_participant_added_creates_platform_event(
        self, mock_ws_class, mock_ws_client
    ):
        """_on_participant_added() should create PlatformEvent."""
        mock_ws_class.return_value = mock_ws_client

        received_event = None
        event_received = asyncio.Event()

        async def handler(event):
            nonlocal received_event
            received_event = event
            event_received.set()

        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")
        link.on_event = handler

        payload = {"id": "user-123", "name": "Test User", "type": "User"}

        await link._on_participant_added("room-123", payload)
        await asyncio.wait_for(event_received.wait(), timeout=1.0)

        assert received_event is not None
        assert received_event.type == "participant_added"
        assert received_event.room_id == "room-123"
        assert received_event.payload["id"] == "user-123"

    @patch("thenvoi.platform.link.WebSocketClient")
    async def test_on_participant_removed_creates_platform_event(
        self, mock_ws_class, mock_ws_client
    ):
        """_on_participant_removed() should create PlatformEvent."""
        mock_ws_class.return_value = mock_ws_client

        received_event = None
        event_received = asyncio.Event()

        async def handler(event):
            nonlocal received_event
            received_event = event
            event_received.set()

        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")
        link.on_event = handler

        payload = {"id": "user-123", "name": "Test User", "type": "User"}

        await link._on_participant_removed("room-123", payload)
        await asyncio.wait_for(event_received.wait(), timeout=1.0)

        assert received_event is not None
        assert received_event.type == "participant_removed"
        assert received_event.room_id == "room-123"
