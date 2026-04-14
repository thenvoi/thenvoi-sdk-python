"""Tests for WebSocket disconnect reason surfacing.

Covers:
- phx_close / phx_error parsing in WebSocketClient
- Reason extraction from various payload shapes
- Human-readable reason mapping (known + unknown)
- Transport-level disconnect reason extraction
- DisconnectedEvent propagation through ThenvoiLink
- DisconnectedEvent routing in RoomPresence
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from thenvoi.client.streaming.client import (
    KNOWN_DISCONNECT_REASONS,
    WebSocketClient,
    extract_disconnect_reason,
    humanize_disconnect_reason,
)
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close
from thenvoi.platform.event import DisconnectedEvent
from thenvoi.platform.link import ThenvoiLink
from thenvoi.runtime.presence import RoomPresence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakePHXMessage:
    """Minimal stand-in for phoenix_channels_python_client.phx_messages.PHXMessage."""

    event: str
    topic: str
    payload: dict
    ref: str | None = None
    join_ref: str | None = None


# ---------------------------------------------------------------------------
# extract_disconnect_reason
# ---------------------------------------------------------------------------


class TestExtractDisconnectReason:
    """Test reason extraction from various Phoenix payload shapes."""

    def test_reason_field(self):
        assert extract_disconnect_reason({"reason": "replaced"}) == "replaced"

    def test_nested_response_reason(self):
        payload = {"response": {"reason": "kicked"}}
        assert extract_disconnect_reason(payload) == "kicked"

    def test_message_field(self):
        assert extract_disconnect_reason({"message": "bye"}) == "bye"

    def test_empty_payload(self):
        assert extract_disconnect_reason({}) == "unknown"

    def test_prefers_reason_over_message(self):
        payload = {"reason": "replaced", "message": "other"}
        assert extract_disconnect_reason(payload) == "replaced"

    def test_non_string_reason_coerced(self):
        assert extract_disconnect_reason({"reason": 42}) == "42"


# ---------------------------------------------------------------------------
# humanize_disconnect_reason
# ---------------------------------------------------------------------------


class TestHumanizeDisconnectReason:
    """Test human-readable message generation."""

    def test_known_reason(self):
        result = humanize_disconnect_reason("replaced")
        assert "same agent ID" in result

    def test_all_known_reasons_produce_output(self):
        for key in KNOWN_DISCONNECT_REASONS:
            assert humanize_disconnect_reason(key)

    def test_unknown_reason_with_raw_payload(self):
        result = humanize_disconnect_reason("custom_err", {"code": 99})
        assert "custom_err" in result
        assert "99" in result

    def test_unknown_reason_without_payload(self):
        result = humanize_disconnect_reason("custom_err")
        assert result == "Disconnected: custom_err"


# ---------------------------------------------------------------------------
# Transport reason extraction (tested via _on_transport_disconnect)
# ---------------------------------------------------------------------------


class TestExtractTransportReason:
    """Test transport-level disconnect reason extraction via _on_transport_disconnect."""

    @pytest.mark.asyncio
    async def test_none_error(self):
        on_disconnect = AsyncMock()
        client = WebSocketClient(
            ws_url="wss://test.com/ws", api_key="key", on_disconnect=on_disconnect
        )
        await client._on_transport_disconnect(None)
        human, _, _ = on_disconnect.call_args[0]
        assert "connection_closed" in human

    @pytest.mark.asyncio
    async def test_connection_closed_with_reason(self):
        on_disconnect = AsyncMock()
        client = WebSocketClient(
            ws_url="wss://test.com/ws", api_key="key", on_disconnect=on_disconnect
        )
        err = ConnectionClosedError(Close(4001, "replaced"), None)
        assert err.rcvd is not None and err.rcvd.reason == "replaced"
        await client._on_transport_disconnect(err)
        human, _, _ = on_disconnect.call_args[0]
        assert "same agent ID" in human

    @pytest.mark.asyncio
    async def test_connection_closed_with_code_only(self):
        on_disconnect = AsyncMock()
        client = WebSocketClient(
            ws_url="wss://test.com/ws", api_key="key", on_disconnect=on_disconnect
        )
        err = ConnectionClosedError(Close(4001, ""), None)
        assert err.rcvd is not None and err.rcvd.reason == ""
        await client._on_transport_disconnect(err)
        human, _, _ = on_disconnect.call_args[0]
        assert "4001" in human

    @pytest.mark.asyncio
    async def test_generic_exception(self):
        on_disconnect = AsyncMock()
        client = WebSocketClient(
            ws_url="wss://test.com/ws", api_key="key", on_disconnect=on_disconnect
        )
        await client._on_transport_disconnect(RuntimeError("oops"))
        human, _, _ = on_disconnect.call_args[0]
        assert "oops" in human


# ---------------------------------------------------------------------------
# WebSocketClient._handle_events — phx_close / phx_error interception
# ---------------------------------------------------------------------------


class TestWebSocketClientHandlePhxEvents:
    """Test that phx_close and phx_error are intercepted in _handle_events."""

    @pytest.fixture
    def ws_client(self):
        on_disconnect = AsyncMock()
        client = WebSocketClient(
            ws_url="wss://test.com/ws",
            api_key="key",
            agent_id="agent-1",
            on_disconnect=on_disconnect,
        )
        return client, on_disconnect

    @pytest.mark.asyncio
    async def test_phx_close_triggers_callback(self, ws_client):
        client, on_disconnect = ws_client
        msg = FakePHXMessage(
            event="phx_close",
            topic="agent_rooms:agent-1",
            payload={"reason": "replaced"},
        )
        await client._handle_events(msg, {"room_added": AsyncMock()})  # type: ignore[arg-type]

        on_disconnect.assert_awaited_once()
        human, raw, topic = on_disconnect.call_args[0]
        assert "same agent ID" in human
        assert raw == {"reason": "replaced"}
        assert topic == "agent_rooms:agent-1"

    @pytest.mark.asyncio
    async def test_phx_error_triggers_callback(self, ws_client):
        client, on_disconnect = ws_client
        msg = FakePHXMessage(
            event="phx_error",
            topic="chat_room:room-1",
            payload={"response": {"reason": "unauthorized"}},
        )
        await client._handle_events(msg, {})  # type: ignore[arg-type]

        on_disconnect.assert_awaited_once()
        human, _, topic = on_disconnect.call_args[0]
        assert "Not authorized" in human
        assert topic == "chat_room:room-1"

    @pytest.mark.asyncio
    async def test_phx_close_unknown_reason_includes_raw(self, ws_client):
        client, on_disconnect = ws_client
        msg = FakePHXMessage(
            event="phx_close",
            topic="agent_rooms:agent-1",
            payload={"weird_field": "data"},
        )
        await client._handle_events(msg, {})  # type: ignore[arg-type]

        on_disconnect.assert_awaited_once()
        human, raw, _ = on_disconnect.call_args[0]
        assert "unknown" in human
        assert "weird_field" in str(raw)

    @pytest.mark.asyncio
    async def test_phx_close_does_not_call_event_handlers(self, ws_client):
        """phx_close should be intercepted before event handler dispatch."""
        client, _ = ws_client
        handler = AsyncMock()
        msg = FakePHXMessage(
            event="phx_close",
            topic="agent_rooms:agent-1",
            payload={"reason": "replaced"},
        )
        await client._handle_events(msg, {"phx_close": handler})  # type: ignore[arg-type]

        handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_callback_does_not_raise(self):
        """If on_disconnect is None, phx_close should still be handled gracefully."""
        client = WebSocketClient(
            ws_url="wss://test.com/ws",
            api_key="key",
        )
        msg = FakePHXMessage(
            event="phx_close",
            topic="agent_rooms:agent-1",
            payload={"reason": "replaced"},
        )
        # Should not raise
        await client._handle_events(msg, {})  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_callback_error_is_swallowed(self, ws_client):
        """Errors in the disconnect callback must not propagate."""
        client, on_disconnect = ws_client
        on_disconnect.side_effect = RuntimeError("boom")
        msg = FakePHXMessage(
            event="phx_close",
            topic="agent_rooms:agent-1",
            payload={"reason": "replaced"},
        )
        # Should not raise
        await client._handle_events(msg, {})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# WebSocketClient — transport disconnect handler
# ---------------------------------------------------------------------------


class TestWebSocketClientTransportDisconnect:
    """Test _on_transport_disconnect handler."""

    @pytest.mark.asyncio
    async def test_transport_disconnect_calls_callback(self):
        on_disconnect = AsyncMock()
        client = WebSocketClient(
            ws_url="wss://test.com/ws",
            api_key="key",
            on_disconnect=on_disconnect,
        )
        err = ConnectionClosedError(Close(4001, "replaced"), None)
        assert err.rcvd is not None and err.rcvd.reason == "replaced"
        await client._on_transport_disconnect(err)

        on_disconnect.assert_awaited_once()
        human, raw, topic = on_disconnect.call_args[0]
        assert "same agent ID" in human
        assert raw is not None
        assert topic is None  # transport-level has no topic

    @pytest.mark.asyncio
    async def test_transport_disconnect_none_error(self):
        on_disconnect = AsyncMock()
        client = WebSocketClient(
            ws_url="wss://test.com/ws",
            api_key="key",
            on_disconnect=on_disconnect,
        )
        await client._on_transport_disconnect(None)

        on_disconnect.assert_awaited_once()
        human, raw, topic = on_disconnect.call_args[0]
        assert "connection_closed" in human
        assert raw is None
        assert topic is None


# ---------------------------------------------------------------------------
# ThenvoiLink._on_ws_disconnect
# ---------------------------------------------------------------------------


class TestThenvoiLinkDisconnect:
    """Test ThenvoiLink queues DisconnectedEvent on disconnect."""

    @pytest.mark.asyncio
    async def test_queues_disconnected_event_without_topic(self):
        link = ThenvoiLink(agent_id="agent-1", api_key="key")
        link._is_connected = True

        await link._on_ws_disconnect("Server closed normally", {"reason": "normal"})

        assert link._is_connected is False
        assert not link._event_queue.empty()
        event = link._event_queue.get_nowait()
        assert isinstance(event, DisconnectedEvent)
        assert event.reason == "Server closed normally"
        assert event.raw == {"reason": "normal"}
        assert event.room_id is None

    @pytest.mark.asyncio
    async def test_queues_disconnected_event_with_chat_room_topic(self):
        link = ThenvoiLink(agent_id="agent-1", api_key="key")
        link._is_connected = True

        await link._on_ws_disconnect(
            "Not authorized",
            {"response": {"reason": "unauthorized"}},
            "chat_room:room-42",
        )

        event = link._event_queue.get_nowait()
        assert isinstance(event, DisconnectedEvent)
        assert event.room_id == "room-42"

    @pytest.mark.asyncio
    async def test_queues_disconnected_event_with_non_chat_topic(self):
        link = ThenvoiLink(agent_id="agent-1", api_key="key")
        link._is_connected = True

        await link._on_ws_disconnect("replaced", None, "agent_rooms:agent-1")

        event = link._event_queue.get_nowait()
        assert event.room_id is None  # agent_rooms topic doesn't map to room_id

    @pytest.mark.asyncio
    async def test_disconnect_sets_is_connected_false(self):
        link = ThenvoiLink(agent_id="agent-1", api_key="key")
        link._is_connected = True

        await link._on_ws_disconnect("replaced", None)

        assert link._is_connected is False

    @pytest.mark.asyncio
    async def test_second_transport_disconnect_is_deduped(self):
        """Only the first transport-level disconnect should queue an event."""
        link = ThenvoiLink(agent_id="agent-1", api_key="key")
        link._is_connected = True

        await link._on_ws_disconnect("replaced", {"reason": "replaced"})
        await link._on_ws_disconnect("transport closed", None)

        # Only one event should be queued (both are transport-level, no topic)
        assert link._event_queue.qsize() == 1
        event = link._event_queue.get_nowait()
        assert event.reason == "replaced"

    @pytest.mark.asyncio
    async def test_channel_disconnect_does_not_flip_is_connected(self):
        """Channel-level disconnects should not change _is_connected."""
        link = ThenvoiLink(agent_id="agent-1", api_key="key")
        link._is_connected = True

        await link._on_ws_disconnect("unauthorized", None, "chat_room:room-1")

        assert link._is_connected is True
        assert link._event_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_multiple_channel_disconnects_all_surface(self):
        """Each channel-level disconnect should produce an event."""
        link = ThenvoiLink(agent_id="agent-1", api_key="key")
        link._is_connected = True

        await link._on_ws_disconnect("Not authorized", None, "chat_room:room-1")
        await link._on_ws_disconnect(
            "Another instance connected", None, "agent_rooms:agent-1"
        )

        assert link._event_queue.qsize() == 2
        e1 = link._event_queue.get_nowait()
        e2 = link._event_queue.get_nowait()
        assert e1.reason == "Not authorized"
        assert e1.room_id == "room-1"
        assert e2.reason == "Another instance connected"
        assert e2.room_id is None  # agent_rooms doesn't map to room_id

    @pytest.mark.asyncio
    async def test_channel_then_transport_produces_two_events(self):
        """A channel close followed by transport drop should both fire."""
        link = ThenvoiLink(agent_id="agent-1", api_key="key")
        link._is_connected = True

        # Channel-level phx_close (with topic)
        await link._on_ws_disconnect(
            "Another instance connected", {"reason": "replaced"}, "agent_rooms:agent-1"
        )
        # Transport-level disconnect (no topic)
        await link._on_ws_disconnect("connection_closed", None)

        assert link._event_queue.qsize() == 2
        assert link._is_connected is False


# ---------------------------------------------------------------------------
# AgentRuntime — execution cleanup on disconnect
# ---------------------------------------------------------------------------


class TestRuntimeDisconnect:
    """Test that AgentRuntime cancels executions on disconnect."""

    @pytest.fixture
    def mock_link(self):
        link = MagicMock()
        link.agent_id = "agent-1"
        link.is_connected = True
        link.connect = AsyncMock()
        link.subscribe_agent_rooms = AsyncMock()
        link.subscribe_room = AsyncMock()
        link.unsubscribe_room = AsyncMock()
        link.run_forever = AsyncMock()
        link.rest = MagicMock()
        link.rest.agent_api_chats = MagicMock()
        link.rest.agent_api_chats.list_agent_chats = AsyncMock(
            return_value=MagicMock(data=[])
        )
        return link

    @pytest.mark.asyncio
    async def test_on_disconnected_cancels_executions(self, mock_link):
        from thenvoi.runtime.runtime import AgentRuntime

        runtime = AgentRuntime(
            link=mock_link, agent_id="agent-1", on_execute=AsyncMock()
        )

        # Add mock executions
        exec1 = MagicMock()
        exec1.stop = AsyncMock(return_value=True)
        exec2 = MagicMock()
        exec2.stop = AsyncMock(return_value=True)
        runtime.executions["room-1"] = exec1
        runtime.executions["room-2"] = exec2

        await runtime._on_disconnected("replaced")

        # Both executions should have been stopped with timeout=None (immediate)
        exec1.stop.assert_awaited_once_with(timeout=None)
        exec2.stop.assert_awaited_once_with(timeout=None)
        assert len(runtime.executions) == 0

    @pytest.mark.asyncio
    async def test_on_disconnected_handles_stop_errors(self, mock_link):
        from thenvoi.runtime.runtime import AgentRuntime

        runtime = AgentRuntime(
            link=mock_link, agent_id="agent-1", on_execute=AsyncMock()
        )

        # First execution raises on stop, second should still be stopped
        exec1 = MagicMock()
        exec1.stop = AsyncMock(side_effect=RuntimeError("boom"))
        exec2 = MagicMock()
        exec2.stop = AsyncMock(return_value=True)
        runtime.executions["room-1"] = exec1
        runtime.executions["room-2"] = exec2

        # Should not raise
        await runtime._on_disconnected("replaced")

        exec1.stop.assert_awaited_once()
        exec2.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_on_disconnected_calls_session_cleanup(self, mock_link):
        from thenvoi.runtime.runtime import AgentRuntime

        cleanup = AsyncMock()
        runtime = AgentRuntime(
            link=mock_link,
            agent_id="agent-1",
            on_execute=AsyncMock(),
            on_session_cleanup=cleanup,
        )

        exec1 = MagicMock()
        exec1.stop = AsyncMock(return_value=True)
        runtime.executions["room-1"] = exec1

        await runtime._on_disconnected("replaced")

        cleanup.assert_awaited_once_with("room-1")

    @pytest.mark.asyncio
    async def test_on_disconnected_no_executions(self, mock_link):
        """Should handle empty executions gracefully."""
        from thenvoi.runtime.runtime import AgentRuntime

        runtime = AgentRuntime(
            link=mock_link, agent_id="agent-1", on_execute=AsyncMock()
        )

        await runtime._on_disconnected("replaced")  # Should not raise


# ---------------------------------------------------------------------------
# RoomPresence — DisconnectedEvent routing
# ---------------------------------------------------------------------------


class TestPresenceDisconnectedEvent:
    """Test that DisconnectedEvent is routed to on_disconnected callback."""

    @pytest.fixture
    def mock_link(self):
        link = MagicMock()
        link.agent_id = "agent-123"
        link.is_connected = True
        link.connect = AsyncMock()
        link.subscribe_agent_rooms = AsyncMock()
        link.subscribe_room = AsyncMock()
        link.unsubscribe_room = AsyncMock()
        link.rest = MagicMock()
        link.rest.agent_api_chats = MagicMock()
        link.rest.agent_api_chats.list_agent_chats = AsyncMock(
            return_value=MagicMock(data=[])
        )
        return link

    @pytest.mark.asyncio
    async def test_disconnected_event_calls_callback(self, mock_link):
        presence = RoomPresence(mock_link)
        callback = AsyncMock()
        presence.on_disconnected = callback

        event = DisconnectedEvent(reason="replaced by another connection")
        await presence._on_platform_event(event)

        callback.assert_awaited_once_with("replaced by another connection")

    @pytest.mark.asyncio
    async def test_disconnected_event_no_callback(self, mock_link):
        """Should not raise if no callback is set."""
        presence = RoomPresence(mock_link)
        event = DisconnectedEvent(reason="test")
        await presence._on_platform_event(event)  # No error

    @pytest.mark.asyncio
    async def test_disconnected_event_callback_error_swallowed(self, mock_link):
        """Errors in on_disconnected must not propagate."""
        presence = RoomPresence(mock_link)
        callback = AsyncMock(side_effect=RuntimeError("boom"))
        presence.on_disconnected = callback

        event = DisconnectedEvent(reason="test")
        await presence._on_platform_event(event)  # Should not raise
