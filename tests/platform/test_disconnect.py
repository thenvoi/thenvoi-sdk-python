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
    _extract_transport_reason,
)
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
# _extract_transport_reason
# ---------------------------------------------------------------------------


class TestExtractTransportReason:
    """Test transport-level disconnect reason extraction."""

    def test_none_error(self):
        assert _extract_transport_reason(None) == "connection_closed"

    def test_error_with_reason_attr(self):
        err = Exception("boom")
        err.reason = "replaced"  # type: ignore[attr-defined]
        assert _extract_transport_reason(err) == "replaced"

    def test_error_with_code_attr(self):
        err = Exception("boom")
        err.code = 4001  # type: ignore[attr-defined]
        assert _extract_transport_reason(err) == "close_code_4001"

    def test_error_with_empty_reason_falls_to_code(self):
        err = Exception("boom")
        err.reason = ""  # type: ignore[attr-defined]
        err.code = 4001  # type: ignore[attr-defined]
        assert _extract_transport_reason(err) == "close_code_4001"

    def test_generic_exception(self):
        assert _extract_transport_reason(RuntimeError("oops")) == "oops"


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
        human, raw = on_disconnect.call_args[0]
        assert "same agent ID" in human
        assert raw == {"reason": "replaced"}

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
        human, _ = on_disconnect.call_args[0]
        assert "Not authorized" in human

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
        human, raw = on_disconnect.call_args[0]
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
        err = Exception("connection reset")
        err.reason = "replaced"  # type: ignore[attr-defined]
        await client._on_transport_disconnect(err)

        on_disconnect.assert_awaited_once()
        human, raw = on_disconnect.call_args[0]
        assert "same agent ID" in human
        assert raw is not None

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
        human, raw = on_disconnect.call_args[0]
        assert "connection_closed" in human
        assert raw is None


# ---------------------------------------------------------------------------
# ThenvoiLink._on_ws_disconnect
# ---------------------------------------------------------------------------


class TestThenvoiLinkDisconnect:
    """Test ThenvoiLink queues DisconnectedEvent on disconnect."""

    @pytest.mark.asyncio
    async def test_queues_disconnected_event(self):
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
    async def test_disconnect_sets_is_connected_false(self):
        link = ThenvoiLink(agent_id="agent-1", api_key="key")
        link._is_connected = True

        await link._on_ws_disconnect("replaced", None)

        assert link._is_connected is False

    @pytest.mark.asyncio
    async def test_second_disconnect_is_ignored(self):
        """Only the first disconnect should queue an event."""
        link = ThenvoiLink(agent_id="agent-1", api_key="key")
        link._is_connected = True

        await link._on_ws_disconnect("replaced", {"reason": "replaced"})
        await link._on_ws_disconnect("transport closed", None)

        # Only one event should be queued
        assert link._event_queue.qsize() == 1
        event = link._event_queue.get_nowait()
        assert event.reason == "replaced"


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
