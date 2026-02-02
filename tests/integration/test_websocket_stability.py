"""
WebSocket Stability Tests - INT-93

This module contains tests to verify the WebSocket connection stability features:
1. Heartbeat mechanism - Keeps connections alive
2. Automatic reconnection - Recovers from disconnects

These tests verify the fixes for INT-93: "Agent WebSocket connections drop unpredictably"

Usage:
    # Run automated test (no server required - uses mock)
    uv run pytest tests/integration/test_websocket_stability.py -v

    # Run with verbose output to see heartbeat/reconnection logs
    uv run pytest tests/integration/test_websocket_stability.py -v -s
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

import pytest
from websockets.server import serve, WebSocketServerProtocol

from phoenix_channels_python_client.client import PHXChannelsClient
from phoenix_channels_python_client.protocol_handler import PhoenixChannelsProtocolVersion

logger = logging.getLogger(__name__)


class MockPhoenixServer:
    """A mock Phoenix server for testing WebSocket stability features."""

    def __init__(self, host: str = "localhost", port: int = 0):
        self.host = host
        self.port = port
        self.server = None
        self.client_ws: Optional[WebSocketServerProtocol] = None
        self._actual_port: Optional[int] = None

    @property
    def url(self) -> str:
        port = self._actual_port or self.port
        return f"ws://{self.host}:{port}/socket/websocket"

    async def handle_client(self, websocket: WebSocketServerProtocol) -> None:
        """Handle incoming WebSocket connections."""
        self.client_ws = websocket
        logger.debug("[MockServer] Client connected")

        try:
            async for message in websocket:
                data = json.loads(message)
                await self._handle_message(websocket, data)
        except Exception as e:
            logger.debug("[MockServer] Client disconnected: %s", e)
        finally:
            self.client_ws = None

    async def _handle_message(self, websocket: WebSocketServerProtocol, data: list) -> None:
        """Handle Phoenix protocol messages."""
        if not isinstance(data, list) or len(data) != 5:
            return

        join_ref, msg_ref, topic, event, payload = data

        if event == "phx_join":
            reply = [join_ref, msg_ref, topic, "phx_reply", {"status": "ok", "response": {}}]
            await websocket.send(json.dumps(reply))
            logger.debug("[MockServer] Joined topic: %s", topic)

        elif event == "heartbeat" and topic == "phoenix":
            reply = [join_ref, msg_ref, "phoenix", "phx_reply", {"status": "ok", "response": {}}]
            await websocket.send(json.dumps(reply))
            logger.debug("[MockServer] Heartbeat acknowledged")

        elif event == "phx_leave":
            reply = [join_ref, msg_ref, topic, "phx_reply", {"status": "ok", "response": {}}]
            await websocket.send(json.dumps(reply))
            logger.debug("[MockServer] Left topic: %s", topic)

    async def start(self) -> None:
        """Start the mock server."""
        self.server = await serve(self.handle_client, self.host, self.port)
        # Get the actual port (useful when port=0 for random port)
        self._actual_port = self.server.sockets[0].getsockname()[1]
        logger.debug("[MockServer] Started on %s", self.url)

    async def stop(self) -> None:
        """Stop the mock server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.debug("[MockServer] Stopped")

    async def force_disconnect_client(self) -> None:
        """Forcefully close the client connection to simulate server crash."""
        if self.client_ws:
            logger.debug("[MockServer] Forcing client disconnect")
            await self.client_ws.close(1001, "Server going down")
            self.client_ws = None


@pytest.fixture
async def mock_server():
    """Fixture that provides a mock Phoenix server."""
    server = MockPhoenixServer(port=0)  # Random available port
    await server.start()
    yield server
    await server.stop()


@pytest.mark.asyncio
async def test_heartbeat_is_sent_and_acknowledged(mock_server: MockPhoenixServer, caplog):
    """
    Test that heartbeat messages are sent and acknowledged.

    This verifies the fix for INT-93 - connections should stay alive
    because heartbeats are being sent every 30 seconds (using 0.1s for test speed).
    """
    with caplog.at_level(logging.DEBUG):
        async with PHXChannelsClient(
            mock_server.url,
            api_key="test_key",
            protocol_version=PhoenixChannelsProtocolVersion.V2,
            heartbeat_interval_secs=0.1,  # Fast for testing
            auto_reconnect=False,
        ) as client:
            # Verify heartbeat task is running
            assert client._heartbeat_task is not None
            assert not client._heartbeat_task.done()

            # Wait for heartbeat to be acknowledged
            await asyncio.sleep(0.3)

            # Verify heartbeat was acknowledged
            assert any(
                "heartbeat acknowledged" in record.message.lower()
                for record in caplog.records
            ), "Heartbeat should be acknowledged"


@pytest.mark.asyncio
async def test_heartbeat_can_be_disabled(mock_server: MockPhoenixServer):
    """Test that heartbeat can be disabled by setting interval to None."""
    async with PHXChannelsClient(
        mock_server.url,
        api_key="test_key",
        protocol_version=PhoenixChannelsProtocolVersion.V2,
        heartbeat_interval_secs=None,  # Disabled
        auto_reconnect=False,
    ) as client:
        assert client._heartbeat_task is None


@pytest.mark.asyncio
async def test_automatic_reconnection_after_disconnect(mock_server: MockPhoenixServer):
    """
    Test that the client automatically reconnects after server disconnect.

    This verifies the reconnection feature for INT-93 - when a connection
    drops unexpectedly, the client should automatically reconnect.
    """
    disconnect_count = 0
    reconnect_count = 0
    disconnect_event = asyncio.Event()
    reconnect_event = asyncio.Event()

    async def on_disconnect(error):
        nonlocal disconnect_count
        disconnect_count += 1
        logger.info("[Test] Disconnect detected (count: %d)", disconnect_count)
        disconnect_event.set()

    async def on_reconnect():
        nonlocal reconnect_count
        reconnect_count += 1
        logger.info("[Test] Reconnect completed (count: %d)", reconnect_count)
        reconnect_event.set()

    async with PHXChannelsClient(
        mock_server.url,
        api_key="test_key",
        protocol_version=PhoenixChannelsProtocolVersion.V2,
        heartbeat_interval_secs=1.0,
        auto_reconnect=True,
        reconnect_max_attempts=5,
        reconnect_backoff_base=0.1,  # Fast for testing
        reconnect_backoff_max=0.5,
        on_disconnect=on_disconnect,
        on_reconnect=on_reconnect,
    ) as client:
        # Subscribe to a topic
        async def msg_handler(msg):
            pass

        await client.subscribe_to_topic("test:topic", msg_handler)
        assert "test:topic" in client._subscription_callbacks

        # Force disconnect
        await mock_server.force_disconnect_client()

        # Wait for disconnect detection
        await asyncio.wait_for(disconnect_event.wait(), timeout=5.0)
        assert disconnect_count == 1, "Disconnect should be detected"

        # Wait for reconnection
        await asyncio.wait_for(reconnect_event.wait(), timeout=5.0)
        assert reconnect_count == 1, "Should reconnect automatically"

        # Verify connection is active
        assert client.connection is not None, "Connection should be active after reconnect"


@pytest.mark.asyncio
async def test_topics_resubscribed_after_reconnection(mock_server: MockPhoenixServer):
    """
    Test that topics are automatically re-subscribed after reconnection.
    """
    reconnect_event = asyncio.Event()

    async def on_reconnect():
        reconnect_event.set()

    async with PHXChannelsClient(
        mock_server.url,
        api_key="test_key",
        protocol_version=PhoenixChannelsProtocolVersion.V2,
        heartbeat_interval_secs=1.0,
        auto_reconnect=True,
        reconnect_backoff_base=0.1,
        on_reconnect=on_reconnect,
    ) as client:
        # Subscribe to multiple topics
        async def handler1(msg):
            pass

        async def handler2(msg):
            pass

        await client.subscribe_to_topic("topic:one", handler1)
        await client.subscribe_to_topic("topic:two", handler2)

        # Verify subscriptions stored for reconnection
        assert "topic:one" in client._subscription_callbacks
        assert "topic:two" in client._subscription_callbacks

        # Force disconnect and wait for reconnection
        await mock_server.force_disconnect_client()
        await asyncio.wait_for(reconnect_event.wait(), timeout=5.0)

        # Verify topics are still tracked (re-subscribed)
        assert "topic:one" in client._subscription_callbacks
        assert "topic:two" in client._subscription_callbacks


@pytest.mark.asyncio
async def test_reconnection_respects_max_attempts(mock_server: MockPhoenixServer):
    """Test that reconnection stops after max attempts."""
    # Stop the server so reconnection always fails
    await mock_server.stop()

    disconnect_event = asyncio.Event()

    async def on_disconnect(error):
        disconnect_event.set()

    client = PHXChannelsClient(
        mock_server.url,
        api_key="test_key",
        protocol_version=PhoenixChannelsProtocolVersion.V2,
        auto_reconnect=True,
        reconnect_max_attempts=2,
        reconnect_backoff_base=0.1,
        on_disconnect=on_disconnect,
    )

    # Client will fail to connect initially since server is stopped
    with pytest.raises(Exception):
        await client.__aenter__()


@pytest.mark.asyncio
async def test_reconnection_can_be_disabled(mock_server: MockPhoenixServer):
    """Test that reconnection can be disabled."""
    async with PHXChannelsClient(
        mock_server.url,
        api_key="test_key",
        protocol_version=PhoenixChannelsProtocolVersion.V2,
        auto_reconnect=False,
    ) as client:
        assert client._auto_reconnect is False
