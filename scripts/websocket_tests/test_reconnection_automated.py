#!/usr/bin/env python3
"""
Automated Reconnection Test - Simulates server disconnect and verifies reconnection.

This test creates a mock Phoenix server, connects a client, then forcefully
closes the connection to trigger automatic reconnection.

Usage:
    uv run python test_reconnection_automated.py
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from websockets.server import serve, WebSocketServerProtocol

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("websockets").setLevel(logging.WARNING)


class MockPhoenixServer:
    """A mock Phoenix server that can be controlled for testing."""

    def __init__(self, host: str = "localhost", port: int = 9999):
        self.host = host
        self.port = port
        self.server = None
        self.client_ws: Optional[WebSocketServerProtocol] = None
        self._running = True

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}/socket/websocket"

    async def handle_client(self, websocket: WebSocketServerProtocol) -> None:
        """Handle incoming WebSocket connections."""
        self.client_ws = websocket
        logger.info("[SERVER] Client connected")

        try:
            async for message in websocket:
                data = json.loads(message)
                await self._handle_message(websocket, data)
        except Exception as e:
            logger.info("[SERVER] Client disconnected: %s", e)
        finally:
            self.client_ws = None

    async def _handle_message(self, websocket: WebSocketServerProtocol, data: list) -> None:
        """Handle Phoenix protocol messages."""
        if not isinstance(data, list) or len(data) != 5:
            return

        join_ref, msg_ref, topic, event, payload = data

        # Handle join
        if event == "phx_join":
            reply = [join_ref, msg_ref, topic, "phx_reply", {"status": "ok", "response": {}}]
            await websocket.send(json.dumps(reply))
            logger.info("[SERVER] Joined topic: %s", topic)

        # Handle heartbeat
        elif event == "heartbeat" and topic == "phoenix":
            reply = [join_ref, msg_ref, "phoenix", "phx_reply", {"status": "ok", "response": {}}]
            await websocket.send(json.dumps(reply))
            logger.debug("[SERVER] Heartbeat acknowledged")

        # Handle leave
        elif event == "phx_leave":
            reply = [join_ref, msg_ref, topic, "phx_reply", {"status": "ok", "response": {}}]
            await websocket.send(json.dumps(reply))
            logger.info("[SERVER] Left topic: %s", topic)

    async def start(self) -> None:
        """Start the mock server."""
        self.server = await serve(self.handle_client, self.host, self.port)
        logger.info("[SERVER] Started on %s", self.url)

    async def stop(self) -> None:
        """Stop the mock server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("[SERVER] Stopped")

    async def force_disconnect_client(self) -> None:
        """Forcefully close the client connection to simulate server crash."""
        if self.client_ws:
            logger.info("[SERVER] Forcing client disconnect...")
            await self.client_ws.close(1001, "Server going down")
            self.client_ws = None


async def test_reconnection() -> bool:
    """Test automatic reconnection after server disconnect."""
    from phoenix_channels_python_client.client import PHXChannelsClient
    from phoenix_channels_python_client.protocol_handler import PhoenixChannelsProtocolVersion

    logger.info("")
    logger.info("=" * 70)
    logger.info("AUTOMATED RECONNECTION TEST")
    logger.info("=" * 70)
    logger.info("")

    # Track events
    disconnect_count = 0
    reconnect_count = 0
    disconnect_event = asyncio.Event()
    reconnect_event = asyncio.Event()

    async def on_disconnect(error):
        nonlocal disconnect_count
        disconnect_count += 1
        logger.info("[CLIENT] DISCONNECTED! (count: %d, error: %s)", disconnect_count, error)
        disconnect_event.set()

    async def on_reconnect():
        nonlocal reconnect_count
        reconnect_count += 1
        logger.info("[CLIENT] RECONNECTED! (count: %d)", reconnect_count)
        reconnect_event.set()

    # Start mock server
    server = MockPhoenixServer()
    await server.start()

    try:
        # Create client with fast reconnection for testing
        client = PHXChannelsClient(
            server.url,
            api_key="test_key",
            protocol_version=PhoenixChannelsProtocolVersion.V2,
            heartbeat_interval_secs=5,  # Fast heartbeat for testing
            auto_reconnect=True,
            reconnect_max_attempts=5,
            reconnect_backoff_base=0.5,  # Fast backoff for testing
            reconnect_backoff_max=2.0,
            on_disconnect=on_disconnect,
            on_reconnect=on_reconnect,
        )

        async with client:
            logger.info("[CLIENT] Connected to mock server")

            # Subscribe to a test topic
            async def msg_handler(msg):
                pass

            await client.subscribe_to_topic("test:topic", msg_handler)
            logger.info("[CLIENT] Subscribed to test:topic")

            # Wait a moment for stable connection
            await asyncio.sleep(1)

            # STEP 1: Force disconnect from server side
            logger.info("")
            logger.info("-" * 70)
            logger.info("STEP 1: Forcing server disconnect...")
            logger.info("-" * 70)
            await server.force_disconnect_client()

            # Wait for disconnect to be detected
            try:
                await asyncio.wait_for(disconnect_event.wait(), timeout=5.0)
                logger.info("[OK] Disconnect detected!")
            except asyncio.TimeoutError:
                logger.error("[FAIL] Disconnect not detected within 5 seconds")
                return False

            # STEP 2: Wait for automatic reconnection
            logger.info("")
            logger.info("-" * 70)
            logger.info("STEP 2: Waiting for automatic reconnection...")
            logger.info("-" * 70)

            try:
                await asyncio.wait_for(reconnect_event.wait(), timeout=10.0)
                logger.info("[OK] Reconnected successfully!")
            except asyncio.TimeoutError:
                logger.error("[FAIL] Reconnection not completed within 10 seconds")
                return False

            # STEP 3: Verify connection is working
            logger.info("")
            logger.info("-" * 70)
            logger.info("STEP 3: Verifying connection is working...")
            logger.info("-" * 70)

            # Wait for a heartbeat to confirm connection
            await asyncio.sleep(6)  # Wait for heartbeat cycle

            if client.connection is not None:
                logger.info("[OK] Connection is active!")
            else:
                logger.error("[FAIL] Connection is not active")
                return False

    except Exception as e:
        logger.error("Test error: %s", e)
        return False
    finally:
        await server.stop()

    # Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    logger.info("Disconnects detected: %d", disconnect_count)
    logger.info("Reconnects completed: %d", reconnect_count)
    logger.info("")

    if disconnect_count >= 1 and reconnect_count >= 1:
        logger.info("[PASS] RECONNECTION TEST PASSED!")
        logger.info("")
        logger.info("The automatic reconnection feature is working correctly:")
        logger.info("  1. Client detected server disconnect")
        logger.info("  2. Client automatically reconnected with backoff")
        logger.info("  3. Connection is active after reconnection")
        return True
    else:
        logger.error("[FAIL] RECONNECTION TEST FAILED!")
        return False


def main() -> None:
    logging.getLogger("phoenix_channels_python_client").setLevel(logging.INFO)

    success = asyncio.run(test_reconnection())
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
