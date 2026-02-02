#!/usr/bin/env python3
"""
Reconnection Test - Verify INT-93 Fix on Localhost

This script tests the reconnection feature by:
1. Connecting to localhost Phoenix server
2. Subscribing to a topic
3. Waiting for you to restart the server (simulating disconnect)
4. Verifying automatic reconnection and re-subscription

Usage:
    # Start your local platform first, then run:
    uv run python test_reconnection_localhost.py --api-key YOUR_LOCAL_API_KEY

    # Or with custom URL:
    uv run python test_reconnection_localhost.py --ws-url ws://localhost:4000/api/v1/socket/websocket --api-key YOUR_KEY
"""

import argparse
import asyncio
import logging
from datetime import datetime

from phoenix_channels_python_client.client import PHXChannelsClient
from phoenix_channels_python_client.protocol_handler import (
    PhoenixChannelsProtocolVersion,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Show reconnection logs
logging.getLogger("phoenix_channels_python_client").setLevel(logging.DEBUG)
# Reduce websocket noise
logging.getLogger("websockets").setLevel(logging.WARNING)


class ReconnectionTracker:
    """Track disconnect and reconnect events."""

    def __init__(self):
        self.disconnect_count = 0
        self.reconnect_count = 0
        self.disconnect_time = None
        self.reconnect_time = None
        self.messages_received = 0

    async def on_disconnect(self, error):
        self.disconnect_count += 1
        self.disconnect_time = datetime.now()
        logger.warning("=" * 60)
        logger.warning("🔴 DISCONNECTED! (count: %d)", self.disconnect_count)
        logger.warning("   Error: %s", error)
        logger.warning("   Time: %s", self.disconnect_time.strftime("%H:%M:%S"))
        logger.warning("=" * 60)
        logger.info("")
        logger.info("👉 The client will now attempt to reconnect automatically...")
        logger.info("   (Make sure your local server is back up)")
        logger.info("")

    async def on_reconnect(self):
        self.reconnect_count += 1
        self.reconnect_time = datetime.now()

        if self.disconnect_time:
            recovery_time = (self.reconnect_time - self.disconnect_time).total_seconds()
        else:
            recovery_time = 0

        logger.info("=" * 60)
        logger.info("🟢 RECONNECTED! (count: %d)", self.reconnect_count)
        logger.info("   Time: %s", self.reconnect_time.strftime("%H:%M:%S"))
        logger.info("   Recovery time: %.1f seconds", recovery_time)
        logger.info("=" * 60)


async def test_reconnection(args):
    """Test reconnection against localhost."""

    tracker = ReconnectionTracker()

    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════════════╗")
    logger.info("║           RECONNECTION TEST (INT-93)                             ║")
    logger.info("╚══════════════════════════════════════════════════════════════════╝")
    logger.info("")
    logger.info("WebSocket URL: %s", args.ws_url)
    logger.info("Reconnection: ENABLED (backoff: 1s, 2s, 4s... up to 30s)")
    logger.info("")
    logger.info("=" * 70)
    logger.info("INSTRUCTIONS:")
    logger.info("  1. This script will connect and subscribe to a test topic")
    logger.info("  2. RESTART your local Phoenix server to simulate disconnect")
    logger.info("  3. Watch the automatic reconnection happen")
    logger.info("  4. Press Ctrl+C when done testing")
    logger.info("=" * 70)
    logger.info("")

    try:
        client = PHXChannelsClient(
            args.ws_url,
            args.api_key,
            protocol_version=PhoenixChannelsProtocolVersion.V2,
            # Heartbeat settings
            heartbeat_interval_secs=30,
            # Reconnection settings
            auto_reconnect=True,
            reconnect_max_attempts=0,  # Unlimited attempts
            reconnect_backoff_base=1.0,
            reconnect_backoff_max=30.0,
            on_disconnect=tracker.on_disconnect,
            on_reconnect=tracker.on_reconnect,
        )

        async with client:
            logger.info("✅ Connected to Phoenix server!")
            logger.info("")

            # Try to subscribe to a test topic (may fail if topic doesn't exist, that's OK)
            try:

                async def message_handler(message):
                    tracker.messages_received += 1
                    logger.info("📨 Received message: %s", message.event)

                # Use a generic topic that might exist
                test_topic = f"test:reconnection_{datetime.now().strftime('%H%M%S')}"
                logger.info("Attempting to subscribe to topic: %s", test_topic)
                logger.info(
                    "(Subscription may fail if topic doesn't exist - that's OK)"
                )

                try:
                    await asyncio.wait_for(
                        client.subscribe_to_topic(test_topic, message_handler),
                        timeout=5.0,
                    )
                    logger.info("✅ Subscribed to topic: %s", test_topic)
                except Exception as e:
                    logger.warning("⚠️  Could not subscribe to test topic: %s", e)
                    logger.info("   (This is OK - we're testing connection stability)")

            except Exception as e:
                logger.warning("Subscription error (continuing anyway): %s", e)

            logger.info("")
            logger.info("=" * 70)
            logger.info("🔄 CONNECTION ACTIVE - Waiting for events...")
            logger.info("")
            logger.info(
                "   👉 NOW: Restart your local Phoenix server to test reconnection"
            )
            logger.info(
                "   👉 Or wait and watch heartbeats being sent every 30 seconds"
            )
            logger.info("   👉 Press Ctrl+C to stop")
            logger.info("=" * 70)
            logger.info("")

            # Keep running and report status every 30 seconds
            start_time = datetime.now()
            while True:
                await asyncio.sleep(30)
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(
                    "📊 Status: Connected for %.0fs | Disconnects: %d | Reconnects: %d",
                    elapsed,
                    tracker.disconnect_count,
                    tracker.reconnect_count,
                )

    except KeyboardInterrupt:
        logger.info("")
        logger.info("=" * 70)
        logger.info("TEST COMPLETE")
        logger.info("=" * 70)
        logger.info("")
        logger.info("Summary:")
        logger.info("  - Disconnections detected: %d", tracker.disconnect_count)
        logger.info("  - Successful reconnections: %d", tracker.reconnect_count)
        logger.info("  - Messages received: %d", tracker.messages_received)
        logger.info("")

        if tracker.reconnect_count > 0:
            logger.info("✅ RECONNECTION VERIFIED! The fix works.")
        elif tracker.disconnect_count == 0:
            logger.info("ℹ️  No disconnections occurred during test.")
            logger.info("   Try restarting your server to test reconnection.")
        else:
            logger.warning("⚠️  Disconnections occurred but no reconnections.")
            logger.warning("   Check if your server came back up.")

    except Exception as e:
        logger.error("Test failed: %s", e)
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test reconnection feature against localhost",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--ws-url",
        default="ws://localhost:4000/api/v1/socket/websocket",
        help="WebSocket URL (default: ws://localhost:4000/api/v1/socket/websocket)",
    )
    parser.add_argument("--api-key", required=True, help="API key for authentication")

    args = parser.parse_args()

    asyncio.run(test_reconnection(args))


if __name__ == "__main__":
    main()
