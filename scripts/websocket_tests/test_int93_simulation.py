#!/usr/bin/env python3
"""
INT-93 Simulation Test - Verify WebSocket Connection Stability Fix

This script simulates the scenarios from ticket INT-93:
"Agent WebSocket connections drop unpredictably despite active messaging"

The test verifies:
1. WITHOUT heartbeat: Connection drops after ~60 seconds (proves the problem)
2. WITH heartbeat: Connection stays alive (proves heartbeat fix works)
3. WITH reconnection: Connection recovers after server disconnect (proves reconnection works)

Usage:
    # Run against localhost Phoenix server
    uv run python test_int93_simulation.py --ws-url ws://localhost:4000/api/v1/socket/websocket --api-key YOUR_KEY

    # Run against production (not recommended for testing disconnects)
    uv run python test_int93_simulation.py --ws-url wss://app.thenvoi.com/api/v1/socket/websocket --api-key YOUR_KEY

    # Only run heartbeat test (skip the 70-second no-heartbeat test)
    uv run python test_int93_simulation.py --skip-no-heartbeat --ws-url ws://localhost:4000/api/v1/socket/websocket --api-key YOUR_KEY

    # Test reconnection (requires manual server restart)
    uv run python test_int93_simulation.py --test-reconnection --ws-url ws://localhost:4000/api/v1/socket/websocket --api-key YOUR_KEY
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Reduce websocket noise
logging.getLogger("websockets").setLevel(logging.WARNING)


def print_banner(title: str) -> None:
    """Print a banner with title."""
    logger.info("")
    logger.info("=" * 70)
    logger.info(title.center(70))
    logger.info("=" * 70)


def print_result(success: bool, message: str) -> None:
    """Print a result with status icon."""
    icon = "[PASS]" if success else "[FAIL]"
    logger.info("%s %s", icon, message)


async def test_connection_without_heartbeat(ws_url: str, api_key: str) -> dict:
    """
    Test connection WITHOUT heartbeat - should drop after ~60 seconds.

    This proves the problem described in INT-93.
    """
    # Import here to avoid issues if phoenix client not installed
    from phoenix_channels_python_client.client import PHXChannelsClient
    from phoenix_channels_python_client.protocol_handler import (
        PhoenixChannelsProtocolVersion,
    )

    print_banner("TEST 1: Connection WITHOUT Heartbeat")
    logger.info("Expected: Connection should DROP after ~60 seconds")
    logger.info("This proves the problem described in INT-93")
    logger.info("-" * 70)

    result = {
        "test_name": "Without Heartbeat",
        "expected_behavior": "Connection drops after ~60s",
        "actual_duration": 0.0,
        "connection_dropped": False,
        "success": False,
        "error": None,
    }

    start_time = datetime.now()
    test_duration = 70  # Wait past the 60s server timeout

    try:
        client = PHXChannelsClient(
            ws_url,
            api_key,
            protocol_version=PhoenixChannelsProtocolVersion.V2,
            heartbeat_interval_secs=None,  # DISABLED - no heartbeat
            auto_reconnect=False,  # Don't reconnect for this test
        )

        async with client:
            logger.info("[OK] Connected at %s", start_time.strftime("%H:%M:%S"))
            logger.info(
                "[WARN] Heartbeat DISABLED - server will timeout connection after ~60s"
            )

            check_interval = 5
            for elapsed in range(0, test_duration + 1, check_interval):
                if elapsed > 0:
                    await asyncio.sleep(check_interval)

                actual_elapsed = (datetime.now() - start_time).total_seconds()

                # Check if connection is still alive
                if client.connection is None:
                    logger.info(
                        "[X] Connection DROPPED at %.1f seconds!", actual_elapsed
                    )
                    result["connection_dropped"] = True
                    result["actual_duration"] = actual_elapsed
                    result["success"] = True  # Expected behavior!
                    return result

                status = "[OK]" if actual_elapsed <= 55 else "[?]"
                logger.info(
                    "%s Connection alive at %.0f seconds", status, actual_elapsed
                )

            # If we get here, connection stayed alive (unexpected)
            result["actual_duration"] = (datetime.now() - start_time).total_seconds()
            result["success"] = False  # Connection should have dropped!
            result["error"] = "Connection unexpectedly stayed alive without heartbeat"

    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info("[X] Connection error at %.1f seconds: %s", elapsed, e)
        result["connection_dropped"] = True
        result["actual_duration"] = elapsed
        result["success"] = True  # Expected - connection should fail

    return result


async def test_connection_with_heartbeat(ws_url: str, api_key: str) -> dict:
    """
    Test connection WITH heartbeat - should stay alive beyond 60 seconds.

    This proves the heartbeat fix for INT-93 works.
    """
    from phoenix_channels_python_client.client import PHXChannelsClient
    from phoenix_channels_python_client.protocol_handler import (
        PhoenixChannelsProtocolVersion,
    )

    print_banner("TEST 2: Connection WITH Heartbeat")
    logger.info("Expected: Connection should STAY ALIVE beyond 60 seconds")
    logger.info("This proves the heartbeat fix works")
    logger.info("-" * 70)

    result = {
        "test_name": "With Heartbeat",
        "expected_behavior": "Connection stays alive beyond 60s",
        "actual_duration": 0.0,
        "connection_dropped": False,
        "success": False,
        "error": None,
    }

    start_time = datetime.now()
    test_duration = 90  # Test well past the 60s timeout

    try:
        client = PHXChannelsClient(
            ws_url,
            api_key,
            protocol_version=PhoenixChannelsProtocolVersion.V2,
            heartbeat_interval_secs=30,  # ENABLED - 30 second heartbeat
            auto_reconnect=False,  # Don't reconnect for this test
        )

        async with client:
            logger.info("[OK] Connected at %s", start_time.strftime("%H:%M:%S"))
            logger.info("[OK] Heartbeat ENABLED (30 second interval)")

            check_interval = 10
            for elapsed in range(0, test_duration + 1, check_interval):
                if elapsed > 0:
                    await asyncio.sleep(check_interval)

                actual_elapsed = (datetime.now() - start_time).total_seconds()

                # Check if connection is still alive
                if client.connection is None:
                    logger.error(
                        "[X] Connection DROPPED at %.1f seconds!", actual_elapsed
                    )
                    result["connection_dropped"] = True
                    result["actual_duration"] = actual_elapsed
                    result["success"] = False  # Should NOT drop with heartbeat!
                    result["error"] = (
                        f"Connection dropped at {actual_elapsed:.1f}s despite heartbeat"
                    )
                    return result

                # Check heartbeat task health
                if client._heartbeat_task and client._heartbeat_task.done():
                    exc = client._heartbeat_task.exception()
                    logger.error("[X] Heartbeat task died: %s", exc)
                    result["error"] = f"Heartbeat task died: {exc}"
                    result["actual_duration"] = actual_elapsed
                    return result

                # Show progress with emphasis on passing 60s mark
                if actual_elapsed > 60:
                    logger.info(
                        "[OK] Connection still alive at %.0f seconds (PAST 60s timeout!)",
                        actual_elapsed,
                    )
                else:
                    logger.info("[..] Connection alive at %.0f seconds", actual_elapsed)

            # Success - connection stayed alive!
            result["actual_duration"] = (datetime.now() - start_time).total_seconds()
            result["success"] = True
            logger.info(
                "[OK] SUCCESS: Connection stayed alive for %.0f seconds!",
                result["actual_duration"],
            )

    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error("[X] Error at %.1f seconds: %s", elapsed, e)
        result["actual_duration"] = elapsed
        result["error"] = str(e)
        result["connection_dropped"] = True

    return result


async def test_reconnection(ws_url: str, api_key: str) -> dict:
    """
    Test automatic reconnection - interactive test requiring server restart.
    """
    from phoenix_channels_python_client.client import PHXChannelsClient
    from phoenix_channels_python_client.protocol_handler import (
        PhoenixChannelsProtocolVersion,
    )

    print_banner("TEST 3: Automatic Reconnection")
    logger.info("This test requires you to restart the server to trigger reconnection")
    logger.info("-" * 70)

    result = {
        "test_name": "Reconnection",
        "disconnect_count": 0,
        "reconnect_count": 0,
        "success": False,
    }

    disconnect_time: Optional[datetime] = None

    async def on_disconnect(error):
        nonlocal disconnect_time
        result["disconnect_count"] += 1
        disconnect_time = datetime.now()
        logger.warning("")
        logger.warning("=" * 60)
        logger.warning("[!] DISCONNECTED (count: %d)", result["disconnect_count"])
        logger.warning("    Error: %s", error)
        logger.warning("    Time: %s", disconnect_time.strftime("%H:%M:%S"))
        logger.warning("=" * 60)
        logger.info("")
        logger.info("    The client will now attempt to reconnect...")
        logger.info("")

    async def on_reconnect():
        result["reconnect_count"] += 1
        reconnect_time = datetime.now()

        if disconnect_time:
            recovery_time = (reconnect_time - disconnect_time).total_seconds()
        else:
            recovery_time = 0

        logger.info("")
        logger.info("=" * 60)
        logger.info("[OK] RECONNECTED! (count: %d)", result["reconnect_count"])
        logger.info("     Recovery time: %.1f seconds", recovery_time)
        logger.info("=" * 60)
        logger.info("")

    try:
        client = PHXChannelsClient(
            ws_url,
            api_key,
            protocol_version=PhoenixChannelsProtocolVersion.V2,
            heartbeat_interval_secs=30,
            auto_reconnect=True,
            reconnect_max_attempts=0,  # Unlimited
            reconnect_backoff_base=1.0,
            reconnect_backoff_max=30.0,
            on_disconnect=on_disconnect,
            on_reconnect=on_reconnect,
        )

        async with client:
            logger.info("[OK] Connected!")
            logger.info("[OK] Heartbeat: ENABLED (30s)")
            logger.info("[OK] Auto-reconnect: ENABLED (unlimited attempts)")
            logger.info("")
            logger.info("=" * 70)
            logger.info("INSTRUCTIONS:")
            logger.info("  1. Restart your local Phoenix server")
            logger.info("  2. Watch for disconnect and automatic reconnection")
            logger.info("  3. Press Ctrl+C when done testing")
            logger.info("=" * 70)
            logger.info("")

            start_time = datetime.now()
            while True:
                await asyncio.sleep(30)
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(
                    "[STATUS] Running for %.0fs | Disconnects: %d | Reconnects: %d",
                    elapsed,
                    result["disconnect_count"],
                    result["reconnect_count"],
                )

    except KeyboardInterrupt:
        logger.info("")
        print_banner("RECONNECTION TEST COMPLETE")
        logger.info("Disconnections: %d", result["disconnect_count"])
        logger.info("Reconnections: %d", result["reconnect_count"])

        if result["reconnect_count"] > 0:
            result["success"] = True
            logger.info("")
            logger.info("[OK] RECONNECTION VERIFIED!")
        elif result["disconnect_count"] == 0:
            logger.info("")
            logger.info(
                "[INFO] No disconnections occurred. Restart your server to test."
            )
        else:
            logger.warning("")
            logger.warning(
                "[WARN] Disconnections but no reconnections. Check server availability."
            )

    except Exception as e:
        logger.error("Test error: %s", e)
        result["error"] = str(e)

    return result


async def run_simulation(args) -> bool:
    """Run the simulation tests."""
    results = []

    print_banner("INT-93 SIMULATION TEST")
    logger.info("")
    logger.info("Ticket: Agent WebSocket connections drop unpredictably")
    logger.info("URL: %s", args.ws_url)
    logger.info("")

    # Test 1: Without heartbeat (should fail after ~60s)
    if not args.skip_no_heartbeat and not args.test_reconnection:
        result1 = await test_connection_without_heartbeat(args.ws_url, args.api_key)
        results.append(result1)

        if result1["success"]:
            logger.info("")
            logger.info(
                "[OK] CONFIRMED: Without heartbeat, connection drops after ~60s"
            )
            logger.info("     This is the bug that INT-93 reports!")
        else:
            logger.warning("")
            logger.warning("[?] UNEXPECTED: Connection stayed alive without heartbeat")
            logger.warning(
                "    Server may have longer timeout or doesn't require heartbeat"
            )

    # Test 2: With heartbeat (should stay alive)
    if not args.test_reconnection:
        result2 = await test_connection_with_heartbeat(args.ws_url, args.api_key)
        results.append(result2)

        if result2["success"]:
            logger.info("")
            logger.info("[OK] CONFIRMED: With heartbeat, connection stays alive!")
            logger.info("     The heartbeat fix works!")
        else:
            logger.error("")
            logger.error("[X] FAILED: Connection dropped even with heartbeat")

    # Test 3: Reconnection (interactive)
    if args.test_reconnection:
        result3 = await test_reconnection(args.ws_url, args.api_key)
        results.append(result3)

    # Print summary
    if results and not args.test_reconnection:
        print_banner("SUMMARY")

        for r in results:
            status = "[PASS]" if r["success"] else "[FAIL]"
            logger.info(
                "%s %s | Duration: %.0fs | %s",
                status,
                r["test_name"],
                r["actual_duration"],
                r["expected_behavior"],
            )

        logger.info("")

        # Determine overall success
        if not args.skip_no_heartbeat:
            no_hb = results[0]
            with_hb = results[1]

            if no_hb["success"] and with_hb["success"]:
                logger.info("[OK] INT-93 FIX VERIFIED!")
                logger.info("    - Without heartbeat: Connection drops (bug confirmed)")
                logger.info("    - With heartbeat: Connection stays alive (fix works)")
                return True
            elif with_hb["success"]:
                logger.info("[OK] Heartbeat keeps connection alive!")
                logger.info("    (No-heartbeat test was inconclusive or skipped)")
                return True
            else:
                logger.error("[X] FIX NOT WORKING - Connection dropped with heartbeat")
                return False
        else:
            with_hb = results[0]
            if with_hb["success"]:
                logger.info("[OK] Heartbeat keeps connection alive!")
                return True
            else:
                logger.error("[X] Connection dropped with heartbeat!")
                return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="INT-93 Simulation Test - Verify WebSocket connection stability fix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full test against localhost
  uv run python test_int93_simulation.py --ws-url ws://localhost:4000/api/v1/socket/websocket --api-key YOUR_KEY

  # Quick test (skip 70-second no-heartbeat test)
  uv run python test_int93_simulation.py --skip-no-heartbeat --ws-url ws://localhost:4000/api/v1/socket/websocket --api-key YOUR_KEY

  # Test automatic reconnection (interactive)
  uv run python test_int93_simulation.py --test-reconnection --ws-url ws://localhost:4000/api/v1/socket/websocket --api-key YOUR_KEY
        """,
    )

    parser.add_argument(
        "--ws-url",
        required=True,
        help="WebSocket URL (e.g., ws://localhost:4000/api/v1/socket/websocket)",
    )
    parser.add_argument("--api-key", required=True, help="API key for authentication")
    parser.add_argument(
        "--skip-no-heartbeat",
        action="store_true",
        help="Skip the 70-second test without heartbeat",
    )
    parser.add_argument(
        "--test-reconnection",
        action="store_true",
        help="Run interactive reconnection test (requires manual server restart)",
    )

    args = parser.parse_args()

    # Enable debug logging for phoenix client to see heartbeat messages
    logging.getLogger("phoenix_channels_python_client").setLevel(logging.DEBUG)

    success = asyncio.run(run_simulation(args))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
