#!/usr/bin/env python3
"""
Heartbeat Simulation Test - Verify INT-93 Fix

This script simulates both scenarios:
1. WITHOUT heartbeat - connection SHOULD drop after ~60 seconds
2. WITH heartbeat - connection SHOULD stay alive

Usage:
    # Test against localhost (default)
    uv run python test_heartbeat_simulation.py

    # Test against custom URL
    uv run python test_heartbeat_simulation.py --ws-url ws://localhost:4000/socket/websocket --api-key your_key

    # Only test with heartbeat (skip the 70-second failure test)
    uv run python test_heartbeat_simulation.py --skip-no-heartbeat
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from typing import Optional

from phoenix_channels_python_client.client import PHXChannelsClient
from phoenix_channels_python_client.protocol_handler import (
    PhoenixChannelsProtocolVersion,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Reduce websocket noise
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("phoenix_channels_python_client").setLevel(logging.DEBUG)


async def test_connection(
    ws_url: str,
    api_key: str,
    heartbeat_interval: Optional[float],
    test_duration: int,
    test_name: str,
) -> dict:
    """
    Test WebSocket connection with or without heartbeat.

    Args:
        ws_url: WebSocket URL
        api_key: API key for authentication
        heartbeat_interval: Heartbeat interval in seconds, or None to disable
        test_duration: How long to keep connection open (seconds)
        test_name: Name for logging

    Returns:
        dict with test results
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST: %s", test_name)
    logger.info("=" * 70)
    logger.info("WebSocket URL: %s", ws_url)
    logger.info(
        "Heartbeat: %s", f"{heartbeat_interval}s" if heartbeat_interval else "DISABLED"
    )
    logger.info("Test duration: %d seconds", test_duration)
    logger.info("-" * 70)

    start_time = datetime.now()
    result = {
        "test_name": test_name,
        "heartbeat_enabled": heartbeat_interval is not None,
        "duration_target": test_duration,
        "duration_actual": 0.0,
        "success": False,
        "error": None,
        "connection_dropped": False,
    }

    try:
        client = PHXChannelsClient(
            ws_url,
            api_key,
            protocol_version=PhoenixChannelsProtocolVersion.V2,
            heartbeat_interval_secs=heartbeat_interval,
        )

        async with client:
            logger.info("✅ Connected at %s", start_time.strftime("%H:%M:%S"))

            if heartbeat_interval:
                logger.info(
                    "✅ Heartbeat task running (interval: %ss)", heartbeat_interval
                )
            else:
                logger.info("⚠️  Heartbeat DISABLED - connection may drop after ~60s")

            # Monitor connection for test duration
            check_interval = 5  # Check every 5 seconds
            for elapsed in range(0, test_duration + 1, check_interval):
                if elapsed > 0:
                    await asyncio.sleep(check_interval)

                elapsed_actual = (datetime.now() - start_time).total_seconds()

                # Check if connection is still alive
                if client.connection is None:
                    logger.error(
                        "❌ Connection DROPPED at %.1f seconds!", elapsed_actual
                    )
                    result["connection_dropped"] = True
                    result["duration_actual"] = elapsed_actual
                    result["error"] = f"Connection dropped at {elapsed_actual:.1f}s"
                    return result

                # Check if heartbeat task died (if enabled)
                if (
                    heartbeat_interval
                    and client._heartbeat_task
                    and client._heartbeat_task.done()
                ):
                    exc = client._heartbeat_task.exception()
                    logger.error(
                        "❌ Heartbeat task died at %.1f seconds! Error: %s",
                        elapsed_actual,
                        exc,
                    )
                    result["error"] = f"Heartbeat task died: {exc}"
                    result["duration_actual"] = elapsed_actual
                    return result

                status = "🟢" if elapsed_actual > 60 else "🔵"
                logger.info(
                    "%s Connection alive at %.0f seconds", status, elapsed_actual
                )

            # Test completed successfully
            result["duration_actual"] = (datetime.now() - start_time).total_seconds()
            result["success"] = True
            logger.info("")
            logger.info(
                "✅ SUCCESS: Connection stayed alive for %.0f seconds!",
                result["duration_actual"],
            )

    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error("❌ ERROR at %.1f seconds: %s", elapsed, e)
        result["duration_actual"] = elapsed
        result["error"] = str(e)
        result["connection_dropped"] = True

    return result


async def run_simulation(args):
    """Run the full simulation comparing with/without heartbeat."""

    results = []

    # Test 1: WITHOUT heartbeat (should fail after ~60s)
    if not args.skip_no_heartbeat:
        logger.info("")
        logger.info("🔴 SCENARIO 1: Testing WITHOUT heartbeat")
        logger.info("   Expected: Connection should DROP after ~60 seconds")
        logger.info("   (This test takes ~70 seconds to verify the drop)")
        logger.info("")

        result1 = await test_connection(
            ws_url=args.ws_url,
            api_key=args.api_key,
            heartbeat_interval=None,  # DISABLED
            test_duration=70,  # Wait past the 60s timeout
            test_name="WITHOUT Heartbeat (should fail)",
        )
        results.append(result1)

        # For this test, "success" means the connection DID drop (proving the bug)
        if result1["connection_dropped"]:
            logger.info("")
            logger.info("✅ CONFIRMED: Without heartbeat, connection drops after ~60s")
            logger.info("   This is the bug that INT-93 fixes!")
        else:
            logger.warning("")
            logger.warning("⚠️  UNEXPECTED: Connection stayed alive without heartbeat")
            logger.warning(
                "   Either the server timeout is longer, or heartbeat isn't required"
            )

    # Test 2: WITH heartbeat (should stay alive)
    logger.info("")
    logger.info("🟢 SCENARIO 2: Testing WITH heartbeat")
    logger.info("   Expected: Connection should STAY ALIVE beyond 60 seconds")
    logger.info("")

    result2 = await test_connection(
        ws_url=args.ws_url,
        api_key=args.api_key,
        heartbeat_interval=30,  # ENABLED (30 seconds, matching Phoenix default)
        test_duration=90,  # Test past the 60s timeout
        test_name="WITH Heartbeat (should succeed)",
    )
    results.append(result2)

    # Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SIMULATION SUMMARY")
    logger.info("=" * 70)

    for r in results:
        status = "✅ PASS" if r["success"] == r["heartbeat_enabled"] else "❌ FAIL"
        hb_status = "enabled" if r["heartbeat_enabled"] else "disabled"
        logger.info(
            "%s | %s | Heartbeat %s | Duration: %.0fs",
            status,
            r["test_name"],
            hb_status,
            r["duration_actual"],
        )

    logger.info("=" * 70)

    # Determine overall result
    if not args.skip_no_heartbeat:
        # Both tests should have expected behavior
        no_hb_correct = results[0]["connection_dropped"]  # Should drop
        with_hb_correct = results[1]["success"]  # Should stay alive

        if no_hb_correct and with_hb_correct:
            logger.info("")
            logger.info("🎉 INT-93 FIX VERIFIED!")
            logger.info("   - Without heartbeat: Connection drops (bug confirmed)")
            logger.info("   - With heartbeat: Connection stays alive (fix works)")
            return True
        else:
            logger.error("")
            logger.error("❌ UNEXPECTED RESULTS - Review the logs above")
            return False
    else:
        # Only tested with heartbeat
        if result2["success"]:
            logger.info("")
            logger.info("✅ Heartbeat keeps connection alive!")
            return True
        else:
            logger.error("")
            logger.error("❌ Connection failed even with heartbeat!")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Test heartbeat fix for INT-93",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test against localhost Phoenix server
  uv run python test_heartbeat_simulation.py --ws-url ws://localhost:4000/socket/websocket --api-key test_key

  # Test against local Thenvoi platform
  uv run python test_heartbeat_simulation.py --ws-url ws://localhost:4000/api/v1/socket/websocket --api-key your_agent_key

  # Skip the 70-second no-heartbeat test (faster)
  uv run python test_heartbeat_simulation.py --skip-no-heartbeat --ws-url ws://localhost:4000/socket/websocket --api-key test_key
        """,
    )

    parser.add_argument(
        "--ws-url",
        default="ws://localhost:4000/api/v1/socket/websocket",
        help="WebSocket URL (default: ws://localhost:4000/api/v1/socket/websocket)",
    )
    parser.add_argument(
        "--api-key",
        default="test_key",
        help="API key for authentication (default: test_key)",
    )
    parser.add_argument(
        "--skip-no-heartbeat",
        action="store_true",
        help="Skip the 70-second test without heartbeat",
    )

    args = parser.parse_args()

    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════════════╗")
    logger.info("║           HEARTBEAT SIMULATION TEST (INT-93)                     ║")
    logger.info("╚══════════════════════════════════════════════════════════════════╝")
    logger.info("")
    logger.info("This test verifies that the heartbeat fix prevents connection drops.")
    logger.info("")

    success = asyncio.run(run_simulation(args))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
