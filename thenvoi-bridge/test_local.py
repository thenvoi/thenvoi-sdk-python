"""Local integration test for LangChainHandler against the echo server.

Prerequisites:
    1. Start the echo server in another terminal:
       cd thenvoi-bridge && python echo_server.py

    2. Run this script:
       cd thenvoi-bridge && python test_local.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from unittest.mock import AsyncMock, MagicMock

# Add parent dir so we can import handlers and thenvoi
sys.path.insert(0, ".")
sys.path.insert(0, "..")

from handlers.chain import LangChainHandler

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger(__name__)

ECHO_SERVER_URL = "http://localhost:8000"


def make_fake_tools() -> MagicMock:
    """Create a mock AgentTools that logs calls instead of hitting the platform."""
    tools = MagicMock()
    tools.participants = [
        {"id": "user-1", "name": "Test User", "type": "User", "handle": "testuser"},
    ]

    async def log_send_message(**kwargs):
        logger.info("[send_message] %s", kwargs)

    async def log_send_event(**kwargs):
        logger.info("[send_event] %s", kwargs)

    tools.send_message = AsyncMock(side_effect=log_send_message)
    tools.send_event = AsyncMock(side_effect=log_send_event)
    return tools


async def test_base_url_mode() -> None:
    """Test with base_url (auto-appends /invoke)."""
    logger.info("=== Test: base_url mode ===")
    handler = LangChainHandler(base_url=ECHO_SERVER_URL)
    tools = make_fake_tools()

    try:
        await handler.handle(
            content="Hello from base_url test!",
            room_id="room-123",
            thread_id="thread-456",
            message_id="msg-789",
            sender_id="user-1",
            sender_name="Test User",
            sender_handle="testuser",
            sender_type="User",
            mentioned_agent="echo_agent",
            tools=tools,
        )
        logger.info(
            "PASSED - send_message called with: %s", tools.send_message.call_args
        )
    finally:
        await handler.close()


async def test_per_agent_urls_mode() -> None:
    """Test with per-agent URL mapping."""
    logger.info("=== Test: per-agent urls mode ===")
    handler = LangChainHandler(urls={"echo_agent": f"{ECHO_SERVER_URL}/invoke"})
    tools = make_fake_tools()

    try:
        await handler.handle(
            content="Hello from per-agent test!",
            room_id="room-123",
            thread_id="thread-456",
            message_id="msg-789",
            sender_id="user-1",
            sender_name="Test User",
            sender_handle="testuser",
            sender_type="User",
            mentioned_agent="echo_agent",
            tools=tools,
        )
        logger.info(
            "PASSED - send_message called with: %s", tools.send_message.call_args
        )
    finally:
        await handler.close()


async def test_from_env() -> None:
    """Test the from_env class method."""
    logger.info("=== Test: from_env ===")
    handler = LangChainHandler.from_env(ECHO_SERVER_URL)
    tools = make_fake_tools()

    try:
        await handler.handle(
            content="Hello from from_env test!",
            room_id="room-123",
            thread_id="thread-456",
            message_id="msg-789",
            sender_id="user-1",
            sender_name="Test User",
            sender_handle="testuser",
            sender_type="User",
            mentioned_agent="echo_agent",
            tools=tools,
        )
        logger.info(
            "PASSED - send_message called with: %s", tools.send_message.call_args
        )
    finally:
        await handler.close()


async def test_unknown_sender() -> None:
    """Test with a sender not in participants (no mention in response)."""
    logger.info("=== Test: unknown sender ===")
    handler = LangChainHandler(base_url=ECHO_SERVER_URL)
    tools = make_fake_tools()
    tools.participants = []

    try:
        await handler.handle(
            content="Hello from unknown sender!",
            room_id="room-123",
            thread_id="thread-456",
            message_id="msg-789",
            sender_id="user-unknown",
            sender_name=None,
            sender_handle=None,
            sender_type="User",
            mentioned_agent="echo_agent",
            tools=tools,
        )
        logger.info(
            "PASSED - send_message called with: %s", tools.send_message.call_args
        )
    finally:
        await handler.close()


async def main() -> None:
    logger.info("Starting local integration tests against %s", ECHO_SERVER_URL)
    logger.info("Make sure echo_server.py is running!\n")

    tests = [
        test_base_url_mode,
        test_per_agent_urls_mode,
        test_from_env,
        test_unknown_sender,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            await test()
            passed += 1
        except Exception:
            failed += 1
            logger.exception("FAILED")
        logger.info("")

    logger.info("Results: %d passed, %d failed", passed, failed)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
