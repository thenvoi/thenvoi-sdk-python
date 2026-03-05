"""Integration tests for MentionRouter dispatch.

Simulates WebSocket payloads, dispatches through the router
to a LangChainHandler, and verifies lifecycle marking
(processing -> processed).

Prerequisites:
    1. Start the echo server:
       cd thenvoi-bridge && python echo_server.py

    2. Run this script (from repo root):
       uv run python thenvoi-bridge/test_bridge_integration.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, ".")
sys.path.insert(0, "thenvoi-bridge")

from thenvoi.client.streaming import Mention, MessageCreatedPayload, MessageMetadata

from bridge_core.router import MentionRouter
from bridge_core.session import InMemorySessionStore
from handlers.chain import LangChainHandler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger(__name__)

ECHO_SERVER_URL = "http://localhost:8000"

passed = 0
failed = 0


def _make_payload(
    sender_id: str = "user-1",
    sender_type: str = "User",
    content: str = "hello",
    msg_id: str = "msg-1",
    mentions: list[Mention] | None = None,
    thread_id: str | None = None,
) -> MessageCreatedPayload:
    return MessageCreatedPayload(
        id=msg_id,
        content=content,
        message_type="text",
        sender_id=sender_id,
        sender_type=sender_type,
        chat_room_id="room-1",
        thread_id=thread_id,
        inserted_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        metadata=MessageMetadata(mentions=mentions or [], status="sent"),
    )


def _make_tools() -> MagicMock:
    """Create mock AgentTools with participant data."""
    tools = MagicMock()
    tools.participants = [
        {"id": "user-1", "name": "Test User", "type": "User", "handle": "testuser"},
    ]
    tools.send_message = AsyncMock()
    tools.send_event = AsyncMock()
    return tools


async def run_test(name: str, coro: object) -> None:
    """Run a single test and track pass/fail."""
    global passed, failed
    logger.info("=== %s ===", name)
    try:
        await coro
        logger.info("PASSED: %s", name)
        passed += 1
    except Exception:
        logger.exception("FAILED: %s", name)
        failed += 1
    logger.info("")


async def test_single_mention_dispatch() -> None:
    """Route a single @mention through MentionRouter to LangChainHandler."""
    handler = LangChainHandler(base_url=ECHO_SERVER_URL)
    mock_link = AsyncMock()
    store = InMemorySessionStore()

    router = MentionRouter(
        agent_mapping={"echo_agent": "echo"},
        handlers={"echo": handler},
        session_store=store,
        agent_id="bridge-id",
        link=mock_link,
    )

    payload = _make_payload(
        content="Hello from integration test!",
        mentions=[Mention(id="echo-id", username="echo_agent")],
    )
    tools = _make_tools()

    try:
        await router.route(payload, "room-1", tools, sender_name="Test User")
        mock_link.mark_processing.assert_called_once_with("room-1", "msg-1")
        mock_link.mark_processed.assert_called_once_with("room-1", "msg-1")
        tools.send_message.assert_called_once()
        # Response content varies by backend (echo server vs real LLM)
        assert tools.send_message.call_args.kwargs["content"]
    finally:
        await handler.close()


async def test_multi_mention_dispatch() -> None:
    """Route a message with two @mentions to two separate handlers."""
    handler_a = LangChainHandler(base_url=ECHO_SERVER_URL)
    handler_b = LangChainHandler(base_url=ECHO_SERVER_URL)
    mock_link = AsyncMock()
    store = InMemorySessionStore()

    router = MentionRouter(
        agent_mapping={"agent_a": "handler_a", "agent_b": "handler_b"},
        handlers={"handler_a": handler_a, "handler_b": handler_b},
        session_store=store,
        agent_id="bridge-id",
        link=mock_link,
    )

    payload = _make_payload(
        content="Hello both agents!",
        mentions=[
            Mention(id="a-id", username="agent_a"),
            Mention(id="b-id", username="agent_b"),
        ],
    )
    tools = _make_tools()

    try:
        await router.route(payload, "room-1", tools, sender_name="Test User")
        mock_link.mark_processing.assert_called_once()
        mock_link.mark_processed.assert_called_once()
        assert tools.send_message.call_count == 2
    finally:
        await handler_a.close()
        await handler_b.close()


async def test_self_message_skipped() -> None:
    """Self-messages from the bridge agent should be silently skipped."""
    handler = LangChainHandler(base_url=ECHO_SERVER_URL)
    mock_link = AsyncMock()
    store = InMemorySessionStore()

    router = MentionRouter(
        agent_mapping={"echo_agent": "echo"},
        handlers={"echo": handler},
        session_store=store,
        agent_id="bridge-id",
        link=mock_link,
    )

    payload = _make_payload(
        sender_id="bridge-id",
        mentions=[Mention(id="echo-id", username="echo_agent")],
    )
    tools = _make_tools()

    try:
        await router.route(payload, "room-1", tools)
        mock_link.mark_processing.assert_not_called()
        tools.send_message.assert_not_called()
    finally:
        await handler.close()


async def test_unmapped_mention_skipped() -> None:
    """Mentions for agents not in the mapping should be skipped."""
    handler = LangChainHandler(base_url=ECHO_SERVER_URL)
    mock_link = AsyncMock()
    store = InMemorySessionStore()

    router = MentionRouter(
        agent_mapping={"echo_agent": "echo"},
        handlers={"echo": handler},
        session_store=store,
        agent_id="bridge-id",
        link=mock_link,
    )

    payload = _make_payload(
        mentions=[Mention(id="unknown-id", username="unknown_agent")],
    )
    tools = _make_tools()

    try:
        await router.route(payload, "room-1", tools)
        mock_link.mark_processing.assert_not_called()
        tools.send_message.assert_not_called()
    finally:
        await handler.close()


async def test_handler_error_marks_failed() -> None:
    """When the handler raises, router should mark_failed and send error event."""
    handler = LangChainHandler(base_url="http://localhost:19999", timeout=1)
    mock_link = AsyncMock()
    store = InMemorySessionStore()

    router = MentionRouter(
        agent_mapping={"echo_agent": "echo"},
        handlers={"echo": handler},
        session_store=store,
        agent_id="bridge-id",
        link=mock_link,
    )

    payload = _make_payload(
        mentions=[Mention(id="echo-id", username="echo_agent")],
    )
    tools = _make_tools()

    try:
        await router.route(payload, "room-1", tools, sender_name="Test User")
        mock_link.mark_processing.assert_called_once()
        mock_link.mark_failed.assert_called_once()
        tools.send_event.assert_called()
    finally:
        await handler.close()


async def test_handle_fallback_mention() -> None:
    """Mention with username=None but handle should resolve via fallback."""
    handler = LangChainHandler(base_url=ECHO_SERVER_URL)
    mock_link = AsyncMock()
    store = InMemorySessionStore()

    router = MentionRouter(
        agent_mapping={"echo_agent": "echo"},
        handlers={"echo": handler},
        session_store=store,
        agent_id="bridge-id",
        link=mock_link,
    )

    payload = _make_payload(
        content="Hello via handle fallback!",
        mentions=[Mention(id="echo-id", username=None, handle="echo_agent")],
    )
    tools = _make_tools()

    try:
        await router.route(payload, "room-1", tools, sender_name="Test User")
        mock_link.mark_processed.assert_called_once()
        tools.send_message.assert_called_once()
        # Response content varies by backend (echo server vs real LLM)
        assert tools.send_message.call_args.kwargs["content"]
    finally:
        await handler.close()


async def main() -> None:
    logger.info("Starting MentionRouter integration tests")
    logger.info("Make sure echo_server.py is running on %s!\n", ECHO_SERVER_URL)

    tests = [
        ("Single mention dispatch", test_single_mention_dispatch()),
        ("Multi-mention dispatch", test_multi_mention_dispatch()),
        ("Self-message skipped", test_self_message_skipped()),
        ("Unmapped mention skipped", test_unmapped_mention_skipped()),
        ("Handler error marks failed", test_handler_error_marks_failed()),
        ("Handle fallback mention", test_handle_fallback_mention()),
    ]

    for name, coro in tests:
        await run_test(name, coro)

    logger.info("=" * 50)
    logger.info("Results: %d passed, %d failed", passed, failed)
    logger.info("=" * 50)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
