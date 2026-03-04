"""E2E test helper functions.

Provides utilities for sending messages, waiting for agent responses,
and asserting on message content in E2E tests.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

import pytest
from thenvoi_rest import AsyncRestClient, ChatMessageRequest, ChatRoomRequest
from thenvoi_rest.types import (
    ChatMessageRequestMentionsItem as Mention,
    ParticipantRequest,
)

from thenvoi.client.streaming import MessageCreatedPayload, WebSocketClient

logger = logging.getLogger(__name__)


class TrackingWebSocketClient:
    """Wrapper around WebSocketClient that tracks joined rooms for cleanup.

    Uses a set to avoid duplicate leave calls when tests manually leave
    and rejoin the same room. Only the methods used in E2E tests are
    explicitly delegated — no ``__getattr__`` proxy — so typos are caught
    by the type checker instead of failing silently at runtime.
    """

    def __init__(self, ws: WebSocketClient) -> None:
        self._ws = ws
        self._joined_rooms: set[str] = set()

    @property
    def ws(self) -> WebSocketClient:
        """Access the underlying WebSocketClient for methods not wrapped here."""
        return self._ws

    async def join_chat_room_channel(
        self,
        chat_room_id: str,
        on_message_created: Callable[[MessageCreatedPayload], Awaitable[None]],
    ):
        result = await self._ws.join_chat_room_channel(chat_room_id, on_message_created)
        self._joined_rooms.add(chat_room_id)
        return result

    async def leave_chat_room_channel(self, chat_room_id: str):
        result = await self._ws.leave_chat_room_channel(chat_room_id)
        self._joined_rooms.discard(chat_room_id)
        return result

    async def cleanup_channels(self) -> None:
        """Leave all tracked channels. Best-effort, errors are logged."""
        for room_id in list(self._joined_rooms):
            try:
                await self._ws.leave_chat_room_channel(room_id)
            except Exception:
                logger.debug("Failed to leave room %s during cleanup", room_id)
        self._joined_rooms.clear()


async def send_user_message(
    client: AsyncRestClient,
    room_id: str,
    content: str,
    agent_name: str,
    agent_id: str,
) -> str:
    """Send a message mentioning an agent, return message_id.

    Uses the agent API to send a message with a self-mention. The platform
    treats self-mentions as triggering events, so this simulates an incoming
    user message that causes the agent to process and respond. This approach
    avoids needing separate user credentials for E2E tests.

    Args:
        client: REST API client for sending messages.
        room_id: Chat room to send the message in.
        content: Message content (agent name will be @mentioned).
        agent_name: Name of the agent to mention.
        agent_id: ID of the agent to mention.

    Returns:
        The message ID of the sent message.
    """
    message_content = f"@{agent_name} {content}"
    response = await client.agent_api_messages.create_agent_chat_message(
        room_id,
        message=ChatMessageRequest(
            content=message_content,
            mentions=[Mention(id=agent_id, name=agent_name)],
        ),
    )
    message_id = response.data.id
    logger.info("Sent message %s to room %s: %s", message_id, room_id, content[:80])
    return message_id


async def create_room_with_user(
    api_client: AsyncRestClient,
    room_tracker: list[str] | None = None,
) -> tuple[str, str, str]:
    """Create a chat room and add a User peer.

    Returns (chat_id, user_id, user_name). Rooms created here will persist
    (no delete API for agents).

    Args:
        api_client: REST API client.
        room_tracker: Optional list to append the created room ID to.
            Managed by the ``e2e_created_room_ids`` session fixture.
    """
    response = await api_client.agent_api_chats.create_agent_chat(
        chat=ChatRoomRequest()
    )
    chat_id = response.data.id

    peers_response = await api_client.agent_api_peers.list_agent_peers()
    user_peer = next((p for p in peers_response.data if p.type == "User"), None)
    if user_peer is None:
        pytest.skip("No User peer available for E2E tests")

    await api_client.agent_api_participants.add_agent_chat_participant(
        chat_id,
        participant=ParticipantRequest(participant_id=user_peer.id, role="member"),
    )

    if room_tracker is not None:
        room_tracker.append(chat_id)
    logger.info(
        "Created chat room %s with user %s (%s) (will persist, no delete API)",
        chat_id,
        user_peer.name,
        user_peer.id,
    )
    return chat_id, user_peer.id, user_peer.name


@asynccontextmanager
async def listening_for_agent_responses(
    ws_client: WebSocketClient | TrackingWebSocketClient,
    room_id: str,
    timeout: float = 30.0,
    min_messages: int = 1,
    raise_on_timeout: bool = False,
) -> AsyncGenerator[Callable[[], Awaitable[list[MessageCreatedPayload]]], None]:
    """Context manager that subscribes to a room before any messages are sent.

    Subscribes to the chat room channel on entry, yields an async ``wait``
    function, and leaves the channel on exit.  Call ``wait()`` after sending
    a message to collect agent responses without a race condition.

    Usage::

        async with listening_for_agent_responses(ws, room_id) as wait:
            await send_user_message(client, room_id, "Hello", ...)
            received = await wait()

    Args:
        ws_client: Connected WebSocket client (or TrackingWebSocketClient).
        room_id: Chat room to listen on.
        timeout: Maximum seconds ``wait()`` will block.
        min_messages: Minimum agent messages to collect before returning.
        raise_on_timeout: If True, ``wait()`` raises ``TimeoutError`` instead
            of returning partial results.

    Yields:
        An async callable that blocks until *min_messages* agent messages
        arrive (or *timeout* elapses) and returns the collected messages.
    """
    received: list[MessageCreatedPayload] = []
    event = asyncio.Event()

    async def handler(payload: MessageCreatedPayload) -> None:
        if payload.sender_type == "Agent" and payload.message_type == "text":
            received.append(payload)
            logger.info(
                "Received agent response in room %s: %s",
                room_id,
                payload.content[:80],
            )
            if len(received) >= min_messages:
                event.set()

    await ws_client.join_chat_room_channel(room_id, handler)
    try:

        async def wait() -> list[MessageCreatedPayload]:
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            except TimeoutError:
                logger.warning(
                    "Timeout waiting for agent response in room %s "
                    "(received %d/%d messages after %.1fs)",
                    room_id,
                    len(received),
                    min_messages,
                    timeout,
                )
                if raise_on_timeout:
                    raise
            return received

        yield wait
    finally:
        await ws_client.leave_chat_room_channel(room_id)


def assert_content_contains(
    messages: list[MessageCreatedPayload],
    expected_substring: str,
) -> None:
    """Assert at least one message contains the expected substring.

    Args:
        messages: List of received messages to check.
        expected_substring: Substring that should appear in at least one message.

    Raises:
        AssertionError: If no message contains the expected substring.
    """
    contents = [m.content for m in messages]
    found = any(expected_substring.lower() in c.lower() for c in contents)
    assert found, (
        f"Expected at least one message to contain '{expected_substring}', "
        f"but got: {contents}"
    )


def assert_no_content_contains(
    messages: list[MessageCreatedPayload],
    unexpected_substring: str,
) -> None:
    """Assert no message contains the unexpected substring.

    Args:
        messages: List of received messages to check.
        unexpected_substring: Substring that should NOT appear in any message.

    Raises:
        AssertionError: If any message contains the unexpected substring.
    """
    contents = [m.content for m in messages]
    found = any(unexpected_substring.lower() in c.lower() for c in contents)
    assert not found, (
        f"Expected no message to contain '{unexpected_substring}', "
        f"but found it in: {contents}"
    )


# =============================================================================
# Shared Test Workflows
# =============================================================================


async def run_smoke_test(
    ws_client: TrackingWebSocketClient,
    api_client: AsyncRestClient,
    chat_id: str,
    agent_name: str,
    e2e_agent_id: str,
    timeout: float,
    adapter_name: str,
) -> list[MessageCreatedPayload]:
    """Run a smoke test: send a message and verify the agent responds.

    Returns the list of received agent messages for further inspection.
    """
    async with listening_for_agent_responses(
        ws_client, chat_id, timeout=timeout
    ) as wait:
        await send_user_message(
            api_client, chat_id, "Say hello", agent_name, e2e_agent_id
        )
        received = await wait()

    assert len(received) > 0, (
        f"[{adapter_name}] Agent should have responded to the message"
    )
    logger.info(
        "[%s] Smoke test passed: received %d response(s)",
        adapter_name,
        len(received),
    )
    return received


async def run_tool_execution_test(
    ws_client: TrackingWebSocketClient,
    api_client: AsyncRestClient,
    chat_id: str,
    agent_name: str,
    e2e_agent_id: str,
    timeout: float,
    adapter_name: str,
) -> list[MessageCreatedPayload]:
    """Run a tool execution test: verify agent uses thenvoi_send_message.

    Asks the agent to reply with a specific keyword (PINEAPPLE) and asserts
    it appears in the response. Returns the received messages.
    """
    async with listening_for_agent_responses(
        ws_client, chat_id, timeout=timeout
    ) as wait:
        await send_user_message(
            api_client,
            chat_id,
            "Reply with the word PINEAPPLE",
            agent_name,
            e2e_agent_id,
        )
        received = await wait()

    assert len(received) > 0, (
        f"[{adapter_name}] Agent should have sent a message via tool"
    )
    assert_content_contains(received, "PINEAPPLE")
    logger.info("[%s] Tool execution test passed", adapter_name)
    return received
