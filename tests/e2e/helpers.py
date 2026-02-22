"""E2E test helper functions.

Provides utilities for sending messages, waiting for agent responses,
and asserting on message content in E2E tests.
"""

from __future__ import annotations

import asyncio
import logging

from thenvoi_rest import AsyncRestClient, ChatMessageRequest
from thenvoi_rest.types import ChatMessageRequestMentionsItem as Mention

from thenvoi.client.streaming import MessageCreatedPayload, WebSocketClient

logger = logging.getLogger(__name__)


async def send_user_message(
    client: AsyncRestClient,
    room_id: str,
    content: str,
    agent_name: str,
    agent_id: str,
) -> str:
    """Send a message mentioning an agent, return message_id.

    The message is sent as the agent (via agent API) mentioning itself
    to simulate an incoming user message that triggers the agent.

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
    response = await client.agent_api.create_agent_chat_message(
        room_id,
        message=ChatMessageRequest(
            content=message_content,
            mentions=[Mention(id=agent_id, name=agent_name)],
        ),
    )
    message_id = response.data.id
    logger.info("Sent message %s to room %s: %s", message_id, room_id, content[:80])
    return message_id


async def wait_for_agent_response_ws(
    ws_client: WebSocketClient,
    room_id: str,
    timeout: float = 30.0,
    min_messages: int = 1,
) -> list[MessageCreatedPayload]:
    """Wait for agent response(s) via WebSocket.

    Subscribes to the chat room channel and waits for messages from
    agents (sender_type == "Agent"). Returns once at least min_messages
    are received or timeout is reached.

    Args:
        ws_client: Connected WebSocket client.
        room_id: Chat room to listen on.
        timeout: Maximum seconds to wait.
        min_messages: Minimum number of agent messages to wait for.

    Returns:
        List of received agent messages.
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

    return received


async def wait_for_agent_response_polling(
    client: AsyncRestClient,
    room_id: str,
    after_message_id: str,
    timeout: float = 30.0,
    poll_interval: float = 1.0,
) -> list[dict]:
    """Wait for agent response by polling /context endpoint.

    Fallback method when WebSocket is not available. Polls the context
    API and looks for new messages after the given message ID.

    Args:
        client: REST API client.
        room_id: Chat room to poll.
        after_message_id: Only return messages sent after this ID.
        timeout: Maximum seconds to wait.
        poll_interval: Seconds between polls.

    Returns:
        List of new context items (messages/events) from the agent.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    seen_ids = {after_message_id}

    while loop.time() < deadline:
        response = await client.agent_api.get_agent_chat_context(room_id)
        context = response.data or []

        new_items = []
        for item in context:
            if hasattr(item, "id") and item.id not in seen_ids:
                # Check if this is an agent message (not our own)
                if hasattr(item, "sender_type") and item.sender_type == "Agent":
                    new_items.append(item)
                    seen_ids.add(item.id)

        if new_items:
            logger.info(
                "Found %d new agent messages in room %s via polling",
                len(new_items),
                room_id,
            )
            return new_items

        await asyncio.sleep(poll_interval)

    logger.warning(
        "Timeout polling for agent response in room %s after %.1fs",
        room_id,
        timeout,
    )
    return []


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
