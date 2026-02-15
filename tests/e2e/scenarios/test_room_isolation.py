"""E2E tests for room isolation.

Tests that agents in different rooms don't see each other's context.
Each room maintains independent conversation history.

This verifies:
- Per-room state management in adapters
- Platform correctly scopes /context to individual rooms
- No cross-room leakage of conversation data

Run with:
    E2E_TESTS_ENABLED=true uv run pytest tests/e2e/scenarios/test_room_isolation.py -v -s --no-cov
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from thenvoi_rest import AsyncRestClient, ChatRoomRequest
from thenvoi_rest.types import ParticipantRequest

from thenvoi.agent import Agent
from thenvoi.client.streaming import MessageCreatedPayload

from tests.e2e.adapters.conftest import ADAPTER_FACTORIES, AdapterFactory
from tests.e2e.conftest import E2ESettings, requires_e2e
from tests.e2e.helpers import (
    assert_content_contains,
    assert_no_content_contains,
    send_user_message,
)

logger = logging.getLogger(__name__)


async def _create_room_with_user(
    api_client: AsyncRestClient,
) -> tuple[str, str, str]:
    """Create a chat room and add a User peer. Returns (chat_id, user_id, user_name)."""
    response = await api_client.agent_api.create_agent_chat(chat=ChatRoomRequest())
    chat_id = response.data.id

    peers_response = await api_client.agent_api.list_agent_peers()
    user_peer = next((p for p in peers_response.data if p.type == "User"), None)
    if user_peer is None:
        pytest.skip("No User peer available for E2E tests")

    await api_client.agent_api.add_agent_chat_participant(
        chat_id,
        participant=ParticipantRequest(participant_id=user_peer.id, role="member"),
    )

    return chat_id, user_peer.id, user_peer.name


@requires_e2e
class TestRoomIsolation:
    """Test that agents in different rooms maintain isolated context."""

    @pytest.fixture(params=list(ADAPTER_FACTORIES.keys()))
    def adapter_entry(
        self, request: pytest.FixtureRequest
    ) -> tuple[str, AdapterFactory]:
        """Parametrized fixture yielding (name, factory) for each adapter."""
        name = request.param
        return name, ADAPTER_FACTORIES[name]

    async def test_agents_in_different_rooms_isolated(
        self,
        e2e_config: E2ESettings,
        ws_client,
        adapter_entry: tuple[str, AdapterFactory],
        api_client,
    ):
        """Agents in different rooms don't see each other's context.

        Room A: Send "The code is APPLE"
        Room B: Send "The code is BANANA"
        Room A: Ask "What's the code?" -> Assert "APPLE", not "BANANA"
        Room B: Ask "What's the code?" -> Assert "BANANA", not "APPLE"
        """
        adapter_name, factory = adapter_entry
        timeout = e2e_config.e2e_timeout

        logger.info("Testing room isolation with %s adapter", adapter_name)

        # Create two separate rooms
        room_a_id, user_id, user_name = await _create_room_with_user(api_client)
        room_b_id, _, _ = await _create_room_with_user(api_client)
        logger.info("Room A: %s, Room B: %s", room_a_id, room_b_id)

        # Create adapter and agent (single agent, two rooms)
        adapter = factory(e2e_config)
        agent = Agent.create(
            adapter=adapter,
            agent_id=e2e_config.test_agent_id,
            api_key=e2e_config.thenvoi_api_key,
            ws_url=e2e_config.thenvoi_ws_url,
            rest_url=e2e_config.thenvoi_base_url,
        )

        # Set up WebSocket handlers for both rooms
        room_a_received: list[MessageCreatedPayload] = []
        room_b_received: list[MessageCreatedPayload] = []
        room_a_event = asyncio.Event()
        room_b_event = asyncio.Event()

        async def room_a_handler(payload: MessageCreatedPayload) -> None:
            if payload.sender_type == "Agent" and payload.message_type == "text":
                room_a_received.append(payload)
                room_a_event.set()

        async def room_b_handler(payload: MessageCreatedPayload) -> None:
            if payload.sender_type == "Agent" and payload.message_type == "text":
                room_b_received.append(payload)
                room_b_event.set()

        await ws_client.join_chat_room_channel(room_a_id, room_a_handler)
        await ws_client.join_chat_room_channel(room_b_id, room_b_handler)

        async with agent:
            agent_name = agent.agent_name
            agent_me = await api_client.agent_api.get_agent_me()
            agent_id = agent_me.data.id

            # --- Phase 1: Set context in both rooms ---
            await send_user_message(
                api_client,
                room_a_id,
                "Remember: the secret code is APPLE. Confirm you remember it.",
                agent_name,
                agent_id,
            )
            try:
                await asyncio.wait_for(room_a_event.wait(), timeout=timeout)
            except TimeoutError:
                pass

            assert len(room_a_received) > 0, (
                f"[{adapter_name}] Room A: Agent should have acknowledged APPLE"
            )

            await send_user_message(
                api_client,
                room_b_id,
                "Remember: the secret code is BANANA. Confirm you remember it.",
                agent_name,
                agent_id,
            )
            try:
                await asyncio.wait_for(room_b_event.wait(), timeout=timeout)
            except TimeoutError:
                pass

            assert len(room_b_received) > 0, (
                f"[{adapter_name}] Room B: Agent should have acknowledged BANANA"
            )

            # --- Phase 2: Query each room and verify isolation ---
            room_a_received.clear()
            room_a_event.clear()
            room_b_received.clear()
            room_b_event.clear()

            await send_user_message(
                api_client,
                room_a_id,
                "What is the secret code? Reply with just the code word.",
                agent_name,
                agent_id,
            )
            try:
                await asyncio.wait_for(room_a_event.wait(), timeout=timeout)
            except TimeoutError:
                pass

            await send_user_message(
                api_client,
                room_b_id,
                "What is the secret code? Reply with just the code word.",
                agent_name,
                agent_id,
            )
            try:
                await asyncio.wait_for(room_b_event.wait(), timeout=timeout)
            except TimeoutError:
                pass

        # Verify Room A knows APPLE but not BANANA
        assert len(room_a_received) > 0, (
            f"[{adapter_name}] Room A: Agent should have responded about the code"
        )
        assert_content_contains(room_a_received, "APPLE")
        assert_no_content_contains(room_a_received, "BANANA")

        # Verify Room B knows BANANA but not APPLE
        assert len(room_b_received) > 0, (
            f"[{adapter_name}] Room B: Agent should have responded about the code"
        )
        assert_content_contains(room_b_received, "BANANA")
        assert_no_content_contains(room_b_received, "APPLE")

        logger.info(
            "[%s] Room isolation test PASSED: rooms are correctly isolated",
            adapter_name,
        )
