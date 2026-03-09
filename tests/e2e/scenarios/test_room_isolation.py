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

import logging

import pytest
from thenvoi_rest import AsyncRestClient

from thenvoi.agent import Agent

from tests.e2e.adapters.conftest import AdapterFactory
from tests.e2e.conftest import E2ESettings, requires_e2e
from tests.e2e.helpers import (
    TrackingWebSocketClient,
    assert_content_contains,
    assert_no_content_contains,
    create_room_with_user,
    listening_for_agent_responses,
    send_user_message,
)

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
@requires_e2e
class TestRoomIsolation:
    """Test that agents in different rooms maintain isolated context."""

    @pytest.mark.flaky(reruns=2)
    async def test_agents_in_different_rooms_isolated(
        self,
        e2e_config: E2ESettings,
        ws_client: TrackingWebSocketClient,
        adapter_entry: tuple[str, AdapterFactory],
        api_client: AsyncRestClient,
        e2e_agent_id: str,
    ):
        """Agents in different rooms don't see each other's context.

        Room A: Send "The code is APPLE"
        Room B: Send "The code is BANANA"
        Room A: Ask "What's the code?" -> Assert "APPLE", not "BANANA"
        Room B: Ask "What's the code?" -> Assert "BANANA", not "APPLE"

        Note: This test creates 2 rooms per adapter via ``create_room_with_user``
        (not the ``e2e_chat_room_with_user`` fixture). These rooms persist because
        there is no delete API for agents. Expect room accumulation across runs.
        """
        adapter_name, factory = adapter_entry
        timeout = e2e_config.e2e_timeout

        logger.info("Testing room isolation with %s adapter", adapter_name)

        # Create two separate rooms
        room_a_id, user_id, user_name = await create_room_with_user(api_client)
        room_b_id, _, _ = await create_room_with_user(api_client)
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

        async with agent:
            agent_name = agent.agent_name

            # --- Phase 1: Set context in both rooms sequentially ---
            # Sequential to avoid flakiness: a single agent processes one
            # room at a time, so concurrent sends can cause timeouts.
            async with listening_for_agent_responses(
                ws_client, room_a_id, timeout=timeout, raise_on_timeout=True
            ) as wait:
                await send_user_message(
                    api_client,
                    room_a_id,
                    "Remember: the secret code is APPLE. Confirm you remember it.",
                    agent_name,
                    e2e_agent_id,
                )
                room_a_phase1 = await wait()

            async with listening_for_agent_responses(
                ws_client, room_b_id, timeout=timeout, raise_on_timeout=True
            ) as wait:
                await send_user_message(
                    api_client,
                    room_b_id,
                    "Remember: the secret code is BANANA. Confirm you remember it.",
                    agent_name,
                    e2e_agent_id,
                )
                room_b_phase1 = await wait()

            logger.info(
                "[%s] Phase 1 complete: Room A got %d, Room B got %d response(s)",
                adapter_name,
                len(room_a_phase1),
                len(room_b_phase1),
            )

            # --- Phase 2: Query each room and verify isolation ---
            async with listening_for_agent_responses(
                ws_client, room_a_id, timeout=timeout, raise_on_timeout=True
            ) as wait:
                await send_user_message(
                    api_client,
                    room_a_id,
                    "What is the secret code? Reply with just the code word.",
                    agent_name,
                    e2e_agent_id,
                )
                room_a_received = await wait()

            async with listening_for_agent_responses(
                ws_client, room_b_id, timeout=timeout, raise_on_timeout=True
            ) as wait:
                await send_user_message(
                    api_client,
                    room_b_id,
                    "What is the secret code? Reply with just the code word.",
                    agent_name,
                    e2e_agent_id,
                )
                room_b_received = await wait()

            # Verify Room A knows APPLE but not BANANA
            assert_content_contains(room_a_received, "APPLE")
            assert_no_content_contains(room_a_received, "BANANA")

            # Verify Room B knows BANANA but not APPLE
            assert_content_contains(room_b_received, "BANANA")
            assert_no_content_contains(room_b_received, "APPLE")

        logger.info(
            "[%s] Room isolation test PASSED: rooms are correctly isolated",
            adapter_name,
        )
        logger.info(
            "E2E test rooms %s, %s will persist (no delete API)",
            room_a_id,
            room_b_id,
        )
