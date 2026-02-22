"""E2E tests for context persistence across agent rejoin.

Tests that when an agent is removed from a room and re-added, it correctly
loads history from the platform and remembers prior context.

This verifies:
- is_session_bootstrap=True correctly loads history on rejoin
- The platform stores and returns conversation history via /context
- Adapters correctly convert and use historical messages

Run with:
    E2E_TESTS_ENABLED=true uv run pytest tests/e2e/scenarios/test_context_persistence.py -v -s --no-cov
"""

from __future__ import annotations

import logging

from thenvoi.agent import Agent

from tests.e2e.adapters.conftest import AdapterFactory
from tests.e2e.conftest import E2ESettings, requires_e2e
from tests.e2e.helpers import (
    assert_content_contains,
    send_user_message,
    wait_for_agent_response_ws,
)

logger = logging.getLogger(__name__)


@requires_e2e
class TestContextPersistence:
    """Test that agents remember context after rejoin via platform history."""

    async def test_agent_remembers_context_after_rejoin(
        self,
        e2e_config: E2ESettings,
        e2e_chat_room_with_user: tuple[str, str, str],
        ws_client,
        adapter_entry: tuple[str, AdapterFactory],
        api_client,
    ):
        """Agent removed then re-added remembers context via platform history.

        Phase 1: Send "Remember the code: ABC123", wait for acknowledgment
        Phase 2: Stop agent (triggers on_cleanup), restart (triggers bootstrap)
        Phase 3: Ask "What was the code?", assert response contains "ABC123"
        """
        adapter_name, factory = adapter_entry
        chat_id, user_id, user_name = e2e_chat_room_with_user
        timeout = e2e_config.e2e_timeout

        logger.info("Testing context persistence with %s adapter", adapter_name)

        # --- Phase 1: Establish context ---
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
            agent_me = await api_client.agent_api.get_agent_me()
            agent_id = agent_me.data.id

            # Send context-setting message
            await send_user_message(
                api_client,
                chat_id,
                "Remember this secret code: ABC123. Respond confirming you remember it.",
                agent_name,
                agent_id,
            )

            phase1_received = await wait_for_agent_response_ws(
                ws_client, chat_id, timeout=timeout, raise_on_timeout=True
            )

        logger.info(
            "[%s] Phase 1 complete: agent acknowledged with %d message(s)",
            adapter_name,
            len(phase1_received),
        )

        # --- Phase 2: Restart agent (new adapter instance, fresh state) ---
        # The old adapter is gone (on_cleanup was called). Create a fresh one.
        adapter2 = factory(e2e_config)
        agent2 = Agent.create(
            adapter=adapter2,
            agent_id=e2e_config.test_agent_id,
            api_key=e2e_config.thenvoi_api_key,
            ws_url=e2e_config.thenvoi_ws_url,
            rest_url=e2e_config.thenvoi_base_url,
        )

        # Leave and rejoin channel to reset the handler for fresh message collection
        await ws_client.leave_chat_room_channel(chat_id)

        async with agent2:
            agent_name2 = agent2.agent_name
            # agent_id is the same

            # Ask about the code - agent should load history from platform
            await send_user_message(
                api_client,
                chat_id,
                "What was the secret code I told you to remember? Reply with just the code.",
                agent_name2,
                agent_id,
            )

            phase2_received = await wait_for_agent_response_ws(
                ws_client, chat_id, timeout=timeout, raise_on_timeout=True
            )

        assert_content_contains(phase2_received, "ABC123")
        logger.info(
            "[%s] Context persistence test PASSED: agent remembered ABC123",
            adapter_name,
        )
