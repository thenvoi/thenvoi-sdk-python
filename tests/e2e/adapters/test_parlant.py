"""E2E tests for the Parlant adapter.

Verifies that the Parlant adapter can:
- Start, process a message, and stop against a real platform
- Execute platform tools (send_message)

Note: Parlant requires a running Parlant server. These tests create
a server + agent in-process using the Parlant SDK.

Run with:
    E2E_TESTS_ENABLED=true uv run pytest tests/e2e/adapters/test_parlant.py -v -s --no-cov
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

import pytest

from thenvoi.agent import Agent

from tests.e2e.conftest import E2ESettings, requires_e2e
from tests.e2e.helpers import (
    assert_content_contains,
    send_user_message,
    wait_for_agent_response_ws,
)

logger = logging.getLogger(__name__)

try:
    import parlant.sdk as p

    HAS_PARLANT = True
except ImportError:
    HAS_PARLANT = False

requires_parlant = pytest.mark.skipif(not HAS_PARLANT, reason="parlant not installed")


@requires_e2e
@requires_parlant
class TestParlantE2E:
    """E2E tests specific to the Parlant adapter.

    These tests require Parlant to be installed and create an in-process
    Parlant server for each test.
    """

    @pytest.fixture
    async def running_parlant_agent(
        self,
        e2e_config: E2ESettings,
    ) -> AsyncGenerator[Agent, None]:
        """Create a Parlant adapter with an in-process server and start the agent.

        Yields a running Agent inside its async context manager.
        """
        from thenvoi.adapters.parlant import ParlantAdapter

        async with p.Server() as server:
            parlant_agent = await server.create_agent(
                name="E2E Test Agent",
                description="A test agent for E2E validation. Keep responses short.",
            )

            adapter = ParlantAdapter(
                server=server,
                parlant_agent=parlant_agent,
                custom_section="Keep responses short and concise.",
            )

            agent = Agent.create(
                adapter=adapter,
                agent_id=e2e_config.test_agent_id,
                api_key=e2e_config.thenvoi_api_key,
                ws_url=e2e_config.thenvoi_ws_url,
                rest_url=e2e_config.thenvoi_base_url,
            )

            async with agent:
                yield agent

    async def test_smoke_responds_to_message(
        self,
        e2e_config: E2ESettings,
        e2e_chat_room_with_user: tuple[str, str, str],
        ws_client,
        running_parlant_agent: Agent,
        api_client,
        e2e_agent_id: str,
    ):
        """Smoke test: agent starts, receives a message, and responds."""
        chat_id, user_id, user_name = e2e_chat_room_with_user
        agent = running_parlant_agent

        # Self-mention triggers agent processing (see send_user_message docs)
        await send_user_message(
            api_client, chat_id, "Say hello", agent.agent_name, e2e_agent_id
        )

        received = await wait_for_agent_response_ws(
            ws_client, chat_id, timeout=e2e_config.e2e_timeout
        )

        assert len(received) > 0, "Agent should have responded to the message"
        logger.info("Parlant smoke test passed: received %d response(s)", len(received))

    async def test_tool_execution_send_message(
        self,
        e2e_config: E2ESettings,
        e2e_chat_room_with_user: tuple[str, str, str],
        ws_client,
        running_parlant_agent: Agent,
        api_client,
        e2e_agent_id: str,
    ):
        """Verify the agent uses thenvoi_send_message tool to respond."""
        chat_id, user_id, user_name = e2e_chat_room_with_user
        agent = running_parlant_agent

        # Self-mention triggers agent processing (see send_user_message docs)
        await send_user_message(
            api_client,
            chat_id,
            "Reply with the word PINEAPPLE",
            agent.agent_name,
            e2e_agent_id,
        )

        received = await wait_for_agent_response_ws(
            ws_client, chat_id, timeout=e2e_config.e2e_timeout
        )

        assert len(received) > 0, "Agent should have sent a message via tool"
        assert_content_contains(received, "PINEAPPLE")
        logger.info("Parlant tool execution test passed")
