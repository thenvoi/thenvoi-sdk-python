"""E2E tests for all standard adapters.

Verifies that each adapter can:
- Start, process a message, and stop against a real platform
- Execute platform tools (send_message)

Adapters tested: langgraph, anthropic, pydantic_ai, claude_sdk, crewai.
Parlant is excluded (requires separate server setup, see test_parlant.py).

Run with:
    E2E_TESTS_ENABLED=true uv run pytest tests/e2e/adapters/ -v -s --no-cov

Run a specific adapter:
    E2E_TESTS_ENABLED=true uv run pytest tests/e2e/adapters/ -k langgraph -v -s --no-cov
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

import pytest
from thenvoi_rest import AsyncRestClient

from thenvoi.agent import Agent

from tests.e2e.adapters.conftest import AdapterFactory
from tests.e2e.conftest import E2ESettings, requires_e2e
from tests.e2e.helpers import (
    TrackingWebSocketClient,
    assert_content_contains,
    send_user_message,
    wait_for_agent_response_ws,
)

logger = logging.getLogger(__name__)


@requires_e2e
class TestAdapterE2E:
    """E2E tests parametrized across all standard adapters."""

    @pytest.fixture
    async def running_agent(
        self,
        e2e_config: E2ESettings,
        adapter_factory: tuple[str, AdapterFactory],
    ) -> AsyncGenerator[tuple[str, Agent], None]:
        """Create and start an agent from the parametrized adapter factory.

        Yields (adapter_name, agent) with the agent running inside its
        async context manager. Ensures clean shutdown on exit.
        """
        adapter_name, factory = adapter_factory
        adapter = factory(e2e_config)

        agent = Agent.create(
            adapter=adapter,
            agent_id=e2e_config.test_agent_id,
            api_key=e2e_config.thenvoi_api_key,
            ws_url=e2e_config.thenvoi_ws_url,
            rest_url=e2e_config.thenvoi_base_url,
        )

        async with agent:
            yield adapter_name, agent

    async def test_smoke_responds_to_message(
        self,
        e2e_config: E2ESettings,
        e2e_chat_room_with_user: tuple[str, str, str],
        ws_client: TrackingWebSocketClient,
        running_agent: tuple[str, Agent],
        api_client: AsyncRestClient,
        e2e_agent_id: str,
    ):
        """Smoke test: agent starts, receives a message, and responds."""
        adapter_name, agent = running_agent
        chat_id, user_id, user_name = e2e_chat_room_with_user

        # Self-mention triggers agent processing (see send_user_message docs)
        await send_user_message(
            api_client, chat_id, "Say hello", agent.agent_name, e2e_agent_id
        )

        received = await wait_for_agent_response_ws(
            ws_client, chat_id, timeout=e2e_config.e2e_timeout
        )

        assert len(received) > 0, (
            f"[{adapter_name}] Agent should have responded to the message"
        )
        logger.info(
            "[%s] Smoke test passed: received %d response(s)",
            adapter_name,
            len(received),
        )

    async def test_tool_execution_send_message(
        self,
        e2e_config: E2ESettings,
        e2e_chat_room_with_user: tuple[str, str, str],
        ws_client: TrackingWebSocketClient,
        running_agent: tuple[str, Agent],
        api_client: AsyncRestClient,
        e2e_agent_id: str,
    ):
        """Verify the agent uses thenvoi_send_message tool to respond."""
        adapter_name, agent = running_agent
        chat_id, user_id, user_name = e2e_chat_room_with_user

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

        assert len(received) > 0, (
            f"[{adapter_name}] Agent should have sent a message via tool"
        )
        assert_content_contains(received, "PINEAPPLE")
        logger.info("[%s] Tool execution test passed", adapter_name)
