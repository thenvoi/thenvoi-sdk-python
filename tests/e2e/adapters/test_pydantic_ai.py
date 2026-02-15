"""E2E tests for the Pydantic AI adapter.

Verifies that the Pydantic AI adapter can:
- Start, process a message, and stop against a real platform
- Execute platform tools (send_message)

Run with:
    E2E_TESTS_ENABLED=true uv run pytest tests/e2e/adapters/test_pydantic_ai.py -v -s --no-cov
"""

from __future__ import annotations

import asyncio
import logging


from thenvoi.agent import Agent
from thenvoi.client.streaming import MessageCreatedPayload

from tests.e2e.conftest import E2ESettings, requires_e2e
from tests.e2e.helpers import (
    assert_content_contains,
    send_user_message,
)

logger = logging.getLogger(__name__)


@requires_e2e
class TestPydanticAIE2E:
    """E2E tests specific to the Pydantic AI adapter."""

    async def test_smoke_responds_to_message(
        self,
        e2e_config: E2ESettings,
        e2e_chat_room_with_user: tuple[str, str, str],
        ws_client,
        pydantic_ai_adapter_factory,
        api_client,
    ):
        """Smoke test: agent starts, receives a message, and responds."""
        chat_id, user_id, user_name = e2e_chat_room_with_user
        adapter = pydantic_ai_adapter_factory(e2e_config)

        agent = Agent.create(
            adapter=adapter,
            agent_id=e2e_config.test_agent_id,
            api_key=e2e_config.thenvoi_api_key,
            ws_url=e2e_config.thenvoi_ws_url,
            rest_url=e2e_config.thenvoi_base_url,
        )

        received: list[MessageCreatedPayload] = []
        response_event = asyncio.Event()

        async def on_message(payload: MessageCreatedPayload) -> None:
            if payload.sender_type == "Agent" and payload.message_type == "text":
                received.append(payload)
                response_event.set()

        await ws_client.join_chat_room_channel(chat_id, on_message)

        async with agent:
            agent_name = agent.agent_name
            agent_me = await api_client.agent_api.get_agent_me()
            agent_id = agent_me.data.id

            await send_user_message(
                api_client, chat_id, "Say hello", agent_name, agent_id
            )

            try:
                await asyncio.wait_for(
                    response_event.wait(), timeout=e2e_config.e2e_timeout
                )
            except TimeoutError:
                pass

        assert len(received) > 0, "Agent should have responded to the message"
        logger.info(
            "Pydantic AI smoke test passed: received %d response(s)", len(received)
        )

    async def test_tool_execution_send_message(
        self,
        e2e_config: E2ESettings,
        e2e_chat_room_with_user: tuple[str, str, str],
        ws_client,
        pydantic_ai_adapter_factory,
        api_client,
    ):
        """Verify the agent uses thenvoi_send_message tool to respond."""
        chat_id, user_id, user_name = e2e_chat_room_with_user
        adapter = pydantic_ai_adapter_factory(e2e_config)

        agent = Agent.create(
            adapter=adapter,
            agent_id=e2e_config.test_agent_id,
            api_key=e2e_config.thenvoi_api_key,
            ws_url=e2e_config.thenvoi_ws_url,
            rest_url=e2e_config.thenvoi_base_url,
        )

        received: list[MessageCreatedPayload] = []
        response_event = asyncio.Event()

        async def on_message(payload: MessageCreatedPayload) -> None:
            if payload.sender_type == "Agent" and payload.message_type == "text":
                received.append(payload)
                response_event.set()

        await ws_client.join_chat_room_channel(chat_id, on_message)

        async with agent:
            agent_name = agent.agent_name
            agent_me = await api_client.agent_api.get_agent_me()
            agent_id = agent_me.data.id

            await send_user_message(
                api_client,
                chat_id,
                "Reply with the word PINEAPPLE",
                agent_name,
                agent_id,
            )

            try:
                await asyncio.wait_for(
                    response_event.wait(), timeout=e2e_config.e2e_timeout
                )
            except TimeoutError:
                pass

        assert len(received) > 0, "Agent should have sent a message via tool"
        assert_content_contains(received, "PINEAPPLE")
        logger.info("Pydantic AI tool execution test passed")
