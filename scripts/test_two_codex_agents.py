"""
Live test: two Codex agents talking to each other on the Thenvoi platform.

Creates two external agents, connects them via WebSocket, creates a chat room,
adds both as participants, sends a seed message, and watches them converse.

Prerequisites:
  - `codex` CLI installed and authenticated (codex login)
  - THENVOI_API_KEY env var set (user API key, thnv_u_ prefix)

Run:
    uv run python scripts/test_two_codex_agents.py

Cleanup: agents are deleted automatically on exit (Ctrl+C to stop).
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import uuid

from thenvoi import Agent
from thenvoi.adapters.codex import CodexAdapter, CodexAdapterConfig
from thenvoi.client.rest import (
    AsyncRestClient,
    ChatMessageRequest,
    ChatMessageRequestMentionsItem,
    ChatRoomRequest,
    DEFAULT_REQUEST_OPTIONS,
    ParticipantRequest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("two_codex_agents")

# Suppress noisy loggers
logging.getLogger("thenvoi.client.streaming").setLevel(logging.WARNING)
logging.getLogger("thenvoi.runtime").setLevel(logging.WARNING)
logging.getLogger("thenvoi.platform").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

REST_URL = os.getenv("THENVOI_REST_URL", "https://app.thenvoi.com")
WS_URL = os.getenv(
    "THENVOI_WS_URL", "wss://app.thenvoi.com/api/v1/socket/websocket"
)
USER_API_KEY = os.getenv("THENVOI_API_KEY")


async def create_agent(
    user_client: AsyncRestClient, name: str, description: str
) -> tuple[str, str]:
    """Create an external agent and return (agent_id, api_key)."""
    from thenvoi_rest.types import AgentRegisterRequest

    response = await user_client.human_api_agents.register_my_agent(
        agent=AgentRegisterRequest(name=name, description=description)
    )
    agent = response.data.agent
    credentials = response.data.credentials
    logger.info("Created agent: %s (ID: %s)", agent.name, agent.id)
    return agent.id, credentials.api_key


async def delete_agent(user_client: AsyncRestClient, agent_id: str) -> None:
    """Delete an agent."""
    try:
        await user_client.human_api_agents.delete_my_agent(
            id=agent_id, force=True
        )
        logger.info("Deleted agent: %s", agent_id)
    except Exception as e:
        logger.warning("Failed to delete agent %s: %s", agent_id, e)


async def main() -> None:
    if not USER_API_KEY:
        raise ValueError(
            "THENVOI_API_KEY environment variable is required (user API key with thnv_u_ prefix)"
        )

    user_client = AsyncRestClient(api_key=USER_API_KEY, base_url=REST_URL)

    suffix = uuid.uuid4().hex[:6]
    agent1_id, agent1_key = await create_agent(
        user_client,
        name=f"Codex-Alpha-{suffix}",
        description="A coding agent who asks insightful questions about software architecture.",
    )
    agent2_id, agent2_key = await create_agent(
        user_client,
        name=f"Codex-Beta-{suffix}",
        description="A coding agent who provides thoughtful answers about software design patterns.",
    )

    agents_to_cleanup = [agent1_id, agent2_id]

    try:
        # Create adapters with different roles for variety
        adapter1 = CodexAdapter(
            config=CodexAdapterConfig(
                transport="stdio",
                cwd=os.getcwd(),
                approval_policy="never",
                personality="pragmatic",
                turn_timeout_s=90.0,
                sandbox="danger-full-access",
                enable_execution_reporting=True,
                emit_thought_events=True,
                custom_section=(
                    "You are Codex-Alpha, a software architect. "
                    "Keep responses concise (2-3 sentences max). "
                    "When someone talks to you, respond briefly and ask a follow-up question."
                ),
            )
        )
        adapter2 = CodexAdapter(
            config=CodexAdapterConfig(
                transport="stdio",
                cwd=os.getcwd(),
                approval_policy="never",
                personality="pragmatic",
                turn_timeout_s=90.0,
                sandbox="danger-full-access",
                enable_execution_reporting=True,
                emit_thought_events=True,
                custom_section=(
                    "You are Codex-Beta, a senior developer. "
                    "Keep responses concise (2-3 sentences max). "
                    "When someone talks to you, respond briefly and ask a follow-up question."
                ),
            )
        )

        agent1 = Agent.create(
            adapter=adapter1,
            agent_id=agent1_id,
            api_key=agent1_key,
            ws_url=WS_URL,
            rest_url=REST_URL,
        )
        agent2 = Agent.create(
            adapter=adapter2,
            agent_id=agent2_id,
            api_key=agent2_key,
            ws_url=WS_URL,
            rest_url=REST_URL,
        )

        # Start both agents
        logger.info("Starting Agent 1 (Alpha)...")
        await agent1.start()
        logger.info("Starting Agent 2 (Beta)...")
        await agent2.start()
        logger.info("Both agents connected to platform")

        # Give WebSocket time to stabilize
        await asyncio.sleep(2)

        # Create a chat room using agent1's REST client
        agent1_client = AsyncRestClient(api_key=agent1_key, base_url=REST_URL)

        logger.info("Creating chat room...")
        chat_response = await agent1_client.agent_api_chats.create_agent_chat(
            chat=ChatRoomRequest(),
            request_options=DEFAULT_REQUEST_OPTIONS,
        )
        room_id = chat_response.data.id
        logger.info("Chat room created: %s", room_id)

        # Add agent2 as participant
        logger.info("Adding Agent 2 to room...")
        await agent1_client.agent_api_participants.add_agent_chat_participant(
            chat_id=room_id,
            participant=ParticipantRequest(participant_id=agent2_id),
            request_options=DEFAULT_REQUEST_OPTIONS,
        )
        logger.info("Agent 2 added to room")

        # Wait for both agents to join
        await asyncio.sleep(3)

        # Send seed message from agent1 mentioning agent2
        logger.info("Sending seed message to kick off conversation...")
        await agent1_client.agent_api_messages.create_agent_chat_message(
            chat_id=room_id,
            message=ChatMessageRequest(
                content=f"Hey @Codex-Beta-{suffix}, what's one design pattern you think is underrated for building resilient microservices?",
                mentions=[
                    ChatMessageRequestMentionsItem(
                        id=agent2_id,
                        name=f"Codex-Beta-{suffix}",
                    )
                ],
            ),
            request_options=DEFAULT_REQUEST_OPTIONS,
        )
        logger.info("Seed message sent! Watching conversation...")
        logger.info("=" * 60)
        logger.info("Press Ctrl+C to stop and clean up agents")
        logger.info("=" * 60)

        # Let them talk for a while
        stop_event = asyncio.Event()

        def handle_signal() -> None:
            logger.info("\nStopping...")
            stop_event.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, handle_signal)

        # Wait for Ctrl+C or 5 minutes max
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=300)
        except asyncio.TimeoutError:
            logger.info("5 minute timeout reached, stopping...")

        # Stop agents gracefully
        logger.info("Stopping agents...")
        await agent1.stop(timeout=10)
        await agent2.stop(timeout=10)
        logger.info("Agents stopped")

    finally:
        # Always clean up agents
        logger.info("Cleaning up agents...")
        for aid in agents_to_cleanup:
            await delete_agent(user_client, aid)
        logger.info("Done!")


if __name__ == "__main__":
    asyncio.run(main())
