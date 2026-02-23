"""
Live test: three Codex agents discussing the codex integration in this repo.

Creates three external agents, connects them via WebSocket, creates a chat room,
adds all as participants, sends a seed message about the codex adapter code.

Prerequisites:
  - `codex` CLI installed and authenticated (codex login)
  - THENVOI_API_KEY env var set (user API key, thnv_u_ prefix)

Run:
    uv run python scripts/test_three_codex_agents.py

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
logger = logging.getLogger("three_codex_agents")

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
CWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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

    agent_defs = [
        (
            f"Architect-{suffix}",
            "A software architect who reviews code structure, patterns, and integration design.",
            (
                "You are Architect, a senior software architect. "
                "You are reviewing the Codex adapter integration in this Python SDK repo. "
                "The codebase is at your cwd. Focus on src/thenvoi/adapters/codex.py and "
                "tests/adapters/test_codex_adapter.py. "
                "Keep responses concise (3-4 sentences max). "
                "Read files before commenting. Mention specific code when possible."
            ),
        ),
        (
            f"Reviewer-{suffix}",
            "A code reviewer who checks for bugs, edge cases, and test coverage.",
            (
                "You are Reviewer, a meticulous code reviewer. "
                "You are reviewing the Codex adapter integration in this Python SDK repo. "
                "The codebase is at your cwd. Focus on src/thenvoi/adapters/codex.py and "
                "tests/adapters/test_codex_adapter.py. "
                "Keep responses concise (3-4 sentences max). "
                "Read files before commenting. Look for edge cases and missing tests."
            ),
        ),
        (
            f"Tester-{suffix}",
            "A QA engineer who validates test coverage and suggests improvements.",
            (
                "You are Tester, a QA engineer. "
                "You are reviewing the Codex adapter integration in this Python SDK repo. "
                "The codebase is at your cwd. Focus on tests/adapters/test_codex_adapter.py "
                "and tests/integrations/codex/test_adapter_e2e.py. "
                "Keep responses concise (3-4 sentences max). "
                "Read test files before commenting. Suggest concrete missing test scenarios."
            ),
        ),
    ]

    agents_to_cleanup: list[str] = []
    agent_ids: list[str] = []
    agent_keys: list[str] = []
    agent_names: list[str] = []

    try:
        for name, description, _ in agent_defs:
            aid, akey = await create_agent(user_client, name, description)
            agents_to_cleanup.append(aid)
            agent_ids.append(aid)
            agent_keys.append(akey)
            agent_names.append(name)

        adapters = []
        for _, _, custom_section in agent_defs:
            adapters.append(
                CodexAdapter(
                    config=CodexAdapterConfig(
                        transport="stdio",
                        cwd=CWD,
                        approval_policy="never",
                        personality="pragmatic",
                        turn_timeout_s=120.0,
                        sandbox="danger-full-access",
                        enable_execution_reporting=True,
                        emit_thought_events=True,
                        custom_section=custom_section,
                    )
                )
            )

        agents = []
        for i in range(3):
            agents.append(
                Agent.create(
                    adapter=adapters[i],
                    agent_id=agent_ids[i],
                    api_key=agent_keys[i],
                    ws_url=WS_URL,
                    rest_url=REST_URL,
                )
            )

        for i, agent in enumerate(agents):
            logger.info("Starting %s...", agent_names[i])
            await agent.start()
            logger.info("%s connected", agent_names[i])

        await asyncio.sleep(2)

        # Create chat room using first agent's credentials
        room_client = AsyncRestClient(api_key=agent_keys[0], base_url=REST_URL)

        logger.info("Creating chat room...")
        chat_response = await room_client.agent_api_chats.create_agent_chat(
            chat=ChatRoomRequest(),
            request_options=DEFAULT_REQUEST_OPTIONS,
        )
        room_id = chat_response.data.id
        logger.info("Chat room created: %s", room_id)

        # Add other agents as participants
        for i in range(1, 3):
            logger.info("Adding %s to room...", agent_names[i])
            await room_client.agent_api_participants.add_agent_chat_participant(
                chat_id=room_id,
                participant=ParticipantRequest(participant_id=agent_ids[i]),
                request_options=DEFAULT_REQUEST_OPTIONS,
            )

        await asyncio.sleep(3)

        # Build mentions for all other agents
        mentions = [
            ChatMessageRequestMentionsItem(id=agent_ids[1], name=agent_names[1]),
            ChatMessageRequestMentionsItem(id=agent_ids[2], name=agent_names[2]),
        ]

        seed_message = (
            f"@{agent_names[1]} @{agent_names[2]} — "
            "Let's review the recent changes to the Codex adapter. "
            "Start by reading src/thenvoi/adapters/codex.py — specifically the "
            "_emit_item_completed_events method and the item/completed handler in on_message. "
            "These were just added to forward internal Codex operations (shell commands, file edits, "
            "MCP tool calls) as platform events. Also check the execution reporting suppression "
            "for thenvoi_send_message and thenvoi_send_event. "
            "What do you think of the implementation? Any issues?"
        )

        logger.info("Sending seed message...")
        await room_client.agent_api_messages.create_agent_chat_message(
            chat_id=room_id,
            message=ChatMessageRequest(content=seed_message, mentions=mentions),
            request_options=DEFAULT_REQUEST_OPTIONS,
        )
        logger.info("Seed message sent!")
        logger.info("=" * 60)
        logger.info("Room: %s", room_id)
        logger.info("Press Ctrl+C to stop and clean up agents")
        logger.info("=" * 60)

        stop_event = asyncio.Event()

        def handle_signal() -> None:
            logger.info("\nStopping...")
            stop_event.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, handle_signal)

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=600)
        except asyncio.TimeoutError:
            logger.info("10 minute timeout reached, stopping...")

        logger.info("Stopping agents...")
        for i, agent in enumerate(agents):
            await agent.stop(timeout=10)
            logger.info("%s stopped", agent_names[i])

    finally:
        logger.info("Cleaning up agents...")
        for aid in agents_to_cleanup:
            await delete_agent(user_client, aid)
        logger.info("Done!")


if __name__ == "__main__":
    asyncio.run(main())
