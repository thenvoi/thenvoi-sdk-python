#!/usr/bin/env python3
"""Test inter-agent communication between planner and reviewer.

Creates a chat room, adds all 3 agents, sends a test message,
and verifies delivery by checking container logs.

Usage:
    python test_communication.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / "agent_config.yaml"


def load_agent_configs() -> tuple[dict, dict]:
    """Load planner + reviewer entries from the keyed agent_config.yaml.

    `coding_agents/` uses a single keyed YAML for all agents (matching
    docker-compose's `AGENT_KEY: planner` / `AGENT_KEY: reviewer` selectors)
    rather than the older per-role files.
    """
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f) or {}
    if "planner" not in config or "reviewer" not in config:
        raise ValueError(
            f"agent_config.yaml at {CONFIG_PATH} must define both "
            "`planner` and `reviewer` keys (copy from agent_config.yaml.example)."
        )
    return config["planner"], config["reviewer"]


async def main() -> None:
    from thenvoi_rest import AsyncRestClient
    from thenvoi_rest.types import (
        ChatMessageRequest,
        ChatMessageRequestMentionsItem,
        ChatRoomRequest,
        ParticipantRequest,
    )

    # Load agent configs from the shared keyed yaml.
    planner, reviewer = load_agent_configs()

    base_url = os.environ.get("THENVOI_REST_URL", "https://app.band.ai")

    # Use planner as the "orchestrator" to create the room
    client = AsyncRestClient(api_key=planner["api_key"], base_url=base_url)

    # Step 1: Create a chat room
    logger.info("Creating chat room...")
    room_response = await client.agent_api_chats.create_agent_chat(
        chat=ChatRoomRequest()
    )
    room = room_response.data
    room_id = room.id
    logger.info("  Room created: %s", room_id)

    # Step 2: Add reviewer as participant
    for name, agent_config in [("Reviewer", reviewer)]:
        logger.info("Adding %s to room...", name)
        await client.agent_api_participants.add_agent_chat_participant(
            chat_id=room_id,
            participant=ParticipantRequest(participant_id=agent_config["agent_id"]),
        )
        logger.info("  %s added", name)

    # Give agents time to join the room via WebSocket
    logger.info("Waiting for agents to join room...")
    await asyncio.sleep(3)

    # Step 3: Send a test message mentioning reviewer
    logger.info("Sending test message...")
    mentions = [
        ChatMessageRequestMentionsItem(
            id=reviewer["agent_id"],
            name="Reviewer",
        ),
    ]

    msg_response = await client.agent_api_messages.create_agent_chat_message(
        chat_id=room_id,
        message=ChatMessageRequest(
            content="Hello @Reviewer! This is a test message from the planner. Please confirm you received this by saying 'acknowledged'.",
            mentions=mentions,
        ),
    )
    logger.info("  Message sent: %s", msg_response.data.id)

    # Step 4: Wait and check for responses
    logger.info("\nWaiting 30s for agent responses...")
    await asyncio.sleep(30)

    # List messages in the room to see responses
    logger.info("\n=== Messages in room ===")
    messages_response = await client.agent_api_messages.list_agent_messages(
        chat_id=room_id,
    )

    for msg in messages_response.data:
        sender = getattr(msg, "sender_name", None) or msg.sender_id
        content = msg.content[:120] if msg.content else "(empty)"
        msg_type = getattr(msg, "message_type", "unknown")
        logger.info("[%s] %s: %s", msg_type, sender, content)

    logger.info("\nRoom ID: %s", room_id)
    logger.info("Test complete! Check docker compose logs for full agent activity.")


if __name__ == "__main__":
    asyncio.run(main())
