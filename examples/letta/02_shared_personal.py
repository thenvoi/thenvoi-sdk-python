#!/usr/bin/env python3
"""
Personal Letta agent example - Shared Mode.

Single Letta agent across all rooms, maintaining persistent identity and
shared memory. Uses Conversations API for thread-safe parallel room handling.

Usage:
    # Set environment variables
    export THENVOI_AGENT_ID="your-agent-id"
    export THENVOI_API_KEY="your-api-key"
    export LETTA_BASE_URL="http://localhost:8283"  # Or https://api.letta.com

    # Run
    uv run --extra letta python examples/letta/02_shared_personal.py
"""

from __future__ import annotations

import asyncio
import logging
import os

from thenvoi import Agent
from thenvoi.adapters.letta import LettaAdapter, LettaConfig, LettaMode
from thenvoi.runtime.types import SessionConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    # Load configuration from environment
    agent_id = os.environ.get("THENVOI_AGENT_ID")
    api_key = os.environ.get("THENVOI_API_KEY")
    letta_base_url = os.environ.get("LETTA_BASE_URL", "http://localhost:8283")

    if not agent_id or not api_key:
        raise ValueError(
            "Missing required environment variables: "
            "THENVOI_AGENT_ID and THENVOI_API_KEY"
        )

    # Configure Letta adapter in SHARED mode
    # One agent serves all rooms via separate conversations
    adapter = LettaAdapter(
        config=LettaConfig(
            # api_key is optional for self-hosted Letta
            mode=LettaMode.SHARED,
            base_url=letta_base_url,
            # Model format: "provider/model-name"
            # Examples: "openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet-20241022"
            model="openai/gpt-4o-mini",
            embedding_model="openai/text-embedding-3-small",
            persona="""You are a personal AI assistant with persistent memory.

Your capabilities:
- Remember information across all conversations
- Track context for different rooms via your room_contexts memory
- Maintain a consistent personality and relationship with users
- Help with tasks, answer questions, and provide recommendations

Personality:
- Friendly and helpful
- Proactive in offering assistance
- Remember user preferences and past interactions

Memory Management:
- Update your room_contexts memory after significant interactions
- Keep track of important decisions, preferences, and action items
- Reference past conversations when relevant""",
        ),
        # State persisted to this file (room -> conversation mappings)
        state_storage_path="~/.thenvoi/letta_shared_state.json",
    )

    # Create Thenvoi agent
    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        # IMPORTANT: Disable Thenvoi's context hydration
        # Letta manages its own conversation history server-side
        session_config=SessionConfig(enable_context_hydration=False),
    )

    logger.info("=" * 60)
    logger.info("Starting Letta agent in SHARED mode")
    logger.info("=" * 60)
    logger.info("")
    logger.info("In this mode:")
    logger.info("  - One Letta agent serves all rooms")
    logger.info("  - Each room has its own conversation (thread-safe)")
    logger.info("  - Memory blocks are shared across all conversations")
    logger.info("  - The agent maintains a single persistent identity")
    logger.info("")
    logger.info("Architecture:")
    logger.info("  room-A -> conversation-1 ──┐")
    logger.info("  room-B -> conversation-2 ──┼─> Same agent, shared memory")
    logger.info("  room-C -> conversation-3 ──┘")
    logger.info("")
    logger.info(f"Letta server: {letta_base_url}")
    logger.info("")

    # Run the agent
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
