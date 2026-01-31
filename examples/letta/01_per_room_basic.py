#!/usr/bin/env python3
"""
Basic Letta adapter example - Per-Room Mode.

Each Thenvoi room gets its own dedicated Letta agent with isolated memory
and conversation history.

Usage:
    # Set environment variables
    export THENVOI_AGENT_ID="your-agent-id"
    export THENVOI_API_KEY="your-api-key"
    export LETTA_BASE_URL="http://localhost:8283"  # Or https://api.letta.com

    # Run
    uv run --extra letta python examples/letta/01_per_room_basic.py
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

    # Configure Letta adapter in PER_ROOM mode
    # Each Thenvoi room will get its own dedicated Letta agent
    adapter = LettaAdapter(
        config=LettaConfig(
            # api_key is optional for self-hosted Letta
            mode=LettaMode.PER_ROOM,
            base_url=letta_base_url,
            # Model format: "provider/model-name"
            # Examples: "openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet-20241022"
            model="openai/gpt-4o-mini",
            embedding_model="openai/text-embedding-3-small",
            persona="""You are a helpful assistant connected to the Thenvoi platform.

Your role:
- Answer questions clearly and concisely
- Help users with their tasks
- Remember context within this conversation

Keep responses focused and helpful.""",
        ),
        # State persisted to this file (room -> agent mappings)
        state_storage_path="~/.thenvoi/letta_per_room_state.json",
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
    logger.info("Starting Letta agent in PER_ROOM mode")
    logger.info("=" * 60)
    logger.info("")
    logger.info("In this mode:")
    logger.info("  - Each room gets its own dedicated Letta agent")
    logger.info("  - Conversations are completely isolated between rooms")
    logger.info("  - Memory persists across reconnections")
    logger.info("")
    logger.info(f"Letta server: {letta_base_url}")
    logger.info("")

    # Run the agent
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
