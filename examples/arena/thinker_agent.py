# /// script
# requires-python = ">=3.11"
# dependencies = ["thenvoi-sdk[langgraph]"]
#
# [tool.uv.sources]
# thenvoi-sdk = { git = "https://github.com/thenvoi/thenvoi-sdk-python.git" }
# ///
"""
Thinker agent for the 20 Questions arena game.

The Thinker picks a secret word from a category (animals, foods, objects,
vehicles), announces the category to the room, invites the Guesser agent,
and answers yes/no questions for up to 20 rounds.

Supports both OpenAI and Anthropic LLMs -- set either OPENAI_API_KEY or
ANTHROPIC_API_KEY in your environment.

Run with (from repo root):
    uv run examples/arena/thinker_agent.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from arena.prompts import create_llm, generate_thinker_prompt

from arena.setup_logging import setup_logging
from thenvoi import Agent
from thenvoi.adapters import LangGraphAdapter
from thenvoi.config import load_agent_config

setup_logging()
logger = logging.getLogger(__name__)


async def main() -> None:
    load_dotenv()

    ws_url = os.getenv("THENVOI_WS_URL")
    rest_url = os.getenv("THENVOI_REST_URL")

    if not ws_url:
        raise ValueError("THENVOI_WS_URL environment variable is required")
    if not rest_url:
        raise ValueError("THENVOI_REST_URL environment variable is required")

    # Load Thinker's credentials from agent_config.yaml
    agent_id, api_key = load_agent_config("arena_thinker")

    # Select LLM based on available API keys
    llm = create_llm()

    # Create adapter with Thinker's game prompt
    adapter = LangGraphAdapter(
        llm=llm,
        checkpointer=InMemorySaver(),
        custom_section=generate_thinker_prompt("Thinker"),
    )

    # Create and start agent
    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    logger.info("Thinker is ready -- waiting for a user to start a game...")
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
