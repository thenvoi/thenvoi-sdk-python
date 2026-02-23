# /// script
# requires-python = ">=3.11"
# dependencies = ["thenvoi-sdk[langgraph]"]
#
# [tool.uv.sources]
# thenvoi-sdk = { git = "https://github.com/thenvoi/thenvoi-sdk-python.git" }
# ///
"""
Guesser agent for the 20 Questions arena game.

The Guesser is invited into a room by the Thinker, reads the category
announcement, and asks strategic yes/no questions to deduce the secret
word within 20 rounds.

Supports both OpenAI and Anthropic LLMs -- set either OPENAI_API_KEY or
ANTHROPIC_API_KEY in your environment.

Run with (from repo root):
    uv run examples/arena/guesser_agent.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from arena.prompts import generate_guesser_prompt

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

    # Load Guesser's credentials from agent_config.yaml
    agent_id, api_key = load_agent_config("arena_guesser")

    # Guesser uses GPT-5.2
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-5.2")

    # Create adapter with Guesser's game prompt
    adapter = LangGraphAdapter(
        llm=llm,
        checkpointer=InMemorySaver(),
        custom_section=generate_guesser_prompt("Guesser"),
    )

    # Create and start agent
    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    logger.info("Guesser is ready -- waiting to be invited to a game...")
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
