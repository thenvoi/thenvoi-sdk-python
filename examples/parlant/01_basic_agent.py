"""
Basic Parlant agent example.

This is the simplest way to create a Thenvoi agent using the Parlant framework.
The adapter handles conversation history, tool calling, and platform integration.

Parlant (https://github.com/emcie-co/parlant) provides:
- Behavioral guidelines for consistent agent responses
- Built-in guardrails against hallucination
- Explainability for agent decisions

Run with:
    OPENAI_API_KEY=xxx python 01_basic_agent.py
"""

import asyncio
import logging
import os

from dotenv import load_dotenv

from setup_logging import setup_logging
from thenvoi import Agent
from thenvoi.adapters import ParlantAdapter
from thenvoi.config import load_agent_config

setup_logging()
logger = logging.getLogger(__name__)


async def main():
    load_dotenv()

    ws_url = os.getenv("THENVOI_WS_URL")
    rest_url = os.getenv("THENVOI_REST_API_URL")

    if not ws_url:
        raise ValueError("THENVOI_WS_URL environment variable is required")
    if not rest_url:
        raise ValueError("THENVOI_REST_API_URL environment variable is required")

    # Load agent credentials from agent_config.yaml
    agent_id, api_key = load_agent_config("parlant_agent")

    # Create adapter with framework-specific settings
    adapter = ParlantAdapter(
        model="gpt-4o",
        custom_section="You are a helpful assistant. Be concise and friendly.",
    )

    # Create and start agent
    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    logger.info("Starting Parlant agent...")
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())

