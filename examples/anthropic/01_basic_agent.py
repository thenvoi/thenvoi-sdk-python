"""
Basic Anthropic SDK agent example.

This is the simplest way to create a Thenvoi agent using the Anthropic Python SDK.
The adapter handles conversation history, tool calling, and platform integration.

Run with:
    ANTHROPIC_API_KEY=xxx python 01_basic_agent.py
"""

import asyncio
import logging
import os

from dotenv import load_dotenv

from setup_logging import setup_logging
from thenvoi import Agent
from thenvoi.adapters import AnthropicAdapter
from thenvoi.config import load_agent_config

setup_logging()
logger = logging.getLogger(__name__)


async def main():
    load_dotenv()

    ws_url = os.getenv("THENVOI_WS_URL")
    rest_url = os.getenv("THENVOI_REST_URL")

    if not ws_url:
        raise ValueError("THENVOI_WS_URL environment variable is required")
    if not rest_url:
        raise ValueError("THENVOI_REST_URL environment variable is required")

    # Load agent credentials from agent_config.yaml
    agent_id, api_key = load_agent_config("anthropic_agent")

    # Create adapter with framework-specific settings
    adapter = AnthropicAdapter(
        model="claude-sonnet-4-5-20250929",
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

    logger.info("Starting Anthropic agent...")
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
