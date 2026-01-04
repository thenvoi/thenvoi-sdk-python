"""
Basic CrewAI agent example.

This is the simplest way to create a Thenvoi agent using the CrewAI framework.
The adapter handles conversation history, tool calling, and platform integration.

CrewAI (https://docs.crewai.com/) provides:
- Agent collaboration with defined roles and goals
- Task orchestration with processes
- Memory and knowledge management

Run with:
    OPENAI_API_KEY=xxx python 01_basic_agent.py
"""

import asyncio
import logging
import os

from dotenv import load_dotenv

from setup_logging import setup_logging
from thenvoi import Agent
from thenvoi.adapters import CrewAIAdapter
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
    agent_id, api_key = load_agent_config("crewai_agent")

    # Create adapter with framework-specific settings
    adapter = CrewAIAdapter(
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

    logger.info("Starting CrewAI agent...")
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
