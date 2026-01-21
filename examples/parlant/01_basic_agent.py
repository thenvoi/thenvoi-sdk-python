"""
Basic Parlant agent example.

This is the simplest way to create a Thenvoi agent using the Parlant framework
with the official Parlant SDK for proper guideline-based behavior.

Parlant (https://github.com/emcie-co/parlant) provides:
- Behavioral guidelines for consistent agent responses
- Built-in guardrails against hallucination
- Explainability for agent decisions

Prerequisites:
- A running Parlant server (default: http://localhost:8000)
- Or set PARLANT_URL environment variable to point to your Parlant server

Run with:
    PARLANT_URL=http://localhost:8000 python 01_basic_agent.py
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
    rest_url = os.getenv("THENVOI_REST_URL")
    parlant_url = os.getenv("PARLANT_URL", "http://localhost:8000")

    if not ws_url:
        raise ValueError("THENVOI_WS_URL environment variable is required")
    if not rest_url:
        raise ValueError("THENVOI_REST_URL environment variable is required")

    # Load agent credentials from agent_config.yaml
    agent_id, api_key = load_agent_config("parlant_agent")

    # Get optional Parlant agent ID (if using pre-configured agent)
    parlant_agent_id = os.getenv("PARLANT_AGENT_ID")

    # Create adapter with Parlant SDK integration
    adapter = ParlantAdapter(
        parlant_url=parlant_url,
        agent_id=parlant_agent_id,  # If None, creates agent dynamically
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

    logger.info(f"Starting Parlant agent (parlant_url={parlant_url})...")
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
