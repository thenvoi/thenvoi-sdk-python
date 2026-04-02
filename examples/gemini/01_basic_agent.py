# /// script
# requires-python = ">=3.11"
# dependencies = ["thenvoi-sdk[gemini]"]
#
# [tool.uv.sources]
# thenvoi-sdk = { git = "https://github.com/thenvoi/thenvoi-sdk-python.git" }
# ///
"""
Basic Gemini agent example.

This is the simplest way to create a Thenvoi agent with the Gemini SDK.
The adapter handles tool registration and function-calling loops automatically.

Requires:
    - agent_config.yaml with gemini_agent credentials
    - GEMINI_API_KEY environment variable

Run with:
    uv run examples/gemini/01_basic_agent.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from thenvoi import Agent
from thenvoi.adapters import GeminiAdapter
from thenvoi.config import load_agent_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    # Load agent credentials from agent_config.yaml
    agent_id, api_key = load_agent_config("gemini_agent")

    # Create adapter with Gemini settings
    # Requires GEMINI_API_KEY environment variable or pass gemini_api_key explicitly
    adapter = GeminiAdapter(
        model="gemini-2.5-flash",
        custom_section="You are a helpful assistant. Be concise and friendly.",
    )

    # Create and start agent
    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
    )

    logger.info("Starting Gemini agent...")
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
