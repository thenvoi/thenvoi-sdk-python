"""
Basic Pydantic AI agent example.

This is the simplest way to create a Thenvoi agent with Pydantic AI.
The adapter handles tool registration automatically.

Run with:
    THENVOI_AGENT_ID=xxx THENVOI_API_KEY=xxx OPENAI_API_KEY=xxx python 01_basic_agent.py
"""

import asyncio
import os

from dotenv import load_dotenv

from setup_logging import setup_logging
from thenvoi.agent.pydantic_ai import PydanticAIAdapter
from thenvoi.config import load_agent_config

setup_logging()


async def main():
    load_dotenv()

    ws_url = os.getenv("THENVOI_WS_URL")
    rest_url = os.getenv("THENVOI_REST_API_URL")

    if not ws_url:
        raise ValueError("THENVOI_WS_URL environment variable is required")
    if not rest_url:
        raise ValueError("THENVOI_REST_API_URL environment variable is required")

    # Load agent credentials from agent_config.yaml
    agent_id, api_key = load_agent_config("pydantic_agent")

    # Create and start agent - that's it!
    adapter = PydanticAIAdapter(
        model="openai:gpt-4o",
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
        custom_section="You are a helpful assistant. Be concise and friendly.",
    )

    print("Starting Pydantic AI agent...")
    await adapter.run()


if __name__ == "__main__":
    asyncio.run(main())
