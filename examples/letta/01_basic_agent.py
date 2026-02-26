# /// script
# requires-python = ">=3.11"
# dependencies = ["thenvoi-sdk[letta]"]
#
# [tool.uv.sources]
# thenvoi-sdk = { git = "https://github.com/thenvoi/thenvoi-sdk-python.git" }
# ///
"""
Basic Letta agent example.

Connects a Letta agent to the Thenvoi platform using MCP tools for
bidirectional communication.  Requires a running self-hosted Letta
server and a thenvoi-mcp server.

Prerequisites:
    1. Start Letta server: docker run -p 8283:8283 letta/letta:latest
    2. Start thenvoi-mcp:  See examples/letta/docker-compose.yml
    3. Set environment variables in .env or export them

Run with:
    uv run examples/letta/01_basic_agent.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from setup_logging import setup_logging

from thenvoi import Agent
from thenvoi.adapters.letta import LettaAdapter, LettaAdapterConfig
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

    # Load agent credentials from agent_config.yaml
    agent_id, api_key = load_agent_config("letta_agent")

    # Create adapter with Letta-specific settings
    adapter = LettaAdapter(
        config=LettaAdapterConfig(
            # Self-hosted Letta server (default)
            base_url=os.getenv("LETTA_BASE_URL", "http://localhost:8283"),
            # Optional — not required for self-hosted Letta
            api_key=os.getenv("LETTA_API_KEY"),
            # LLM model to use
            model=os.getenv("LETTA_MODEL", "openai/gpt-4o"),
            # thenvoi-mcp server for platform tool execution
            mcp_server_url=os.getenv("MCP_SERVER_URL", "http://localhost:8002/sse"),
            # Custom prompt section
            custom_section="You are a helpful assistant. Be concise and friendly.",
        ),
    )

    # Create and start agent
    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    logger.info("Starting Letta agent...")
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
