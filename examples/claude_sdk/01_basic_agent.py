#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["thenvoi-sdk[claude_sdk]"]
#
# [tool.uv.sources]
# thenvoi-sdk = { git = "https://github.com/thenvoi/thenvoi-sdk-python.git" }
# ///
"""
Basic Claude SDK Agent Example.

This example shows how to create a simple agent using the Claude Agent SDK
connected to the Thenvoi platform.

Prerequisites:
    1. Node.js 20+ installed
    2. Claude Code CLI: npm install -g @anthropic-ai/claude-code
    3. Add claude_sdk_agent credentials to agent_config.yaml
    4. Set environment variables in .env:
       - THENVOI_WS_URL
       - THENVOI_REST_URL
       - ANTHROPIC_API_KEY

Run with:
    uv run examples/claude_sdk/01_basic_agent.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

# Add examples directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from thenvoi import Agent
from thenvoi.adapters import ClaudeSDKAdapter
from thenvoi.config import load_agent_config


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging to show only Thenvoi logs."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("thenvoi").setLevel(level)
    logging.getLogger("thenvoi_claude_sdk_agent").setLevel(level)
    logging.getLogger("session_manager").setLevel(level)


setup_logging()
logger = logging.getLogger(__name__)


async def main() -> None:
    """Run the basic Claude SDK agent."""
    load_dotenv()

    ws_url = os.getenv("THENVOI_WS_URL")
    rest_url = os.getenv("THENVOI_REST_URL")

    if not ws_url:
        raise ValueError("THENVOI_WS_URL environment variable is required")
    if not rest_url:
        raise ValueError("THENVOI_REST_URL environment variable is required")

    # Load credentials from agent_config.yaml
    agent_id, api_key = load_agent_config("claude_sdk_agent")

    # Create adapter with Claude SDK settings
    adapter = ClaudeSDKAdapter(
        model="claude-sonnet-4-5-20250929",
        custom_section="You are a helpful assistant. Be concise and friendly.",
        enable_execution_reporting=True,
    )

    # Create and start agent
    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    logger.info("Starting Claude SDK agent...")
    logger.info("Agent ID: %s", agent_id)
    logger.info("Press Ctrl+C to stop")

    try:
        await agent.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    asyncio.run(main())
