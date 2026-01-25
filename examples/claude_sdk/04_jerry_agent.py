"""
Jerry the mouse agent using Claude SDK.

This example shows how to create a character agent with a custom personality
using the Claude Agent SDK. Jerry is a clever mouse who lives in his hole
and teases Tom the cat while staying safe from being caught.

Prerequisites:
    - Node.js 20+ installed
    - Claude Code CLI: npm install -g @anthropic-ai/claude-code
    - Add jerry_agent credentials to agent_config.yaml
    - Tom agent should be online for full interaction

Run with:
    ANTHROPIC_API_KEY=xxx python 04_jerry_agent.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompts.characters import generate_jerry_prompt
from setup_logging import setup_logging
from thenvoi import Agent
from thenvoi.adapters import ClaudeSDKAdapter
from thenvoi.config import load_agent_config

setup_logging()
logger = logging.getLogger(__name__)


async def main() -> None:
    """Run Jerry the mouse agent."""
    load_dotenv()

    ws_url = os.getenv("THENVOI_WS_URL")
    rest_url = os.getenv("THENVOI_REST_URL")

    if not ws_url:
        raise ValueError("THENVOI_WS_URL environment variable is required")
    if not rest_url:
        raise ValueError("THENVOI_REST_URL environment variable is required")

    # Load Jerry's credentials from agent_config.yaml
    agent_id, api_key = load_agent_config("jerry_agent")

    # Create adapter with Jerry's character prompt
    adapter = ClaudeSDKAdapter(
        model="claude-sonnet-4-5-20250929",
        custom_section=generate_jerry_prompt("Jerry"),
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

    logger.info("Jerry is cozy in his hole, watching for Tom...")
    logger.info(f"Agent ID: {agent_id}")
    logger.info("Press Ctrl+C to stop")

    try:
        await agent.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    asyncio.run(main())
