#!/usr/bin/env python3
"""
Jerry the mouse agent using Claude SDK.

This example shows how to create a character agent with a custom personality
using the Claude Agent SDK. Jerry is a clever mouse who lives in his hole
and teases Tom the cat while staying safe from being caught.

Prerequisites:
    1. Node.js 20+ installed
    2. Claude Code CLI: npm install -g @anthropic-ai/claude-code
    3. Environment variables set:
       - THENVOI_AGENT_ID: Jerry's agent ID (540a6fcb-f570-4d40-b39b-6c26404bc3de)
       - THENVOI_API_KEY: Jerry's API key
       - ANTHROPIC_API_KEY: Your Anthropic API key

Usage:
    python 04_jerry_agent.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

# Add examples directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompts.characters import generate_jerry_prompt
from setup_logging import setup_logging
from thenvoi import Agent
from thenvoi.adapters import ClaudeSDKAdapter

setup_logging()
logger = logging.getLogger(__name__)


async def main():
    """Run Jerry the mouse agent."""

    # Get credentials from environment
    agent_id = os.environ.get("THENVOI_AGENT_ID", "")
    api_key = os.environ.get("THENVOI_API_KEY", "")

    if not agent_id or not api_key:
        logger.error("THENVOI_AGENT_ID and THENVOI_API_KEY must be set")
        sys.exit(1)

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
