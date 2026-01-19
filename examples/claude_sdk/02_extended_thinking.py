#!/usr/bin/env python3
"""
Extended Thinking Claude SDK Agent Example.

This example shows how to create an agent with extended thinking enabled,
which allows Claude to reason through complex problems step-by-step.

Extended thinking is useful for:
- Complex problem solving
- Multi-step reasoning
- Code analysis and debugging
- Planning and decision making

Prerequisites:
    1. Node.js 20+ installed
    2. Claude Code CLI: npm install -g @anthropic-ai/claude-code
    3. Environment variables set:
       - THENVOI_AGENT_ID: Your agent ID
       - THENVOI_API_KEY: Your API key
       - ANTHROPIC_API_KEY: Your Anthropic API key

Usage:
    python 02_extended_thinking.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

# Add examples directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from setup_logging import setup_logging
from thenvoi import Agent
from thenvoi.adapters import ClaudeSDKAdapter

setup_logging()
logger = logging.getLogger(__name__)


async def main():
    """Run the extended thinking Claude SDK agent."""

    # Get credentials from environment
    agent_id = os.environ.get("THENVOI_AGENT_ID", "")
    api_key = os.environ.get("THENVOI_API_KEY", "")

    if not agent_id or not api_key:
        logger.error("THENVOI_AGENT_ID and THENVOI_API_KEY must be set")
        sys.exit(1)

    # Create adapter with extended thinking enabled
    adapter = ClaudeSDKAdapter(
        model="claude-sonnet-4-5-20250929",
        custom_section="""You are a thoughtful AI assistant that excels at
complex problem-solving. When faced with challenging questions:
1. Break down the problem into smaller parts
2. Consider multiple approaches
3. Evaluate trade-offs
4. Provide clear, well-reasoned answers""",
        max_thinking_tokens=10000,  # Enable extended thinking
        enable_execution_reporting=True,  # Report thinking as events
    )

    # Create and start agent
    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
    )

    logger.info("Starting Claude SDK agent with extended thinking...")
    logger.info(f"Agent ID: {agent_id}")
    logger.info("Max thinking tokens: 10000")
    logger.info("Press Ctrl+C to stop")

    try:
        await agent.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    asyncio.run(main())
