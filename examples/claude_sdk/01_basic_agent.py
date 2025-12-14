#!/usr/bin/env python3
"""
Basic Claude SDK Agent Example.

This example shows how to create a simple agent using the Claude Agent SDK
connected to the Thenvoi platform.

Prerequisites:
    1. Node.js 20+ installed
    2. Claude Code CLI: npm install -g @anthropic-ai/claude-code
    3. Environment variables set:
       - THENVOI_AGENT_ID: Your agent ID
       - THENVOI_API_KEY: Your API key
       - ANTHROPIC_API_KEY: Your Anthropic API key

Usage:
    python 01_basic_agent.py
"""

from __future__ import annotations

import asyncio
import os
import sys

# Add examples directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from setup_logging import setup_logging
from thenvoi_claude_sdk_agent import ThenvoiClaudeSDKAgent


async def main():
    """Run the basic Claude SDK agent."""
    setup_logging()

    # Get credentials from environment
    agent_id = os.environ.get("THENVOI_AGENT_ID", "")
    api_key = os.environ.get("THENVOI_API_KEY", "")

    if not agent_id or not api_key:
        print("Error: THENVOI_AGENT_ID and THENVOI_API_KEY must be set")
        sys.exit(1)

    # Create agent with basic configuration
    agent = ThenvoiClaudeSDKAgent(
        model="claude-sonnet-4-5-20250929",
        agent_id=agent_id,
        api_key=api_key,
        custom_section="You are a helpful assistant. Be concise and friendly.",
        enable_execution_reporting=True,
    )

    print("Starting Claude SDK agent...")
    print(f"Agent ID: {agent_id}")
    print("Press Ctrl+C to stop")

    try:
        await agent.run()
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    asyncio.run(main())
