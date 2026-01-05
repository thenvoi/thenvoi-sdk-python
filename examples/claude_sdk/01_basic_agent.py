#!/usr/bin/env python3
"""
Basic Claude SDK Agent Example.

This example shows how to create a simple agent using the Claude Agent SDK
connected to the Thenvoi platform.

Prerequisites:
    1. Node.js 20+ installed
    2. Claude Code CLI: npm install -g @anthropic-ai/claude-code
    3. Environment variables set:
       - ANTHROPIC_API_KEY: Your Anthropic API key
    4. Agent config: agent_config.yaml with claude_sdk_basic_agent entry

Usage:
    # Using config file (recommended for Docker):
    python 01_basic_agent.py

    # Using environment variables:
    THENVOI_AGENT_ID=xxx THENVOI_API_KEY=yyy python 01_basic_agent.py
"""

from __future__ import annotations

import asyncio
import os
import sys

# Add examples directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from setup_logging import setup_logging
from thenvoi import Agent
from thenvoi.adapters import ClaudeSDKAdapter
from thenvoi.config import load_agent_config


async def main():
    """Run the basic Claude SDK agent."""
    setup_logging()

    # Try loading from config file first, fall back to environment variables
    agent_id = os.environ.get("THENVOI_AGENT_ID", "")
    api_key = os.environ.get("THENVOI_API_KEY", "")

    if not agent_id or not api_key:
        try:
            agent_id, api_key = load_agent_config("claude_sdk_basic_agent")
            print("Loaded credentials from agent_config.yaml")
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            print("Set THENVOI_AGENT_ID and THENVOI_API_KEY environment variables,")
            print("or configure claude_sdk_basic_agent in agent_config.yaml")
            sys.exit(1)

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
        rest_url=os.environ.get("THENVOI_REST_API_URL"),
        ws_url=os.environ.get("THENVOI_WS_URL"),
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
