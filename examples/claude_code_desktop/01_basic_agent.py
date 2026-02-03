#!/usr/bin/env python3
"""
Basic Claude Code Desktop Agent Example.

This example shows how to create a simple agent using your local Claude Code CLI
connected to the Thenvoi platform WITHOUT requiring an Anthropic API key.

This is ideal for users who want to use their existing Claude Code installation
(including access to local files, MCP tools, and extended thinking) as a Thenvoi
agent, without consuming usage-based API credits.

Prerequisites:
    1. Node.js 20+ installed
    2. Claude Code CLI: npm install -g @anthropic-ai/claude-code
    3. Claude Code authenticated (run `claude` once to set up)
    4. Environment variables set (see .env.example):
       - THENVOI_AGENT_ID: Your agent ID from Thenvoi dashboard
       - THENVOI_API_KEY: Your API key from Thenvoi dashboard
       - THENVOI_REST_URL: Thenvoi REST API URL
       - THENVOI_WS_URL: Thenvoi WebSocket URL

Optional environment variables:
    - CLAUDE_CODE_PATH: Path to Claude CLI (if not in PATH)

How it works:
    - Invokes Claude Code CLI as subprocess with --print --output-format json
    - Supports session persistence for multi-turn conversations per room
    - Claude responds with structured JSON actions to interact with Thenvoi

Usage:
    cp .env.example .env
    # Edit .env with your Thenvoi credentials
    python 01_basic_agent.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

# Add examples directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env file from the same directory as this script
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from setup_logging import setup_logging
from thenvoi import Agent
from thenvoi.adapters import ClaudeCodeDesktopAdapter

setup_logging()
logger = logging.getLogger(__name__)


async def main():
    """Run the basic Claude Code Desktop agent."""

    # Get credentials from environment
    agent_id = os.environ.get("THENVOI_AGENT_ID", "")
    api_key = os.environ.get("THENVOI_API_KEY", "")
    ws_url = os.environ.get("THENVOI_WS_URL")
    rest_url = os.environ.get("THENVOI_REST_URL")

    if not agent_id or not api_key:
        logger.error("THENVOI_AGENT_ID and THENVOI_API_KEY must be set")
        sys.exit(1)

    if not ws_url or not rest_url:
        logger.error("THENVOI_WS_URL and THENVOI_REST_URL must be set")
        sys.exit(1)

    # Create adapter with Claude Code Desktop settings
    adapter = ClaudeCodeDesktopAdapter(
        custom_section="You are a helpful assistant. Be concise and friendly.",
        # Optionally set CLI path if not in PATH:
        # cli_path="/path/to/claude",
        # Optionally set timeout (default: 2 minutes):
        # cli_timeout=180000,  # 3 minutes
    )

    # Create and start agent
    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    logger.info("Starting Claude Code Desktop agent...")
    logger.info(f"Agent ID: {agent_id}")
    logger.info("Press Ctrl+C to stop")

    try:
        await agent.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    asyncio.run(main())
