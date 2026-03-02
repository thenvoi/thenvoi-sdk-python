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

from thenvoi.testing.example_logging import setup_logging_profile
from thenvoi.example_support.bootstrap import bootstrap_agent
from thenvoi.example_support.scenarios import (
    basic_adapter_kwargs,
    basic_agent_key,
    basic_startup_message,
)
from thenvoi.adapters import ClaudeSDKAdapter

setup_logging_profile("claude_sdk")
logger = logging.getLogger(__name__)


async def main() -> None:
    """Run the basic Claude SDK agent."""
    session = bootstrap_agent(
        agent_key=basic_agent_key("claude_sdk"),
        adapter=ClaudeSDKAdapter(**basic_adapter_kwargs("claude_sdk")),
    )

    logger.info(basic_startup_message("claude_sdk"))
    logger.info("Agent ID: %s", session.runtime.agent_id)
    logger.info("Press Ctrl+C to stop")
    await session.agent.run()


if __name__ == "__main__":
    asyncio.run(main())
