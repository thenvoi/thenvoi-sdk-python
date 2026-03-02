# /// script
# requires-python = ">=3.11"
# dependencies = ["thenvoi-sdk[claude_sdk]"]
#
# [tool.uv.sources]
# thenvoi-sdk = { git = "https://github.com/thenvoi/thenvoi-sdk-python.git" }
# ///
"""
Tom the cat agent using Claude SDK.

This example shows how to create a character agent with a custom personality
using the Claude Agent SDK. Tom uses platform tools to find and invite Jerry,
then tries various tactics to lure Jerry out of his mouse hole.

Prerequisites:
    - Node.js 20+ installed
    - Claude Code CLI: npm install -g @anthropic-ai/claude-code
    - Add tom_agent credentials to agent_config.yaml
    - Jerry agent should be online for full interaction

Run with (from repo root):
    uv run examples/claude_sdk/03_tom_agent.py

Note: Must be run from repo as it imports prompts/characters.py
"""

from __future__ import annotations

import asyncio
import logging

from thenvoi.testing.example_logging import setup_logging_profile
from thenvoi.example_support.bootstrap import bootstrap_agent
from thenvoi.example_support.prompts.characters import generate_tom_prompt
from thenvoi.adapters import ClaudeSDKAdapter

setup_logging_profile("claude_sdk")
logger = logging.getLogger(__name__)


async def main() -> None:
    """Run Tom the cat agent."""
    adapter = ClaudeSDKAdapter(
        model="claude-sonnet-4-5-20250929",
        custom_section=generate_tom_prompt("Tom"),
        enable_execution_reporting=True,
    )
    session = bootstrap_agent(agent_key="tom_agent", adapter=adapter)

    logger.info("Tom is on the prowl, looking for Jerry...")
    logger.info("Agent ID: %s", session.runtime.agent_id)
    logger.info("Press Ctrl+C to stop")
    await session.agent.run()


if __name__ == "__main__":
    asyncio.run(main())
