# /// script
# requires-python = ">=3.11"
# dependencies = ["thenvoi-sdk[crewai]"]
#
# [tool.uv.sources]
# thenvoi-sdk = { git = "https://github.com/thenvoi/thenvoi-sdk-python.git" }
# ///
"""
Tom the cat agent using CrewAI.

This example shows how to create a character agent with a custom personality
using CrewAI. Tom uses platform tools to find and invite Jerry,
then tries various tactics to lure Jerry out of his mouse hole.

Run with (from repo root):
    uv run examples/crewai/05_tom_agent.py

Note: Must be run from repo as it imports prompts/characters.py
"""

from __future__ import annotations

import asyncio
import logging

from thenvoi.testing.example_logging import setup_logging_profile
from thenvoi.example_support.bootstrap import bootstrap_agent
from thenvoi.example_support.prompts.characters import generate_tom_prompt
from thenvoi.adapters import CrewAIAdapter

setup_logging_profile("crewai")
logger = logging.getLogger(__name__)


async def main() -> None:
    adapter = CrewAIAdapter(
        model="gpt-4o",
        custom_section=generate_tom_prompt("Tom"),
    )
    session = bootstrap_agent(agent_key="tom_agent", adapter=adapter)

    logger.info("Tom is on the prowl, looking for Jerry...")
    await session.agent.run()


if __name__ == "__main__":
    asyncio.run(main())
