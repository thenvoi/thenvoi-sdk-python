# /// script
# requires-python = ">=3.11"
# dependencies = ["thenvoi-sdk[pydantic-ai]"]
#
# [tool.uv.sources]
# thenvoi-sdk = { git = "https://github.com/thenvoi/thenvoi-sdk-python.git" }
# ///
"""
Jerry the mouse agent using Pydantic AI.

This example shows how to create a character agent with a custom personality
using Pydantic AI. Jerry is a clever mouse who lives in his hole
and teases Tom the cat while staying safe from being caught.

Run with (from repo root):
    uv run examples/pydantic_ai/04_jerry_agent.py

Note: Must be run from repo as it imports prompts/characters.py
"""

from __future__ import annotations

import asyncio
import logging

from thenvoi.example_support.bootstrap import bootstrap_agent
from thenvoi.example_support.prompts.characters import generate_jerry_prompt
from thenvoi.testing.example_logging import setup_logging_profile
from thenvoi.adapters import PydanticAIAdapter

setup_logging_profile("pydantic_ai")
logger = logging.getLogger(__name__)


async def main() -> None:
    adapter = PydanticAIAdapter(
        model="openai:gpt-4o",
        custom_section=generate_jerry_prompt("Jerry"),
    )
    session = bootstrap_agent(agent_key="jerry_agent", adapter=adapter)

    logger.info("Jerry is cozy in his hole, watching for Tom...")
    await session.agent.run()


if __name__ == "__main__":
    asyncio.run(main())
