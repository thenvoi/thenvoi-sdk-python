# /// script
# requires-python = ">=3.11"
# dependencies = ["thenvoi-sdk[crewai]"]
#
# [tool.uv.sources]
# thenvoi-sdk = { git = "https://github.com/thenvoi/thenvoi-sdk-python.git" }
# ///
"""
Basic CrewAI agent example.

This is the simplest way to create a Thenvoi agent using the CrewAI framework.
The adapter handles conversation history, tool calling, and platform integration.

CrewAI (https://docs.crewai.com/) provides:
- Agent collaboration with defined roles and goals
- Task orchestration with processes
- Memory and knowledge management

Run with:
    uv run examples/crewai/01_basic_agent.py
"""

from __future__ import annotations

import asyncio
import logging

from thenvoi.example_support.bootstrap import bootstrap_agent
from thenvoi.example_support.scenarios import (
    basic_adapter_kwargs,
    basic_agent_key,
    basic_startup_message,
)
from thenvoi.testing.example_logging import setup_logging_profile
from thenvoi.adapters import CrewAIAdapter

setup_logging_profile("crewai")
logger = logging.getLogger(__name__)


async def main() -> None:
    session = bootstrap_agent(
        agent_key=basic_agent_key("crewai"),
        adapter=CrewAIAdapter(**basic_adapter_kwargs("crewai")),
    )

    logger.info(basic_startup_message("crewai"))
    await session.agent.run()


if __name__ == "__main__":
    asyncio.run(main())
