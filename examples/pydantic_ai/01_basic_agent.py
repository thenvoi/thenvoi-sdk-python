# /// script
# requires-python = ">=3.11"
# dependencies = ["thenvoi-sdk[pydantic-ai]"]
#
# [tool.uv.sources]
# thenvoi-sdk = { git = "https://github.com/thenvoi/thenvoi-sdk-python.git" }
# ///
"""
Basic Pydantic AI agent example.

This is the simplest way to create a Thenvoi agent with Pydantic AI.
The adapter handles tool registration automatically.

Run with:
    uv run examples/pydantic_ai/01_basic_agent.py
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
from thenvoi.adapters import PydanticAIAdapter

setup_logging_profile("pydantic_ai")
logger = logging.getLogger(__name__)


async def main() -> None:
    session = bootstrap_agent(
        agent_key=basic_agent_key("pydantic_ai"),
        adapter=PydanticAIAdapter(**basic_adapter_kwargs("pydantic_ai")),
    )

    logger.info(basic_startup_message("pydantic_ai"))
    await session.agent.run()


if __name__ == "__main__":
    asyncio.run(main())
