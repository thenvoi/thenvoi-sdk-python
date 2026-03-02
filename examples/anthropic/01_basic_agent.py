# /// script
# requires-python = ">=3.11"
# dependencies = ["thenvoi-sdk[anthropic]"]
#
# [tool.uv.sources]
# thenvoi-sdk = { git = "https://github.com/thenvoi/thenvoi-sdk-python.git" }
# ///
"""
Basic Anthropic SDK agent example.

This is the simplest way to create a Thenvoi agent using the Anthropic Python SDK.
The adapter handles conversation history, tool calling, and platform integration.

Run with:
    uv run examples/anthropic/01_basic_agent.py
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
from thenvoi.adapters import AnthropicAdapter

setup_logging_profile("anthropic")
logger = logging.getLogger(__name__)


async def main() -> None:
    session = bootstrap_agent(
        agent_key=basic_agent_key("anthropic"),
        adapter=AnthropicAdapter(**basic_adapter_kwargs("anthropic")),
    )

    logger.info(basic_startup_message("anthropic"))
    await session.agent.run()


if __name__ == "__main__":
    asyncio.run(main())
