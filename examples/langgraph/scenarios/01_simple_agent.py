# /// script
# requires-python = ">=3.11"
# dependencies = ["thenvoi-sdk[langgraph]"]
#
# [tool.uv.sources]
# thenvoi-sdk = { git = "https://github.com/thenvoi/thenvoi-sdk-python.git" }
# ///
"""
Simple LangGraph agent example using the composition API.

This is the simplest way to create a Thenvoi agent - just provide
the LLM and checkpointer, and the adapter handles everything.

Run with:
    uv run examples/langgraph/scenarios/01_simple_agent.py
"""

from __future__ import annotations

import asyncio
import logging

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from thenvoi.example_support.bootstrap import bootstrap_agent
from thenvoi.example_support.scenarios import basic_agent_key, basic_startup_message
from thenvoi.testing.example_logging import setup_logging_profile
from thenvoi.adapters import LangGraphAdapter

setup_logging_profile("langgraph")
logger = logging.getLogger(__name__)


async def main() -> None:
    session = bootstrap_agent(
        agent_key=basic_agent_key("langgraph"),
        adapter=LangGraphAdapter(
            llm=ChatOpenAI(model="gpt-4o"),
            checkpointer=InMemorySaver(),
        ),
    )

    logger.info(basic_startup_message("langgraph"))
    await session.agent.run()


if __name__ == "__main__":
    asyncio.run(main())
