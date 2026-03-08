"""Run test-agent2 as a live agent on the platform.

Usage:
    uv run python scripts/run_agent2.py
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from thenvoi.adapters.langgraph import LangGraphAdapter
from thenvoi.agent import Agent

load_dotenv(Path(__file__).parent.parent / ".env.test", override=False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    api_key = os.environ.get("THENVOI_API_KEY_2")
    agent_id = os.environ.get("TEST_AGENT_ID_2")
    ws_url = os.environ.get(
        "THENVOI_WS_URL", "wss://app.thenvoi.com/api/v1/socket/websocket"
    )
    rest_url = os.environ.get("THENVOI_REST_URL", "https://app.thenvoi.com")

    if not api_key or not agent_id:
        raise ValueError(
            "THENVOI_API_KEY_2 and TEST_AGENT_ID_2 must be set in .env.test"
        )

    adapter = LangGraphAdapter(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        checkpointer=MemorySaver(),
        custom_section="Keep responses short and concise.",
    )
    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    logger.info("Starting test-agent2 (id=%s)...", agent_id)
    async with agent:
        logger.info("test-agent2 is running. Press Ctrl+C to stop.")
        await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("test-agent2 stopped.")
