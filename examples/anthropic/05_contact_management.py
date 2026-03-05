# /// script
# requires-python = ">=3.11"
# dependencies = ["thenvoi-sdk[anthropic]"]
#
# [tool.uv.sources]
# thenvoi-sdk = { git = "https://github.com/thenvoi/thenvoi-sdk-python.git" }
# ///
"""
Contact management example using the Anthropic adapter.

Demonstrates how to configure an agent with LLM-driven contact handling
via the HUB_ROOM strategy. Contact requests are routed to a dedicated hub
room where the LLM decides whether to approve or reject them. The agent
can also use contact tools (list, add, remove contacts and manage requests)
through normal LLM tool calling.

Run with:
    uv run examples/anthropic/05_contact_management.py
"""

from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv

from setup_logging import setup_logging
from thenvoi import Agent
from thenvoi.adapters import AnthropicAdapter
from thenvoi.config import load_agent_config
from thenvoi.runtime.types import ContactEventConfig, ContactEventStrategy

setup_logging()
logger = logging.getLogger(__name__)


async def main() -> None:
    load_dotenv()

    ws_url = os.getenv("THENVOI_WS_URL")
    rest_url = os.getenv("THENVOI_REST_URL")

    if not ws_url:
        raise ValueError("THENVOI_WS_URL environment variable is required")
    if not rest_url:
        raise ValueError("THENVOI_REST_URL environment variable is required")

    agent_id, api_key = load_agent_config("anthropic_agent")

    adapter = AnthropicAdapter(
        model="claude-sonnet-4-5-20250929",
        custom_section=(
            "You are a helpful assistant with contact management capabilities.\n"
            "You can list, add, and remove contacts, and manage contact requests.\n"
            "Contact requests are routed to you for review - decide whether to "
            "approve or reject each one based on context."
        ),
    )

    contact_config = ContactEventConfig(
        strategy=ContactEventStrategy.HUB_ROOM,
        broadcast_changes=True,
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
        contact_config=contact_config,
    )

    logger.info("Starting Anthropic agent with contact management...")
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
