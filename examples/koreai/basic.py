# /// script
# requires-python = ">=3.11"
# dependencies = ["thenvoi-sdk[koreai]", "python-dotenv"]
#
# [tool.uv.sources]
# thenvoi-sdk = { git = "https://github.com/thenvoi/thenvoi-sdk-python.git" }
# ///
"""
Connect a Kore.ai XO Platform bot to Thenvoi.

The adapter bridges Thenvoi's WebSocket-based agent model and Kore.ai's
HTTP webhook model. It runs a callback server to receive bot responses
and forwards them to the appropriate Thenvoi chat room.

Prerequisites:
    1. Configure a Webhook V2 channel in Kore.ai (async mode)
    2. Point the POST_URL to this adapter's callback URL
    3. Set agent credentials in agent_config.yaml

Run with:
    uv run examples/koreai/basic.py
"""

from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv

from thenvoi import Agent
from thenvoi.adapters.koreai import KoreAIAdapter, KoreAIConfig
from thenvoi.config import load_agent_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    """Run the Kore.ai adapter."""
    load_dotenv()

    agent_cfg = load_agent_config("koreai_bot")

    ws_url = os.environ.get("THENVOI_WS_URL")
    rest_url = os.environ.get("THENVOI_REST_URL")
    if not ws_url:
        raise ValueError("THENVOI_WS_URL environment variable is required")
    if not rest_url:
        raise ValueError("THENVOI_REST_URL environment variable is required")

    bot_id = os.environ.get("KOREAI_BOT_ID")
    client_id = os.environ.get("KOREAI_CLIENT_ID")
    client_secret = os.environ.get("KOREAI_CLIENT_SECRET")
    callback_url = os.environ.get("KOREAI_CALLBACK_URL")

    if not bot_id:
        raise ValueError("KOREAI_BOT_ID environment variable is required")
    if not client_id:
        raise ValueError("KOREAI_CLIENT_ID environment variable is required")
    if not client_secret:
        raise ValueError("KOREAI_CLIENT_SECRET environment variable is required")
    if not callback_url:
        raise ValueError("KOREAI_CALLBACK_URL environment variable is required")

    koreai_config = KoreAIConfig(
        bot_id=bot_id,
        client_id=client_id,
        client_secret=client_secret,
        callback_url=callback_url,
        api_host=os.environ.get("KOREAI_API_HOST", "https://bots.kore.ai"),
        webhook_secret=os.environ.get("KOREAI_WEBHOOK_SECRET"),
    )

    adapter = KoreAIAdapter(config=koreai_config)

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_cfg["agent_id"],
        api_key=agent_cfg["api_key"],
        ws_url=ws_url,
        rest_url=rest_url,
    )

    logger.info("Starting Kore.ai adapter for bot %s", bot_id)
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
