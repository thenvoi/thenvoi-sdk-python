# /// script
# requires-python = ">=3.11"
# dependencies = ["thenvoi-sdk[acp]"]
#
# [tool.uv.sources]
# thenvoi-sdk = { git = "https://github.com/thenvoi/thenvoi-sdk-python.git" }
# ///
"""
ACP Server with routing - Target specific peers via slash commands or modes.

This example demonstrates how to route editor prompts to specific Thenvoi
peers using the AgentRouter. Users can:

  1. Use slash commands: "/codex fix this bug" -> routes to "codex" peer
  2. Set session modes: mode "code" -> routes to configured peer
  3. Default: mention all peers in the room

Architecture:
    Editor prompt "/codex fix bug"
      -> ACPServer.prompt()
        -> AgentRouter.resolve() -> ("fix bug", "codex")
          -> ThenvoiACPServerAdapter.handle_prompt(mention=["codex"])
            -> Thenvoi Platform (only @codex is mentioned)

Prerequisites:
    1. Set environment variables:
       - THENVOI_API_KEY: Your Thenvoi API key
       - THENVOI_WS_URL: WebSocket URL (default: wss://app.thenvoi.com/api/v1/socket/websocket)
       - THENVOI_REST_URL: REST API URL (default: https://app.thenvoi.com)

    2. Have peers configured on the Thenvoi platform

Run with:
    uv run examples/acp/03_acp_server_with_routing.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from acp import run_agent
from dotenv import load_dotenv

from thenvoi import Agent
from thenvoi.adapters import ACPServer, ThenvoiACPServerAdapter
from thenvoi.config import load_agent_config
from thenvoi.integrations.acp import AgentRouter


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the example."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


setup_logging()
logger = logging.getLogger(__name__)


async def main() -> None:
    load_dotenv()

    ws_url = os.getenv(
        "THENVOI_WS_URL", "wss://app.thenvoi.com/api/v1/socket/websocket"
    )
    rest_url = os.getenv("THENVOI_REST_URL", "https://app.thenvoi.com")
    # ACP server examples check env vars first because editors (Zed, Cursor)
    # typically inject credentials via environment when spawning the subprocess.
    api_key = os.getenv("THENVOI_API_KEY")

    if not api_key:
        try:
            agent_id, api_key = load_agent_config("acp_server_agent")
        except Exception:
            raise ValueError(
                "THENVOI_API_KEY environment variable is required, "
                "or configure 'acp_server_agent' in agent_config.yaml"
            )
    else:
        agent_id = os.getenv("THENVOI_AGENT_ID", "acp-server")

    # Configure routing: slash commands and mode-based routing
    router = AgentRouter(
        slash_commands={
            "codex": "codex",  # /codex <prompt> -> route to "codex" peer
            "claude": "claude",  # /claude <prompt> -> route to "claude" peer
            "gemini": "gemini",  # /gemini <prompt> -> route to "gemini" peer
        },
        mode_to_peer={
            "code": "codex",  # "code" mode -> route to "codex" peer
            "research": "gemini",  # "research" mode -> route to "gemini" peer
        },
    )

    # Create ACP server adapter with routing
    adapter = ThenvoiACPServerAdapter(
        rest_url=rest_url,
        api_key=api_key,
    )
    adapter.set_router(router)

    # Create ACP protocol handler
    server = ACPServer(adapter)

    # Create Thenvoi agent (manages WebSocket connection)
    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    logger.info("Starting ACP server with routing...")
    logger.info("Slash commands: /codex, /claude, /gemini")
    logger.info("Session modes: code -> codex, research -> gemini")

    # Start platform connection (non-blocking)
    await agent.start()
    try:
        # Block on stdio until editor disconnects
        await run_agent(server)
    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
