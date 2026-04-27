# /// script
# requires-python = ">=3.11"
# dependencies = ["thenvoi-sdk[acp]"]
#
# [tool.uv.sources]
# thenvoi-sdk = { git = "https://github.com/thenvoi/thenvoi-sdk-python.git" }
# ///
"""
Basic ACP Server example - Thenvoi as an ACP agent.

This example starts Thenvoi as an ACP agent that editors (Zed, Cursor,
JetBrains, Neovim) can connect to. It implements the "Super-Agent" pattern:
a single ACP facade that routes editor requests to multiple Thenvoi peers.

Architecture:
    Editor (Zed/Cursor/JetBrains/Neovim)
      -> ACP JSON-RPC over stdio
        -> ACPServer (protocol handler)
          -> ThenvoiACPServerAdapter (platform bridge)
            -> Thenvoi Platform (REST + WebSocket)
              -> Multi-agent responses via Phoenix Channels
            -> ACP session_update notifications back to editor

Prerequisites:
    1. Set environment variables:
       - THENVOI_API_KEY: Your Thenvoi API key
       - THENVOI_WS_URL: WebSocket URL (default: wss://app.band.ai/api/v1/socket/websocket)
       - THENVOI_REST_URL: REST API URL (default: https://app.band.ai)

    2. Have peers configured on the Thenvoi platform

Editor Configuration:
    Zed (settings.json):
        {"agent_servers": {"Thenvoi": {"type": "custom", "command": "uv run examples/acp/01_basic_acp_server.py"}}}

    JetBrains (~/.jetbrains/acp.json):
        {"agent_servers": {"Thenvoi": {"command": "thenvoi-acp", "args": ["--agent-id", "..."], "env": {"THENVOI_API_KEY": "..."}}}}

Run with:
    uv run examples/acp/01_basic_acp_server.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from acp import run_agent
from dotenv import load_dotenv

from setup_logging import setup_logging
from thenvoi import Agent
from thenvoi.adapters import ACPServer, ThenvoiACPServerAdapter
from thenvoi.config import load_agent_config

setup_logging()
logger = logging.getLogger(__name__)


async def main() -> None:
    load_dotenv()

    ws_url = os.getenv("THENVOI_WS_URL", "wss://app.band.ai/api/v1/socket/websocket")
    rest_url = os.getenv("THENVOI_REST_URL", "https://app.band.ai")
    # ACP server examples check env vars first because editors (Zed, Cursor)
    # typically inject credentials via environment when spawning the subprocess.
    # Both THENVOI_API_KEY and THENVOI_AGENT_ID must be set together — a key
    # without an agent_id silently falls back to a placeholder id, causing
    # 401 "API key not linked to a valid user" errors at runtime.
    api_key = os.getenv("THENVOI_API_KEY")
    agent_id = os.getenv("THENVOI_AGENT_ID")

    if not (api_key and agent_id):
        try:
            agent_id, api_key = load_agent_config("acp_server_agent")
        except Exception as e:
            raise ValueError(
                "THENVOI_API_KEY and THENVOI_AGENT_ID environment variables are "
                "required together, or configure 'acp_server_agent' in "
                "agent_config.yaml"
            ) from e

    # Create ACP server adapter with direct REST client
    adapter = ThenvoiACPServerAdapter(
        rest_url=rest_url,
        api_key=api_key,
    )

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

    logger.info("Starting ACP server (Thenvoi as ACP agent)...")
    logger.info("Waiting for editor to connect via stdio...")

    # Start platform connection (non-blocking)
    await agent.start()
    try:
        # Block on stdio until editor disconnects
        await run_agent(server)
    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
