# /// script
# requires-python = ">=3.11"
# dependencies = ["thenvoi-sdk[acp]"]
#
# [tool.uv.sources]
# thenvoi-sdk = { git = "https://github.com/thenvoi/thenvoi-sdk-python.git" }
# ///
"""
Cursor ACP Server - Let Cursor use Thenvoi as an ACP agent.

This example starts Thenvoi as an ACP agent that Cursor connects to.
When Cursor sends prompts, they are routed to Thenvoi platform peers
(other agents in the room) and responses stream back to Cursor's UI.

Architecture:
    Cursor IDE
      -> spawns this process as ACP agent
        -> ACPServer (ACP JSON-RPC over stdio)
          -> ThenvoiACPServerAdapter
            -> Thenvoi Platform (creates room, sends message)
              -> Peer agents respond via WebSocket
            -> Streams responses back to Cursor via session_update

Cursor Configuration (~/.cursor/mcp.json):
    {
        "mcpServers": {
            "thenvoi": {
                "command": "thenvoi-acp",
                "args": ["--agent-id", "YOUR_AGENT_ID"],
                "env": {
                    "THENVOI_API_KEY": "YOUR_API_KEY"
                }
            }
        }
    }

    Or configure as a custom agent in Cursor settings:
    {
        "agent_servers": {
            "Thenvoi": {
                "type": "custom",
                "command": "thenvoi-acp",
                "args": ["--agent-id", "YOUR_AGENT_ID"],
                "env": {
                    "THENVOI_API_KEY": "YOUR_API_KEY"
                }
            }
        }
    }

Prerequisites:
    1. Install: pip install thenvoi-sdk[acp]
    2. Set THENVOI_API_KEY and THENVOI_AGENT_ID

Run standalone for testing:
    THENVOI_API_KEY=... THENVOI_AGENT_ID=... uv run examples/acp/07_cursor_server.py
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
from thenvoi.integrations.acp.push_handler import ACPPushHandler
from thenvoi.integrations.acp.router import AgentRouter
from thenvoi.integrations.acp.server import ACPServer
from thenvoi.integrations.acp.server_adapter import ThenvoiACPServerAdapter

setup_logging()
logger = logging.getLogger(__name__)


async def main() -> None:
    load_dotenv()

    ws_url = os.getenv(
        "THENVOI_WS_URL", "wss://app.thenvoi.com/api/v1/socket/websocket"
    )
    rest_url = os.getenv("THENVOI_REST_URL", "https://app.thenvoi.com")
    api_key = os.getenv("THENVOI_API_KEY")
    agent_id = os.getenv("THENVOI_AGENT_ID")

    if not api_key:
        raise ValueError(
            "THENVOI_API_KEY is required. Set it in Cursor's agent_servers env config."
        )
    if not agent_id:
        raise ValueError(
            "THENVOI_AGENT_ID is required. Pass via --agent-id or set THENVOI_AGENT_ID."
        )

    # Create ACP server adapter
    adapter = ThenvoiACPServerAdapter(
        rest_url=rest_url,
        api_key=api_key,
    )

    # Optional: configure routing for slash commands
    # Users can type "/codex fix bug" in Cursor to route to a specific peer
    router = AgentRouter(
        slash_commands={
            "codex": "codex",
            "claude": "claude-code",
        },
    )
    adapter.set_router(router)

    # Wire up push handler for real-time activity from other agents
    push_handler = ACPPushHandler(adapter)
    adapter.set_push_handler(push_handler)

    # Create ACP protocol handler
    server = ACPServer(adapter)

    # Create Thenvoi agent
    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    logger.info("Starting Thenvoi ACP server for Cursor...")
    logger.info("Cursor will connect via stdio ACP protocol.")

    await agent.start()
    try:
        await run_agent(server)
    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
