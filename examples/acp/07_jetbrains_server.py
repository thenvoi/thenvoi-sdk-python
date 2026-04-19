# /// script
# requires-python = ">=3.11"
# dependencies = ["thenvoi-sdk[acp]"]
#
# [tool.uv.sources]
# thenvoi-sdk = { git = "https://github.com/thenvoi/thenvoi-sdk-python.git" }
# ///
"""
JetBrains ACP Server - Use Thenvoi as an ACP agent in JetBrains IDEs.

This example starts Thenvoi as an ACP agent that JetBrains IDEs (IntelliJ,
PyCharm, WebStorm, etc.) can connect to via the ACP protocol. When you type
prompts in the JetBrains AI Chat, they are routed to Thenvoi platform peers
and responses stream back to the IDE.

Architecture:
    JetBrains IDE (AI Chat)
      -> spawns this process as ACP agent
        -> ACPServer (ACP JSON-RPC over stdio)
          -> ThenvoiACPServerAdapter
            -> Thenvoi Platform (creates room, sends message)
              -> Peer agents respond via WebSocket
            -> Streams responses back to IDE via session_update

JetBrains Configuration (~/.jetbrains/acp.json):
    {
        "default_mcp_settings": {},
        "agent_servers": {
            "Thenvoi": {
                "command": "thenvoi-acp",
                "args": ["--agent-id", "YOUR_AGENT_ID"],
                "env": {
                    "THENVOI_API_KEY": "YOUR_API_KEY"
                }
            }
        }
    }

    Or if running from source:
    {
        "default_mcp_settings": {},
        "agent_servers": {
            "Thenvoi": {
                "command": "uv",
                "args": [
                    "run", "--extra", "acp",
                    "thenvoi-acp", "--agent-id", "YOUR_AGENT_ID"
                ],
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
    THENVOI_API_KEY=... THENVOI_AGENT_ID=... uv run examples/acp/07_jetbrains_server.py
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
from thenvoi.config import load_agent_config
from thenvoi.integrations.acp.push_handler import ACPPushHandler
from thenvoi.integrations.acp.router import AgentRouter
from thenvoi.integrations.acp.server import ACPServer
from thenvoi.integrations.acp.server_adapter import ThenvoiACPServerAdapter

setup_logging()
logger = logging.getLogger(__name__)


async def main() -> None:
    load_dotenv()

    ws_url = os.getenv(
        "THENVOI_WS_URL", "wss://app.band.ai/dashboard/api/v1/socket/websocket"
    )
    rest_url = os.getenv("THENVOI_REST_URL", "https://app.band.ai/dashboard")
    # JetBrains IDEs inject credentials via ~/.jetbrains/acp.json env config.
    # Fall back to agent_config.yaml for standalone testing.
    api_key = os.getenv("THENVOI_API_KEY")

    if not api_key:
        try:
            agent_id, api_key = load_agent_config("jetbrains_acp_agent")
        except Exception:
            raise ValueError(
                "THENVOI_API_KEY environment variable is required, "
                "or configure 'jetbrains_acp_agent' in agent_config.yaml"
            )
    else:
        agent_id = os.getenv("THENVOI_AGENT_ID")
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
    # Users can type "/codex fix bug" in the AI Chat to route to a specific peer
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

    logger.info("Starting Thenvoi ACP server for JetBrains...")
    logger.info("IDE will connect via stdio ACP protocol.")

    await agent.start()
    try:
        await run_agent(server)
    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
