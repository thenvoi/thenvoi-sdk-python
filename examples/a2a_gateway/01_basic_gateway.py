"""
Basic A2A Gateway adapter example.

This example creates a gateway that exposes Thenvoi platform peers as A2A
endpoints. External A2A-compliant agents can connect to this gateway and
interact with Thenvoi peers via standard A2A protocol.

Use Case:
    - You have an external agent (e.g., SAP Agent) that uses A2A protocol
    - You want that agent to interact with Thenvoi platform peers
    - This gateway runs as a sidecar, exposing peers as A2A endpoints

Architecture:
    External Agent → A2A HTTP → Gateway → Thenvoi REST API → Platform Peers
                  ↑                                              ↓
                  ←←←←←←← SSE Response Stream ←←←←←←←←←←←←←←←←←←←

Features:
    - Automatic peer discovery from Thenvoi platform
    - Per-peer A2A endpoints with AgentCard discovery
    - SSE streaming for real-time responses
    - Context management (room-per-context)
    - Session rehydration on restart

Prerequisites:
    1. Set environment variables:
       - THENVOI_API_KEY: Your Thenvoi API key
       - THENVOI_WS_URL: WebSocket URL (default: wss://api.thenvoi.com/ws)
       - THENVOI_REST_URL: REST API URL (default: https://api.thenvoi.com)

    2. Have peers configured on the Thenvoi platform

Run with:
    python 01_basic_gateway.py

Then external agents can connect:
    - Discovery: GET http://localhost:10000/agents/weather/.well-known/agent.json
    - Message:   POST http://localhost:10000/agents/weather/v1/message:stream
"""

import asyncio
import os

from dotenv import load_dotenv

from setup_logging import setup_logging
from thenvoi import Agent
from thenvoi.adapters import A2AGatewayAdapter
from thenvoi.config import load_agent_config

setup_logging()


async def main():
    load_dotenv()

    ws_url = os.getenv("THENVOI_WS_URL", "wss://api.thenvoi.com/ws")
    rest_url = os.getenv("THENVOI_REST_URL", "https://api.thenvoi.com")
    api_key = os.getenv("THENVOI_API_KEY")

    if not api_key:
        # Try loading from agent_config.yaml
        try:
            agent_id, api_key = load_agent_config("gateway_agent")
        except Exception:
            raise ValueError(
                "THENVOI_API_KEY environment variable is required, "
                "or configure 'gateway_agent' in agent_config.yaml"
            )
    else:
        agent_id = os.getenv("THENVOI_AGENT_ID", "a2a-gateway")

    # Gateway configuration
    gateway_port = int(os.getenv("GATEWAY_PORT", "10000"))
    gateway_url = os.getenv("GATEWAY_URL", f"http://localhost:{gateway_port}")

    # Create gateway adapter
    # It uses its own REST client for room/message operations
    adapter = A2AGatewayAdapter(
        rest_url=rest_url,
        api_key=api_key,
        gateway_url=gateway_url,
        port=gateway_port,
    )

    # Create and start agent
    # The gateway connects to Thenvoi and starts its HTTP server
    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    print(f"Starting A2A Gateway on {gateway_url}...")
    print("Peers will be exposed at:")
    print(f"  - {gateway_url}/agents/{{peer_id}}/.well-known/agent.json (discovery)")
    print(f"  - {gateway_url}/agents/{{peer_id}}/v1/message:stream (messaging)")
    print()
    print("Waiting for peers to be discovered...")

    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
