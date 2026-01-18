"""
A2A adapter with authentication example.

This example shows how to connect to a remote A2A agent that requires
authentication (API key, bearer token, or custom headers).

Run with:
    A2A_API_KEY=xxx python 02_with_auth.py
"""

import asyncio
import os

from dotenv import load_dotenv

from setup_logging import setup_logging
from thenvoi import Agent
from thenvoi.adapters import A2AAdapter
from thenvoi.config import load_agent_config
from thenvoi.integrations.a2a import A2AAuth

setup_logging()


async def main():
    load_dotenv()

    ws_url = os.getenv("THENVOI_WS_URL")
    rest_url = os.getenv("THENVOI_REST_URL")

    if not ws_url:
        raise ValueError("THENVOI_WS_URL environment variable is required")
    if not rest_url:
        raise ValueError("THENVOI_REST_URL environment variable is required")

    # Load agent credentials from agent_config.yaml
    agent_id, api_key = load_agent_config("a2a_agent")

    # URL of the remote A2A agent
    a2a_url = os.getenv("A2A_AGENT_URL", "http://localhost:10000")

    # A2A agent authentication (if required)
    a2a_api_key = os.getenv("A2A_API_KEY")
    a2a_bearer_token = os.getenv("A2A_BEARER_TOKEN")

    # Configure auth if credentials provided
    auth = None
    if a2a_api_key or a2a_bearer_token:
        auth = A2AAuth(
            api_key=a2a_api_key,
            bearer_token=a2a_bearer_token,
        )
        print("Using authentication for A2A agent")

    # Create adapter with auth
    adapter = A2AAdapter(
        remote_url=a2a_url,
        auth=auth,
        streaming=True,
    )

    # Create and start agent
    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    print(f"Starting A2A bridge agent (forwarding to {a2a_url})...")
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
