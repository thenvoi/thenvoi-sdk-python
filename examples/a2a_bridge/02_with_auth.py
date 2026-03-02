# /// script
# requires-python = ">=3.11"
# dependencies = ["thenvoi-sdk[a2a]"]
#
# [tool.uv.sources]
# thenvoi-sdk = { git = "https://github.com/thenvoi/thenvoi-sdk-python.git" }
# ///
"""
A2A adapter with authentication example.

This example shows how to connect to a remote A2A agent that requires
authentication (API key, bearer token, or custom headers).

Run with:
    uv run examples/a2a_bridge/02_with_auth.py
"""

from __future__ import annotations

import asyncio
import logging
import os

from examples.a2a_bridge.setup_logging import setup_logging
from thenvoi.example_support.bootstrap import bootstrap_agent
from thenvoi.adapters import A2AAdapter
from thenvoi.integrations.a2a import A2AAuth

logger = logging.getLogger(__name__)


async def main() -> None:
    setup_logging()

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
        logger.info("Using authentication for A2A agent")

    # Create adapter with auth
    adapter = A2AAdapter(
        remote_url=a2a_url,
        auth=auth,
        streaming=True,
    )

    session = bootstrap_agent(agent_key="a2a_agent", adapter=adapter)

    logger.info("Starting A2A bridge agent (forwarding to %s)...", a2a_url)
    await session.agent.run()


if __name__ == "__main__":
    asyncio.run(main())
