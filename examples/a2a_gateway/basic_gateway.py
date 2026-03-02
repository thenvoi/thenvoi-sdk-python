"""Basic A2A Gateway adapter example implementation."""

from __future__ import annotations

import logging
import os

from thenvoi.example_support.bootstrap import (
    ExampleRuntimeConfig,
    create_agent_from_runtime,
    load_platform_urls,
    load_runtime_config,
)
from thenvoi.adapters import A2AGatewayAdapter
from thenvoi.config.defaults import DEFAULT_REST_URL, DEFAULT_WS_URL
from thenvoi.testing.example_logging import setup_logging_profile

logger = logging.getLogger(__name__)


async def main() -> None:
    setup_logging_profile("a2a_gateway")

    ws_url, rest_url = load_platform_urls(
        ws_default=DEFAULT_WS_URL,
        rest_default=DEFAULT_REST_URL,
    )
    api_key = os.getenv("THENVOI_API_KEY")

    if api_key:
        runtime = ExampleRuntimeConfig(
            agent_key="gateway_agent",
            agent_id=os.getenv("THENVOI_AGENT_ID", "a2a-gateway"),
            api_key=api_key,
            ws_url=ws_url,
            rest_url=rest_url,
        )
    else:
        try:
            runtime = load_runtime_config(
                "gateway_agent",
                ws_default=DEFAULT_WS_URL,
                rest_default=DEFAULT_REST_URL,
                load_env=False,
            )
        except Exception as exc:
            raise ValueError(
                "THENVOI_API_KEY environment variable is required, "
                "or configure 'gateway_agent' in agent_config.yaml"
            ) from exc

    gateway_port = int(os.getenv("GATEWAY_PORT", "10000"))
    gateway_url = os.getenv("GATEWAY_URL", f"http://localhost:{gateway_port}")

    adapter = A2AGatewayAdapter(
        rest_url=runtime.rest_url,
        api_key=runtime.api_key,
        gateway_url=gateway_url,
        port=gateway_port,
    )

    session = create_agent_from_runtime(runtime, adapter)

    logger.info("Starting A2A Gateway on %s...", gateway_url)
    logger.info("Peers will be exposed at:")
    logger.info(
        "  - %s/agents/{peer_id}/.well-known/agent.json (discovery)", gateway_url
    )
    logger.info("  - %s/agents/{peer_id}/v1/message:stream (messaging)", gateway_url)
    logger.info("Waiting for peers to be discovered...")

    await session.agent.run()
