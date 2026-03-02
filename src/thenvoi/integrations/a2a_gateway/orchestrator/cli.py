"""CLI runtime for the demo A2A orchestrator server."""

from __future__ import annotations

import asyncio
import logging
import os

import click
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from dotenv import load_dotenv

from .agent import OrchestratorAgent
from .agent_executor import OrchestratorAgentExecutor
from .remote_agent import GatewayClient, GatewayDiscoveryError

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _parse_available_peers(peers: str) -> list[str]:
    return [peer.strip() for peer in peers.split(",") if peer.strip()]


async def _discover_peers(gateway_url: str) -> list[str]:
    async with GatewayClient(gateway_url) as client:
        peers_data = await client.list_peers()
    return [peer["slug"] for peer in peers_data if peer.get("slug")]


def run_orchestrator_server(
    host: str,
    port: int,
    gateway_url: str,
    peers: str,
    model: str,
) -> int:
    """Start the demo orchestrator server and return an exit code."""
    load_dotenv()
    _configure_logging()

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("Required runtime configuration is missing")
        return 1

    available_peers = _parse_available_peers(peers)
    if not available_peers:
        logger.info("Discovering peers from gateway at %s...", gateway_url)
        try:
            available_peers = asyncio.run(_discover_peers(gateway_url))
            if available_peers:
                logger.info("Discovered %s peers from gateway", len(available_peers))
            else:
                logger.warning("No peers discovered from gateway")
        except GatewayDiscoveryError as error:
            logger.warning(
                "Could not discover peers from gateway (code=%s, retryable=%s): %s",
                error.code,
                error.retryable,
                error,
            )

    logger.info("Starting Demo Orchestrator Agent on %s:%s", host, port)
    logger.info("Gateway URL: %s", gateway_url)
    if available_peers:
        peers_summary = ", ".join(available_peers[:5])
        suffix = "..." if len(available_peers) > 5 else ""
        logger.info(
            "Available peers (%s): %s%s", len(available_peers), peers_summary, suffix
        )
    else:
        logger.warning("No peers available - orchestrator will have limited functionality")

    try:
        agent = OrchestratorAgent(
            gateway_url=gateway_url,
            available_peers=available_peers,
            model=model,
        )
        capabilities = AgentCapabilities(streaming=True, push_notifications=False)
        skill = AgentSkill(
            id="orchestrate_peers",
            name="Peer Orchestration",
            description="Routes requests to specialized Thenvoi platform peers via A2A Gateway",
            tags=["orchestration", "routing", "multi-agent"],
            examples=[
                "Ask the weather agent about conditions in NYC",
                "Create a ticket using the ServiceNow agent",
                "Get data analysis from the data analyst peer",
            ],
        )
        agent_card = AgentCard(
            name="Demo Orchestrator",
            description=(
                "An orchestrator agent that routes user requests to specialized "
                "Thenvoi platform peers via the A2A Gateway. It intelligently "
                "determines which peer can best handle each request."
            ),
            url=f"http://{host}:{port}/",
            version="1.0.0",
            default_input_modes=OrchestratorAgent.SUPPORTED_CONTENT_TYPES,
            default_output_modes=OrchestratorAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        request_handler = DefaultRequestHandler(
            agent_executor=OrchestratorAgentExecutor(agent),
            task_store=InMemoryTaskStore(),
            push_config_store=InMemoryPushNotificationConfigStore(),
        )
        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )
        uvicorn.run(server.build(), host=host, port=port)
    except Exception:
        logger.exception("Error starting server")
        return 1

    return 0


@click.command()
@click.option("--host", default="localhost", help="Host to bind to")
@click.option("--port", default=10001, help="Port to bind to")
@click.option(
    "--gateway-url",
    default=os.getenv("GATEWAY_URL", "http://localhost:10000"),
    help="URL of the A2A Gateway",
)
@click.option(
    "--peers",
    default=os.getenv("AVAILABLE_PEERS", ""),
    help="Comma-separated list of available peer IDs",
)
@click.option(
    "--model",
    default=os.getenv("OPENAI_MODEL", "gpt-4o"),
    help="OpenAI model to use",
)
def main(host: str, port: int, gateway_url: str, peers: str, model: str) -> None:
    """Start the demo orchestrator A2A server."""
    raise SystemExit(run_orchestrator_server(host, port, gateway_url, peers, model))
