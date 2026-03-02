"""Run an A2A gateway with a demo orchestrator agent."""

from __future__ import annotations

import asyncio
import logging
import os
import threading

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from dotenv import load_dotenv

from thenvoi import Agent
from thenvoi.adapters import A2AGatewayAdapter
from thenvoi.config import load_agent_config
from thenvoi.config.defaults import DEFAULT_REST_URL, DEFAULT_WS_URL
from thenvoi.integrations.a2a_gateway.orchestrator.agent import OrchestratorAgent
from thenvoi.integrations.a2a_gateway.orchestrator.agent_executor import (
    OrchestratorAgentExecutor,
)
from thenvoi.testing.example_logging import setup_logging_profile

logger = logging.getLogger(__name__)

DEFAULT_GATEWAY_HOST = "localhost"
DEFAULT_ORCHESTRATOR_HOST = "localhost"
DEFAULT_GATEWAY_PORT = 10000
DEFAULT_ORCHESTRATOR_PORT = 10001
DEFAULT_MODEL = "gpt-4o"


def _gateway_port() -> int:
    return int(os.getenv("GATEWAY_PORT", str(DEFAULT_GATEWAY_PORT)))


def _orchestrator_port() -> int:
    return int(os.getenv("ORCHESTRATOR_PORT", str(DEFAULT_ORCHESTRATOR_PORT)))


def _local_http_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


async def run_gateway() -> None:
    """Run the A2A gateway that exposes Thenvoi peers."""
    ws_url = os.getenv("THENVOI_WS_URL", DEFAULT_WS_URL)
    rest_url = os.getenv("THENVOI_REST_URL", DEFAULT_REST_URL)
    api_key = os.getenv("THENVOI_API_KEY")

    if not api_key:
        try:
            agent_id, api_key = load_agent_config("gateway_agent")
        except Exception as exc:
            logger.error(
                "THENVOI_API_KEY environment variable is required, "
                "or configure 'gateway_agent' in agent_config.yaml",
                extra={"error": str(exc)},
            )
            return
    else:
        agent_id = os.getenv("THENVOI_AGENT_ID", "a2a-gateway")

    gateway_port = _gateway_port()
    gateway_url = _local_http_url(DEFAULT_GATEWAY_HOST, gateway_port)

    adapter = A2AGatewayAdapter(
        rest_url=rest_url,
        api_key=api_key,
        gateway_url=gateway_url,
        port=gateway_port,
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    logger.info("Starting A2A Gateway on %s...", gateway_url)
    await agent.run()


def run_orchestrator() -> None:
    """Run the demo orchestrator that calls gateway peers."""
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("Orchestrator startup aborted due to missing configuration")
        return

    gateway_url = _local_http_url(DEFAULT_GATEWAY_HOST, _gateway_port())
    available_peers = os.getenv("AVAILABLE_PEERS", "").split(",")
    available_peers = [peer.strip() for peer in available_peers if peer.strip()]

    agent = OrchestratorAgent(
        gateway_url=gateway_url,
        available_peers=available_peers,
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
    )

    capabilities = AgentCapabilities(streaming=True, push_notifications=False)
    skill = AgentSkill(
        id="orchestrate_peers",
        name="Peer Orchestration",
        description="Routes requests to Thenvoi platform peers via A2A Gateway",
        tags=["orchestration", "routing"],
        examples=["Ask the weather peer about NYC conditions"],
    )
    orchestrator_port = _orchestrator_port()
    agent_card = AgentCard(
        name="Demo Orchestrator",
        description="Routes user requests to Thenvoi platform peers via A2A Gateway",
        url=f"{_local_http_url(DEFAULT_ORCHESTRATOR_HOST, orchestrator_port)}/",
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

    logger.info(
        "Starting Demo Orchestrator on http://%s:%s...",
        DEFAULT_ORCHESTRATOR_HOST,
        orchestrator_port,
    )
    uvicorn.run(
        server.build(),
        host=DEFAULT_ORCHESTRATOR_HOST,
        port=orchestrator_port,
    )


async def main() -> None:
    """Run both gateway and orchestrator."""
    setup_logging_profile("a2a_gateway")
    load_dotenv()

    gateway_port = _gateway_port()
    orchestrator_port = _orchestrator_port()

    logger.info("%s", "=" * 60)
    logger.info("A2A Gateway + Demo Orchestrator Example")
    logger.info("%s", "=" * 60)
    logger.info("This example runs:")
    logger.info("  1. A2A Gateway on port %s (exposes Thenvoi peers)", gateway_port)
    logger.info(
        "  2. Demo Orchestrator on port %s (calls gateway peers)", orchestrator_port
    )
    logger.info("Test with:")
    logger.info("  curl http://localhost:%s/.well-known/agent.json", orchestrator_port)

    gateway_task = asyncio.create_task(run_gateway())
    await asyncio.sleep(2)

    orchestrator_thread = threading.Thread(target=run_orchestrator, daemon=True)
    orchestrator_thread.start()

    try:
        await gateway_task
    except asyncio.CancelledError:
        logger.info("Shutting down...")
        raise
