"""Docker entrypoint for Letta bridge with graceful shutdown."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

import yaml

from thenvoi.adapters.letta.adapter import LettaAdapter
from thenvoi.adapters.letta.modes import LettaConfig, LettaMode
from thenvoi.agent import Agent, SessionConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load bridge config file."""
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(path) as f:
        return yaml.safe_load(f)


class GracefulShutdown:
    """Handle graceful shutdown on SIGTERM/SIGINT."""

    def __init__(self) -> None:
        self.shutdown_event = asyncio.Event()

    def setup(self) -> None:
        """Register signal handlers."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._handle_signal)

    def _handle_signal(self, signum: int, frame: object) -> None:
        """Handle shutdown signal."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_event.set()


async def main() -> None:
    """Run the Letta bridge."""
    config_path = os.environ.get("CONFIG_FILE", "/config/bridge_config.yaml")
    logger.info(f"Loading config from {config_path}")
    config = load_config(config_path)

    # Validate required fields
    required = [
        ("platform", "rest_url"),
        ("platform", "ws_url"),
        ("agent", "id"),
        ("agent", "api_key"),
    ]
    for section, key in required:
        if not config.get(section, {}).get(key):
            logger.error(f"Missing required config: {section}.{key}")
            sys.exit(1)

    # Build LettaConfig - URLs from environment (set by docker-compose)
    letta_settings = config.get("letta", {})
    letta_config = LettaConfig(
        base_url=os.environ.get("LETTA_BASE_URL", "http://letta:8283"),
        mcp_server_url=os.environ.get("MCP_SERVER_URL", "http://mcp:8000/sse"),
        mode=(
            LettaMode.SHARED
            if letta_settings.get("mode") == "shared"
            else LettaMode.PER_ROOM
        ),
        model=letta_settings.get("model", "openai/gpt-4o"),
        embedding_model=letta_settings.get(
            "embedding_model", "openai/text-embedding-3-small"
        ),
        persona=config.get("agent", {}).get("persona"),
    )

    adapter = LettaAdapter(
        config=letta_config,
        state_storage_path=os.environ.get(
            "STATE_PATH", "/app/state/letta_adapter_state.json"
        ),
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=config["agent"]["id"],
        api_key=config["agent"]["api_key"],
        ws_url=config["platform"]["ws_url"],
        rest_url=config["platform"]["rest_url"],
        session_config=SessionConfig(enable_context_hydration=False),
    )

    # Setup graceful shutdown
    shutdown = GracefulShutdown()
    shutdown.setup()

    agent_name = config.get("agent", {}).get("name", config["agent"]["id"])
    logger.info(f"Starting bridge for agent {agent_name}")
    logger.info(f"Platform: {config['platform']['rest_url']}")
    logger.info(f"Letta mode: {letta_config.mode.value}")

    # Run until shutdown signal
    try:
        agent_task = asyncio.create_task(agent.run())
        shutdown_task = asyncio.create_task(shutdown.shutdown_event.wait())

        done, pending = await asyncio.wait(
            [agent_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except Exception as e:
        logger.error(f"Bridge error: {e}")
        raise
    finally:
        logger.info("Bridge shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
