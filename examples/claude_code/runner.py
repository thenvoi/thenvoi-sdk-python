#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["thenvoi-sdk", "pyyaml"]
#
# [tool.uv.sources]
# thenvoi-sdk = { git = "https://github.com/thenvoi/thenvoi-sdk-python.git" }
# ///
"""
Claude Code CLI agent runner for Thenvoi platform.

Reads agent configuration from YAML and runs ClaudeCodeDesktopAdapter
with workspace access for file operations.

Usage:
    AGENT_CONFIG=/app/agent_config.yaml python runner.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from pathlib import Path
from typing import Any

import yaml

from thenvoi.config.loader import load_agent_config

# Global shutdown event
_shutdown_event: asyncio.Event | None = None

# Required mount points for Docker operation
NOTES_DIR = os.environ.get("NOTES_DIR", "/workspace/notes")
WORKSPACE_DIR = os.environ.get("WORKSPACE", "/workspace/repo")

REQUIRED_MOUNTS = [
    WORKSPACE_DIR,
    NOTES_DIR,
]

# Retry configuration
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1.0
MAX_RETRY_DELAY = 60.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def validate_mounts() -> None:
    """Validate required mount points exist."""
    missing = [m for m in REQUIRED_MOUNTS if not Path(m).is_dir()]
    if missing:
        raise ValueError(
            f"Missing required mount points: {missing}. "
            "Ensure docker-compose.yml mounts: "
            f"{', '.join(f'{m} (rw)' for m in REQUIRED_MOUNTS)}. "
        )


def load_config(config_path: str) -> dict[str, Any]:
    """Load agent configuration from YAML file.

    Credentials (agent_id, api_key) are validated via the SDK's
    ``load_agent_config()`` so all examples follow a single path
    for credential loading.  Additional fields (role, prompt, etc.)
    are returned as-is for the runner to consume.
    """
    path = Path(config_path).resolve()
    if not path.exists():
        raise ValueError(f"Config file not found: {config_path}")

    try:
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {e}") from e
    except OSError as e:
        raise ValueError(f"Failed to read config file: {e}") from e

    if config is None:
        raise ValueError("Config file is empty")

    # Validate credentials via the SDK config loader (supports flat YAML format)
    agent_id, api_key = load_agent_config("agent", config_path=path)
    config["agent_id"] = agent_id
    config["api_key"] = api_key

    return config


def _handle_signal(sig: signal.Signals) -> None:
    """Handle shutdown signals (SIGTERM, SIGINT)."""
    logger.info("Received %s, initiating graceful shutdown...", sig.name)
    if _shutdown_event:
        _shutdown_event.set()


async def main() -> None:
    """Run the Claude Code agent."""
    global _shutdown_event
    _shutdown_event = asyncio.Event()

    # Setup signal handlers for graceful shutdown (Docker sends SIGTERM)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handle_signal, sig)

    # Get paths from environment
    config_path = os.environ.get("AGENT_CONFIG", "/app/agent_config.yaml")
    prompt_dir = os.environ.get("PROMPT_DIR", "/prompts")
    workspace = os.environ.get("WORKSPACE", "/workspace/repo")

    # Validate Thenvoi platform URLs
    ws_url = os.environ.get(
        "THENVOI_WS_URL", "wss://app.thenvoi.com/api/v1/socket/websocket"
    )
    rest_url = os.environ.get("THENVOI_REST_URL", "https://app.thenvoi.com")

    if not ws_url:
        raise ValueError("THENVOI_WS_URL environment variable is empty")
    if not rest_url:
        raise ValueError("THENVOI_REST_URL environment variable is empty")

    # Validate required mount points
    validate_mounts()

    logger.info("Loading config from: %s", config_path)
    config = load_config(config_path)

    # Import here to allow early config validation
    from thenvoi import Agent
    from thenvoi.adapters import ClaudeCodeDesktopAdapter
    from thenvoi.prompts import load_role_prompt

    # Extract config values
    agent_id = config["agent_id"]
    api_key = config["api_key"]

    # Get role (env overrides config)
    role = os.environ.get("AGENT_ROLE") or config.get("role")
    custom_prompt = config.get("prompt", "")

    # Build final prompt
    final_prompt_parts = []

    if role:
        role_prompt = load_role_prompt(role, prompt_dir)
        if role_prompt:
            final_prompt_parts.append(role_prompt)
            logger.info("Using role: %s", role)

    if custom_prompt:
        final_prompt_parts.append(custom_prompt)

    # Add workspace context
    workspace_context = f"""
## Workspace Access

You have access to the following directories:
- `{workspace}` - Project source code (read-only)
- `{NOTES_DIR}` - Markdown notes, plans, and design documents (read-write)

Current working directory: {workspace}

When creating design docs or plans, save them to `{NOTES_DIR}/`.
"""
    final_prompt_parts.append(workspace_context)

    final_prompt = "\n\n".join(final_prompt_parts) if final_prompt_parts else None

    # Create adapter with file operation tools enabled
    adapter = ClaudeCodeDesktopAdapter(
        custom_section=final_prompt,
        cli_timeout=300000,  # 5 minutes for complex operations
        allowed_tools=["Read", "Write", "Edit", "Glob", "Grep", "Bash"],
        verbose=True,
    )

    # Create agent
    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    logger.info("Starting Claude Code agent: %s", agent_id)
    if role:
        logger.info("Role: %s", role)
    logger.info("Workspace: %s", workspace)
    logger.info("Press Ctrl+C to stop")

    retry_count = 0
    retry_delay = INITIAL_RETRY_DELAY

    while not _shutdown_event.is_set():
        try:
            # Run agent with shutdown event monitoring
            agent_task = asyncio.create_task(agent.run())
            shutdown_task = asyncio.create_task(_shutdown_event.wait())

            # Wait for either agent completion or shutdown signal
            done, pending = await asyncio.wait(
                [agent_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Check if agent task raised an exception
            if agent_task in done:
                agent_task.result()

            # Reset retry state on successful connection
            retry_count = 0
            retry_delay = INITIAL_RETRY_DELAY

            # If we get here without exception, agent completed normally
            break

        except (ConnectionError, OSError) as e:
            retry_count += 1
            if retry_count > MAX_RETRIES:
                logger.error("Max retries (%s) exceeded, giving up", MAX_RETRIES)
                raise

            logger.warning(
                "Connection error: %s. Retrying in %.1fs (attempt %s/%s)",
                e,
                retry_delay,
                retry_count,
                MAX_RETRIES,
            )
            await asyncio.sleep(retry_delay)
            # Exponential backoff with cap
            retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)

        except asyncio.CancelledError:
            logger.info("Agent task cancelled")
            break

    logger.info("Shutting down...")
    try:
        if hasattr(agent, "close"):
            await agent.close()
    except Exception as e:
        logger.warning("Error during agent cleanup: %s", e)
    logger.info("Agent stopped")


if __name__ == "__main__":
    asyncio.run(main())
