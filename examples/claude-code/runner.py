#!/usr/bin/env python3
"""
YAML-based agent runner for Thenvoi Claude SDK.

Reads agent configuration from a YAML file and runs the agent.
Designed for Docker deployment without writing Python code.

Usage:
    AGENT_CONFIG=/app/config/agent.yaml python runner.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load agent configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(path) as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required = ["agent_id", "api_key"]
    missing = [field for field in required if not config.get(field)]
    if missing:
        logger.error(f"Missing required config fields: {missing}")
        sys.exit(1)

    return config


def load_custom_tools(tools_dir: Path, tool_names: list[str]) -> list:
    """Load custom tools from tools directory.

    Returns a list of tool functions (decorated with @tool from claude_agent_sdk).
    """
    tools_init = tools_dir / "__init__.py"
    if not tools_init.exists():
        return []

    # Import the tools module
    sys.path.insert(0, str(tools_dir.parent))
    try:
        from tools import TOOL_REGISTRY

        # Filter to only requested tools, return as list
        return [TOOL_REGISTRY[name] for name in tool_names if name in TOOL_REGISTRY]
    except ImportError as e:
        logger.warning(f"Could not load custom tools: {e}")
        return []


async def main():
    """Run the agent from YAML configuration."""
    # Get config path from environment
    config_path = os.environ.get("AGENT_CONFIG")
    if not config_path:
        logger.error("AGENT_CONFIG environment variable not set")
        sys.exit(1)

    logger.info(f"Loading config from: {config_path}")
    config = load_config(config_path)

    # Import here to allow early config validation
    from thenvoi import Agent
    from thenvoi.adapters import ClaudeSDKAdapter

    # Extract config values
    agent_id = config["agent_id"]
    api_key = config["api_key"]
    model = config.get("model", "claude-sonnet-4-5-20250929")
    prompt = config.get("prompt", "You are a helpful assistant.")
    thinking_tokens = config.get("thinking_tokens")
    tool_names = config.get("tools", [])

    # Load custom tools if specified
    custom_tools = []
    if tool_names:
        tools_dir = Path(config_path).parent / "tools"
        custom_tools = load_custom_tools(tools_dir, tool_names)
        if custom_tools:
            tool_fn_names = [getattr(t, "_tool_name", t.__name__) for t in custom_tools]
            logger.info(f"Loaded custom tools: {tool_fn_names}")

    # Create adapter
    adapter = ClaudeSDKAdapter(
        model=model,
        custom_section=prompt,
        max_thinking_tokens=thinking_tokens,
        enable_execution_reporting=True,
        custom_tools=custom_tools if custom_tools else None,
    )

    # Create agent
    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
    )

    logger.info(f"Starting agent: {agent_id}")
    logger.info(f"Model: {model}")
    if thinking_tokens:
        logger.info(f"Extended thinking: {thinking_tokens} tokens")
    logger.info("Press Ctrl+C to stop")

    try:
        await agent.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    asyncio.run(main())
