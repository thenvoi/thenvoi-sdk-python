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
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict[str, Any]:
    """Load agent configuration from YAML file."""
    path = Path(config_path).resolve()
    if not path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    try:
        with open(path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in config file: {e}")
        sys.exit(1)
    except OSError as e:
        logger.error(f"Failed to read config file: {e}")
        sys.exit(1)

    if config is None:
        logger.error("Config file is empty")
        sys.exit(1)

    # Validate required fields
    required = ["agent_id", "api_key"]
    missing = [field for field in required if not config.get(field)]
    if missing:
        logger.error(f"Missing required config fields: {missing}")
        sys.exit(1)

    return config


def load_custom_tools(tools_dir: Path, config_dir: Path, tool_names: list[str]) -> list:
    """Load custom tools from tools directory.

    Args:
        tools_dir: Path to the tools directory.
        config_dir: Path to the config directory (for path traversal validation).
        tool_names: List of tool names to load.

    Returns a list of tool functions (decorated with @tool from claude_agent_sdk).
    """
    # Resolve and validate path to prevent path traversal
    resolved_tools_dir = tools_dir.resolve()
    resolved_config_dir = config_dir.resolve()

    # Ensure tools_dir is within or a sibling of config_dir
    try:
        resolved_tools_dir.relative_to(resolved_config_dir.parent)
    except ValueError:
        logger.warning(
            f"Tools directory {resolved_tools_dir} is outside allowed path, skipping"
        )
        return []

    tools_init = resolved_tools_dir / "__init__.py"
    if not tools_init.exists():
        return []

    # Use importlib to load module without modifying sys.path
    try:
        spec = importlib.util.spec_from_file_location("tools", tools_init)
        if spec is None or spec.loader is None:
            logger.warning("Could not create module spec for tools")
            return []

        tools_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tools_module)

        tool_registry = getattr(tools_module, "TOOL_REGISTRY", {})
        # Filter to only requested tools, return as list
        return [tool_registry[name] for name in tool_names if name in tool_registry]
    except Exception as e:
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
        config_dir = Path(config_path).parent
        tools_dir = config_dir / "tools"
        custom_tools = load_custom_tools(tools_dir, config_dir, tool_names)
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
    finally:
        if hasattr(agent, "close"):
            await agent.close()
        logger.info("Agent stopped")


if __name__ == "__main__":
    asyncio.run(main())
