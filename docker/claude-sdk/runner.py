#!/usr/bin/env python3
"""
Generic Claude SDK Agent Runner.

Loads configuration from /app/config/agent.yaml
Discovers and registers tools from /app/tools/

This runner is designed to be used inside the Claude SDK Docker container.
It provides a generic way to run Claude SDK agents with custom tools.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Check for PyYAML
try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)

try:
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        tool,
        create_sdk_mcp_server,
    )
except ImportError:
    print("ERROR: claude-agent-sdk is required. Install with: pip install claude-agent-sdk")
    sys.exit(1)

from thenvoi import Agent
from thenvoi.adapters import ClaudeSDKAdapter
from thenvoi.runtime.tools import get_tool_description

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load agent configuration from YAML."""
    try:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in config file: {e}")
        sys.exit(1)


def discover_tools(tools_path: str) -> list[dict]:
    """
    Discover and load custom tools from modules.

    Auto-discovers all .py files in the tools directory (except those starting with _).
    Each module should have a TOOLS list containing tool definitions.
    """
    tools = []
    tools_dir = Path(tools_path)

    if not tools_dir.exists():
        logger.info(f"Tools directory {tools_path} does not exist, skipping custom tools")
        return tools

    # Add tools path to sys.path for imports
    sys.path.insert(0, str(tools_dir))

    for py_file in sorted(tools_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue

        module_name = py_file.stem
        logger.info(f"Loading tools from {module_name}")

        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                logger.warning(f"Could not load spec for {module_name}")
                continue

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Look for TOOLS registry in module
            if hasattr(module, "TOOLS"):
                module_tools = getattr(module, "TOOLS")
                tools.extend(module_tools)
                logger.info(f"  Found {len(module_tools)} tools in {module_name}")
            else:
                logger.warning(f"  No TOOLS list found in {module_name}")

        except Exception as e:
            logger.error(f"Failed to load {module_name}: {e}")

    return tools


class CustomToolsClaudeSDKAdapter(ClaudeSDKAdapter):
    """
    Claude SDK adapter with custom tool support.

    Creates a separate MCP server for custom tools alongside the Thenvoi MCP server.

    NOTE: This requires claude-agent-sdk to support multiple MCP servers.
    If not supported, we need to create a single combined MCP server.
    """

    def __init__(self, custom_tools: list[dict], **kwargs):
        super().__init__(**kwargs)
        self._custom_tools = custom_tools
        self._custom_mcp_server = None

    async def on_started(self, agent_name: str, agent_description: str) -> None:
        """Create MCP servers after agent metadata is fetched."""
        # Create custom tools MCP server if we have custom tools
        if self._custom_tools:
            self._custom_mcp_server = self._create_custom_mcp_server()
            logger.info(f"Created custom MCP server with {len(self._custom_tools)} tools")

        # Call parent to create Thenvoi MCP server and session manager
        await super().on_started(agent_name, agent_description)

        # TODO: If claude-agent-sdk supports multiple MCP servers, add custom server here
        # Otherwise, we need to modify parent's _create_mcp_server to include custom tools

    def _create_custom_mcp_server(self):
        """
        Create MCP server for custom tools.

        Converts discovered tool definitions into claude-agent-sdk tool format.
        """

        def _make_result(data: Any) -> dict[str, Any]:
            return {"content": [{"type": "text", "text": json.dumps(data, default=str)}]}

        def _make_error(error: str) -> dict[str, Any]:
            return {
                "content": [
                    {"type": "text", "text": json.dumps({"status": "error", "message": error})}
                ],
                "is_error": True,
            }

        # Convert custom tool definitions to MCP tools
        mcp_tools = []

        for tool_def in self._custom_tools:
            tool_name = tool_def["name"]
            tool_description = tool_def["description"]
            tool_parameters = tool_def["parameters"]
            tool_handler = tool_def["handler"]

            # Convert parameters to MCP format
            param_types = {}
            for param_name, param_info in tool_parameters.items():
                param_type = param_info.get("type", "string")
                # Map to Python types for MCP
                type_map = {
                    "string": str,
                    "integer": int,
                    "boolean": bool,
                    "number": float,
                }
                param_types[param_name] = type_map.get(param_type, str)

            # Store tool info for later registration
            mcp_tools.append(
                {
                    "name": tool_name,
                    "description": tool_description,
                    "param_types": param_types,
                    "handler": tool_handler,
                }
            )

        # Log discovered tools
        logger.info(f"Custom MCP server prepared with tools: {[t['name'] for t in mcp_tools]}")

        return mcp_tools


async def main():
    """Run the generic Claude SDK agent."""

    # Setup logging
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load configuration
    config_path = os.getenv("CONFIG_PATH", "/app/config/agent.yaml")

    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        logger.error("Mount your config to /app/config/agent.yaml")
        sys.exit(1)

    config = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    # Extract settings
    agent_config = config.get("agent", {})
    claude_config = config.get("claude", {})
    system_config = config.get("system", {})
    tools_config = config.get("tools", {})
    exec_config = config.get("execution", {})

    # Validate required fields (config takes priority, then env vars)
    agent_id = agent_config.get("id") or os.getenv("THENVOI_AGENT_ID")
    api_key = agent_config.get("api_key") or os.getenv("THENVOI_API_KEY")

    if not agent_id or not api_key:
        logger.error("agent.id and agent.api_key are required (in config or env vars)")
        sys.exit(1)

    # Discover custom tools if enabled
    custom_tools = []
    if tools_config.get("custom", {}).get("enabled", True):
        tools_path = os.getenv("TOOLS_PATH", "/app/tools")
        custom_tools = discover_tools(tools_path)
        logger.info(f"Discovered {len(custom_tools)} custom tools")

    # Create adapter - use custom adapter if we have custom tools
    if custom_tools:
        adapter = CustomToolsClaudeSDKAdapter(
            custom_tools=custom_tools,
            model=claude_config.get("model", "claude-sonnet-4-5-20250929"),
            custom_section=system_config.get("custom_section"),
            max_thinking_tokens=claude_config.get("max_thinking_tokens"),
            permission_mode=claude_config.get("permission_mode", "acceptEdits"),
            enable_execution_reporting=exec_config.get("enable_reporting", True),
        )
    else:
        adapter = ClaudeSDKAdapter(
            model=claude_config.get("model", "claude-sonnet-4-5-20250929"),
            custom_section=system_config.get("custom_section"),
            max_thinking_tokens=claude_config.get("max_thinking_tokens"),
            permission_mode=claude_config.get("permission_mode", "acceptEdits"),
            enable_execution_reporting=exec_config.get("enable_reporting", True),
        )

    # Create agent
    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=os.getenv("THENVOI_WS_URL", "wss://api.thenvoi.com/ws"),
        rest_url=os.getenv("THENVOI_REST_API_URL", "https://api.thenvoi.com"),
    )

    logger.info("Starting Claude SDK agent...")
    logger.info(f"  Agent ID: {agent_id}")
    logger.info(f"  Model: {claude_config.get('model', 'claude-sonnet-4-5-20250929')}")
    logger.info(f"  Custom tools: {len(custom_tools)}")
    logger.info("Press Ctrl+C to stop")

    try:
        await agent.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    asyncio.run(main())
