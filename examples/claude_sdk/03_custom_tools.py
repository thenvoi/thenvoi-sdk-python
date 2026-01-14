#!/usr/bin/env python3
"""
Custom Tools Claude SDK Agent Example.

This example shows how to create an agent with custom MCP tools that Claude
can use alongside the built-in Thenvoi platform tools.

Custom tools allow you to:
- Add domain-specific capabilities
- Integrate with external APIs
- Perform calculations or data processing
- Access local resources

Prerequisites:
    1. Node.js 20+ installed
    2. Claude Code CLI: npm install -g @anthropic-ai/claude-code
    3. Configure agent_config.yaml with claude_sdk_custom_tools credentials
    4. Set ANTHROPIC_API_KEY environment variable

Usage:
    python 03_custom_tools.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

# Add examples directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

from setup_logging import setup_logging
from thenvoi import Agent
from thenvoi.adapters import ClaudeSDKAdapter
from thenvoi.config import load_agent_config

# Import the @tool decorator from claude_agent_sdk
from claude_agent_sdk import tool

setup_logging()
logger = logging.getLogger(__name__)


# =============================================================================
# Define Custom Tools
# =============================================================================
# Custom tools use the @tool decorator from claude_agent_sdk.
# Each tool needs:
#   - name: Unique identifier for the tool
#   - description: What the tool does (shown to Claude)
#   - parameters: Dict of parameter names to types
#
# Tools must return a dict with "content" key containing a list of content blocks.
# =============================================================================


@tool(
    "calculator",
    "Perform mathematical calculations. Supports basic arithmetic (+, -, *, /), "
    "powers (**), and common math functions.",
    {"expression": str},
)
async def calculator(args: dict) -> dict:
    """
    Calculate mathematical expressions.

    Example expressions:
    - "2 + 2"
    - "10 * 5"
    - "2 ** 8"
    - "100 / 4"
    """
    try:
        expression = args.get("expression", "")
        # WARNING: eval is dangerous in production! Use a safe math parser instead.
        # This is for demonstration only.
        result = eval(expression, {"__builtins__": {}}, {})
        return {"content": [{"type": "text", "text": f"Result: {result}"}]}
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error: {e}"}],
            "is_error": True,
        }


@tool(
    "get_current_time",
    "Get the current date and time in various formats.",
    {"format": str},
)
async def get_current_time(args: dict) -> dict:
    """
    Get current time.

    Format options:
    - "iso": ISO 8601 format
    - "date": Date only (YYYY-MM-DD)
    - "time": Time only (HH:MM:SS)
    - "full": Full human-readable format
    """
    from datetime import datetime

    try:
        fmt = args.get("format", "iso").lower()
        now = datetime.now()

        if fmt == "iso":
            result = now.isoformat()
        elif fmt == "date":
            result = now.strftime("%Y-%m-%d")
        elif fmt == "time":
            result = now.strftime("%H:%M:%S")
        elif fmt == "full":
            result = now.strftime("%A, %B %d, %Y at %I:%M %p")
        else:
            result = now.isoformat()

        return {"content": [{"type": "text", "text": result}]}
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error: {e}"}],
            "is_error": True,
        }


@tool(
    "random_number",
    "Generate a random number within a specified range.",
    {"min": int, "max": int},
)
async def random_number(args: dict) -> dict:
    """Generate a random integer between min and max (inclusive)."""
    import random

    try:
        min_val = args.get("min", 1)
        max_val = args.get("max", 100)
        result = random.randint(min_val, max_val)
        return {"content": [{"type": "text", "text": str(result)}]}
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error: {e}"}],
            "is_error": True,
        }


# =============================================================================
# Main Agent
# =============================================================================


async def main():
    """Run the Claude SDK agent with custom tools."""
    load_dotenv()

    # Load URLs from environment
    ws_url = os.getenv("THENVOI_WS_URL")
    rest_url = os.getenv("THENVOI_REST_URL")

    if not ws_url:
        raise ValueError("THENVOI_WS_URL environment variable is required")
    if not rest_url:
        raise ValueError("THENVOI_REST_URL environment variable is required")

    # Load agent credentials from agent_config.yaml
    agent_id, api_key = load_agent_config("claude_sdk_custom_tools")

    # Create adapter with custom tools
    adapter = ClaudeSDKAdapter(
        model="claude-sonnet-4-5-20250929",
        custom_section="""You are a helpful assistant with access to custom tools.

Available custom tools:
- calculator: Perform math calculations
- get_current_time: Get current date/time
- random_number: Generate random numbers

Use these tools when appropriate to help users.""",
        custom_tools=[
            calculator,
            get_current_time,
            random_number,
        ],
        enable_execution_reporting=True,
    )

    # Create and start agent
    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    logger.info("Starting Claude SDK agent with custom tools...")
    logger.info(f"Agent ID: {agent_id}")
    logger.info("Custom tools: calculator, get_current_time, random_number")
    logger.info("Press Ctrl+C to stop")

    try:
        await agent.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    asyncio.run(main())
