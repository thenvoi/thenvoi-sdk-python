#!/usr/bin/env python3
"""
Letta adapter with custom tools example.

Demonstrates adding custom tools to Letta agents using the CustomToolBuilder.

Usage:
    # Set environment variables
    export THENVOI_AGENT_ID="your-agent-id"
    export THENVOI_API_KEY="your-api-key"
    export LETTA_BASE_URL="http://localhost:8283"

    # Run
    uv run --extra letta python examples/letta/03_custom_tools.py
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime

from thenvoi import Agent
from thenvoi.adapters.letta import LettaAdapter, LettaConfig, LettaMode
from thenvoi.adapters.letta.tools import CustomToolBuilder
from thenvoi.runtime.types import SessionConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Custom Tools Definition
# ══════════════════════════════════════════════════════════════════════════════

# Create tool builder instance
tool_builder = CustomToolBuilder()


@tool_builder.tool
def calculate(operation: str, a: float, b: float) -> str:
    """
    Perform a mathematical calculation.

    Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide'
        a: First number
        b: Second number

    Returns:
        Result of the calculation
    """
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else "Error: division by zero",
    }

    if operation not in operations:
        return f"Error: Unknown operation '{operation}'. Use: add, subtract, multiply, divide"

    result = operations[operation](a, b)
    return f"{a} {operation} {b} = {result}"


@tool_builder.tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool_builder.tool
def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert temperature between Celsius, Fahrenheit, and Kelvin.

    Args:
        value: Temperature value to convert
        from_unit: Source unit ('C', 'F', or 'K')
        to_unit: Target unit ('C', 'F', or 'K')

    Returns:
        Converted temperature with units
    """
    # Normalize units
    from_unit = from_unit.upper()
    to_unit = to_unit.upper()

    # Convert to Celsius first
    if from_unit == "C":
        celsius = value
    elif from_unit == "F":
        celsius = (value - 32) * 5 / 9
    elif from_unit == "K":
        celsius = value - 273.15
    else:
        return f"Error: Unknown unit '{from_unit}'. Use C, F, or K."

    # Convert from Celsius to target
    if to_unit == "C":
        result = celsius
    elif to_unit == "F":
        result = celsius * 9 / 5 + 32
    elif to_unit == "K":
        result = celsius + 273.15
    else:
        return f"Error: Unknown unit '{to_unit}'. Use C, F, or K."

    return f"{value}°{from_unit} = {result:.2f}°{to_unit}"


@tool_builder.tool
def generate_random_number(min_val: int, max_val: int) -> str:
    """
    Generate a random integer between min and max (inclusive).

    Args:
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)

    Returns:
        A random number in the specified range
    """
    import random

    if min_val > max_val:
        return f"Error: min ({min_val}) must be <= max ({max_val})"
    result = random.randint(min_val, max_val)
    return f"Random number between {min_val} and {max_val}: {result}"


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


async def main():
    # Load configuration from environment
    agent_id = os.environ.get("THENVOI_AGENT_ID")
    api_key = os.environ.get("THENVOI_API_KEY")
    letta_base_url = os.environ.get("LETTA_BASE_URL", "http://localhost:8283")

    if not agent_id or not api_key:
        raise ValueError(
            "Missing required environment variables: "
            "THENVOI_AGENT_ID and THENVOI_API_KEY"
        )

    # Log registered tools
    logger.info("Registered custom tools:")
    for name in tool_builder.get_tool_names():
        logger.info(f"  - {name}")

    # Configure Letta adapter with custom tools
    adapter = LettaAdapter(
        config=LettaConfig(
            # api_key is optional for self-hosted Letta
            mode=LettaMode.PER_ROOM,
            base_url=letta_base_url,
            model="openai/gpt-4o-mini",
            embedding_model="openai/text-embedding-3-small",
            persona="""You are a helpful assistant with calculation and utility capabilities.

Available tools:
- calculate: Perform math operations (add, subtract, multiply, divide)
- get_current_time: Get the current date and time
- convert_temperature: Convert between Celsius, Fahrenheit, and Kelvin
- generate_random_number: Generate random integers

When users ask for calculations or conversions, use the appropriate tool.""",
            # Pass custom tool definitions
            custom_tools=tool_builder.get_tool_definitions(),
        ),
        state_storage_path="~/.thenvoi/letta_tools_state.json",
    )

    # Create Thenvoi agent
    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        session_config=SessionConfig(enable_context_hydration=False),
    )

    logger.info("=" * 60)
    logger.info("Starting Letta agent with custom tools")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Try asking:")
    logger.info('  - "What is 15 multiplied by 7?"')
    logger.info('  - "What time is it?"')
    logger.info('  - "Convert 100 degrees Fahrenheit to Celsius"')
    logger.info('  - "Give me a random number between 1 and 100"')
    logger.info("")
    logger.info(f"Letta server: {letta_base_url}")
    logger.info("")

    # Run the agent
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
