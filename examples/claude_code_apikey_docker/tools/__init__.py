"""
Custom tools for your agent.

To use example tools, uncomment the imports below.
To create your own, copy from example_tools.py or write new ones.

Then enable in agent.yaml:
    tools:
      - calculator
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from .example_tools import calculator, get_time, random_number

ToolFunction = Callable[..., Awaitable[dict[str, Any]]]

TOOL_REGISTRY: dict[str, ToolFunction] = {
    "calculator": calculator,
    "get_time": get_time,
    "random_number": random_number,
}
