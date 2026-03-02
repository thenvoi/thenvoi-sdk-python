"""
Custom tools for your agent.

Example tools (calculator, get_time, random_number) are available here.
To use them, uncomment the tools section in your agent.yaml:
    tools:
      - calculator
      - get_time
      - random_number

To create your own tools, define callables in this package and register them
in TOOL_REGISTRY below.
"""

from __future__ import annotations

from thenvoi.testing.example_tools import (
    build_example_tool_registry,
    calculator as calculator,
    get_time as get_time,
    random_number as random_number,
)

TOOL_REGISTRY: dict[str, object] = build_example_tool_registry()

__all__ = ["TOOL_REGISTRY", "calculator", "get_time", "random_number"]
