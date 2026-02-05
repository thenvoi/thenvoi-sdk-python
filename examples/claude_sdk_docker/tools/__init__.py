"""
Custom tools for your agent.

Example tools (calculator, get_time, random_number) are available here.
To use them, uncomment the tools section in your agent.yaml:
    tools:
      - calculator
      - get_time
      - random_number

To create your own tools, add them to example_tools.py and register them
in TOOL_REGISTRY below.
"""

from __future__ import annotations

from .example_tools import calculator, get_time, random_number

TOOL_REGISTRY: dict[str, object] = {
    "calculator": calculator,
    "get_time": get_time,
    "random_number": random_number,
}
