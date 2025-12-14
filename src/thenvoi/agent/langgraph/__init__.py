"""
LangGraph adapter for Thenvoi platform.

Usage:
    from thenvoi.agent.langgraph import LangGraphAdapter, with_langgraph

    # With graph_factory (recommended - gets Thenvoi tools)
    adapter = await with_langgraph(
        graph_factory=lambda tools: create_react_agent(ChatOpenAI(), tools),
        agent_id="...",
        api_key="...",
    )

    # Or using class directly
    adapter = LangGraphAdapter(
        graph_factory=my_graph_factory,
        agent_id="...",
        api_key="...",
    )
    await adapter.run()

Utilities (for custom integration):
    from thenvoi.agent.langgraph import get_thenvoi_tools
"""

from .adapter import (
    LangGraphAdapter,
    LangGraphMCPAdapter,
    with_langgraph,
)
from .tools import get_thenvoi_tools
from .graph_tools import graph_as_tool

__all__ = [
    # Adapter
    "LangGraphAdapter",
    "LangGraphMCPAdapter",
    "with_langgraph",
    # Utilities
    "get_thenvoi_tools",
    "graph_as_tool",
]
