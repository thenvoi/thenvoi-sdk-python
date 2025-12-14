"""
LangGraph adapter for Thenvoi platform.

NEW ARCHITECTURE (Recommended):
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

LEGACY API (deprecated but still works):
    from thenvoi.agent.langgraph import create_langgraph_agent

    agent = await create_langgraph_agent(
        agent_id=agent_id,
        api_key=api_key,
        llm=ChatOpenAI(model="gpt-4o"),
        checkpointer=InMemorySaver(),
        ws_url=ws_url,
        thenvoi_restapi_url=thenvoi_restapi_url,
    )

Utilities (for custom integration):
    from thenvoi.agent.langgraph import get_thenvoi_tools
"""

# New architecture (recommended)
from .adapter import (
    LangGraphAdapter,
    LangGraphMCPAdapter,
    with_langgraph,
)

# Legacy API (deprecated but kept for backwards compatibility)
from .agent import (
    create_langgraph_agent,
    ThenvoiLangGraphAgent,
    connect_graph_to_platform,
    ConnectedGraphAgent,
)
from .tools import get_thenvoi_tools
from .graph_tools import graph_as_tool

__all__ = [
    # New architecture
    "LangGraphAdapter",
    "LangGraphMCPAdapter",
    "with_langgraph",
    # Legacy (deprecated)
    "create_langgraph_agent",
    "ThenvoiLangGraphAgent",
    "connect_graph_to_platform",
    "ConnectedGraphAgent",
    # Utilities
    "get_thenvoi_tools",
    "graph_as_tool",
]
