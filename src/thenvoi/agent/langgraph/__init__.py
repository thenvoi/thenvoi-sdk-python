"""
LangGraph adapter for Thenvoi platform.

Quick start (SDK-managed agent):
    from thenvoi.agent.langgraph import create_langgraph_agent

    agent = await create_langgraph_agent(
        agent_id=agent_id,
        api_key=api_key,
        llm=ChatOpenAI(model="gpt-4o"),
        checkpointer=InMemorySaver(),
        ws_url=ws_url,
        thenvoi_restapi_url=thenvoi_restapi_url,
    )

Custom graph integration (user-provided graph):
    from thenvoi.agent.langgraph import connect_graph_to_platform

    my_graph = create_my_graph()  # Your compiled LangGraph
    agent = await connect_graph_to_platform(
        agent_id=agent_id,
        api_key=api_key,
        graph=my_graph,
        ws_url=ws_url,
        thenvoi_restapi_url=thenvoi_restapi_url,
    )

Advanced (class-based API):
    from thenvoi.agent.langgraph import ThenvoiLangGraphAgent, ConnectedGraphAgent

    agent = ThenvoiLangGraphAgent(...)
    await agent.start()

Utilities (for custom integration):
    from thenvoi.agent.langgraph import get_thenvoi_tools
    from thenvoi.agent.langgraph.message_formatters import (
        default_messages_state_formatter,
    )
"""

from .agent import (
    create_langgraph_agent,
    ThenvoiLangGraphAgent,
    connect_graph_to_platform,
    ConnectedGraphAgent,
)
from .tools import get_thenvoi_tools
from .graph_tools import graph_as_tool

__all__ = [
    "create_langgraph_agent",
    "ThenvoiLangGraphAgent",
    "connect_graph_to_platform",
    "ConnectedGraphAgent",
    "get_thenvoi_tools",
    "graph_as_tool",
]
