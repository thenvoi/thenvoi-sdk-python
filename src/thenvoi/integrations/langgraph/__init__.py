"""
LangGraph integration for Thenvoi SDK.

This module provides:
- ThenvoiLangGraphAgent: Main adapter class for LangGraph
- create_langgraph_agent: Convenience function to create and run agent
- agent_tools_to_langchain: Convert AgentTools to LangChain tool format
- graph_as_tool: Wrap a LangGraph as a callable tool
- MessageFormatter: Protocol for message formatting
"""

from .langchain_tools import agent_tools_to_langchain
from .graph_tools import graph_as_tool
from .message_formatters import MessageFormatter, default_messages_state_formatter
from .agent import (
    ThenvoiLangGraphAgent,
    ThenvoiLangGraphMCPAgent,
    create_langgraph_agent,
)

__all__ = [
    # Adapter classes
    "ThenvoiLangGraphAgent",
    "ThenvoiLangGraphMCPAgent",
    "create_langgraph_agent",
    # Utilities
    "agent_tools_to_langchain",
    "graph_as_tool",
    "MessageFormatter",
    "default_messages_state_formatter",
]
