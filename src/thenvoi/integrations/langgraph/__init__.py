"""LangGraph integration public namespace."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = (
    "agent_tools_to_langchain",
    "graph_as_tool",
    "MessageFormatter",
    "default_messages_state_formatter",
)

_EXPORT_MODULES: dict[str, str] = {
    "agent_tools_to_langchain": "thenvoi.integrations.langgraph.langchain_tools",
    "graph_as_tool": "thenvoi.integrations.langgraph.graph_tools",
    "MessageFormatter": "thenvoi.integrations.langgraph.message_formatters",
    "default_messages_state_formatter": "thenvoi.integrations.langgraph.message_formatters",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
