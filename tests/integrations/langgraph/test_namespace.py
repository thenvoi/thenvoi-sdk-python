"""Tests for LangGraph integration namespace exports."""

from __future__ import annotations

import thenvoi.integrations.langgraph as langgraph_namespace


def test_namespace_exports_are_lazy_resolvable() -> None:
    formatter = langgraph_namespace.default_messages_state_formatter
    converter = langgraph_namespace.agent_tools_to_langchain
    assert callable(formatter)
    assert callable(converter)


def test_namespace_dir_includes_public_exports() -> None:
    names = dir(langgraph_namespace)
    assert "graph_as_tool" in names
    assert "MessageFormatter" in names

