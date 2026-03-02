"""Tests for LangGraph tool-conversion helpers."""

from __future__ import annotations

import thenvoi.integrations.langgraph.langchain_tools as langchain_tools_module


def test_agent_tools_to_langchain_respects_platform_tool_order(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        langchain_tools_module,
        "get_platform_tool_order",
        lambda include_memory_tools=False: ["thenvoi_send_message"],
    )

    tools = langchain_tools_module.agent_tools_to_langchain(
        tools=object(),
        include_memory_tools=False,
    )
    assert len(tools) == 1
    assert tools[0].name == "thenvoi_send_message"

