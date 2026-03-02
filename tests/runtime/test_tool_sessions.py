"""Tests for tool session registry helpers."""

from __future__ import annotations

import json

from thenvoi.runtime.tool_sessions import (
    ToolSessionRegistry,
    mcp_text_error,
    mcp_text_result,
)


def test_registry_tracks_tools_and_message_flags() -> None:
    registry: ToolSessionRegistry[object] = ToolSessionRegistry()
    tools = object()

    registry.set_tools("session-1", tools)

    assert registry.get_tools("session-1") is tools
    assert registry.was_message_sent("session-1") is False
    assert registry.active_sessions() == ["session-1"]

    registry.mark_message_sent("session-1")
    assert registry.was_message_sent("session-1") is True

    registry.set_tools("session-1", None)
    assert registry.get_tools("session-1") is None
    assert registry.was_message_sent("session-1") is False
    assert registry.active_sessions() == []


def test_mcp_text_result_wraps_data_as_json_text() -> None:
    result = mcp_text_result({"status": "ok", "count": 2})

    assert result["content"][0]["type"] == "text"
    assert json.loads(result["content"][0]["text"]) == {"status": "ok", "count": 2}


def test_mcp_text_error_wraps_message_as_error_payload() -> None:
    result = mcp_text_error("bad input")

    assert result["is_error"] is True
    assert result["content"][0]["type"] == "text"
    assert json.loads(result["content"][0]["text"]) == {
        "status": "error",
        "message": "bad input",
    }
