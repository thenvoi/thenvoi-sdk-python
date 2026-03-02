"""Tests for Claude SDK tool state helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from thenvoi.integrations.claude_sdk import tool_state


def _make_agent(
    *,
    execution: object | None = None,
    room_id: str = "room-1",
) -> SimpleNamespace:
    executions = {room_id: execution} if execution is not None else {}
    return SimpleNamespace(
        runtime=SimpleNamespace(executions=executions),
        link=SimpleNamespace(rest=MagicMock()),
    )


def test_get_execution_returns_execution_or_none() -> None:
    execution = MagicMock()
    agent = _make_agent(execution=execution)

    assert tool_state.get_execution(agent, "room-1") is execution
    assert tool_state.get_execution(agent, "room-2") is None


def test_get_tools_builds_agent_tools_with_execution_participants(
    monkeypatch,
) -> None:
    execution = SimpleNamespace(participants=[{"id": "u-1", "name": "Alice"}])
    agent = _make_agent(execution=execution)
    fake_tools = object()
    agent_tools_cls = MagicMock(return_value=fake_tools)
    monkeypatch.setattr("thenvoi.runtime.tools.AgentTools", agent_tools_cls)

    tools = tool_state.get_tools(agent, "room-1")

    assert tools is fake_tools
    agent_tools_cls.assert_called_once_with(
        "room-1", agent.link.rest, execution.participants
    )


def test_get_participant_handles_filters_missing_values() -> None:
    execution = SimpleNamespace(
        participants=[
            {"id": "u-1", "handle": "@alice"},
            {"id": "u-2"},
            {"id": "u-3", "handle": ""},
            {"id": "u-4", "handle": "@bob"},
        ]
    )
    agent = _make_agent(execution=execution)

    handles = tool_state.get_participant_handles(agent, "room-1")

    assert handles == ["@alice", "@bob"]


def test_parse_mention_handles_supports_list_json_and_invalid_strings() -> None:
    assert tool_state.parse_mention_handles(["@alice", "@bob"]) == ["@alice", "@bob"]
    assert tool_state.parse_mention_handles('["@alice", "@bob"]') == ["@alice", "@bob"]
    assert tool_state.parse_mention_handles("not-json") == []
    assert tool_state.parse_mention_handles(None) == []


def test_update_participants_cache_helpers_delegate_to_execution() -> None:
    execution = MagicMock()
    agent = _make_agent(execution=execution)

    tool_state.update_participants_cache_for_add(
        agent,
        "room-1",
        {"id": "u-1", "name": "Alice"},
    )
    tool_state.update_participants_cache_for_remove(agent, "room-1", "u-1")

    execution.add_participant.assert_called_once_with(
        {"id": "u-1", "name": "Alice", "type": "Agent"}
    )
    execution.remove_participant.assert_called_once_with("u-1")
