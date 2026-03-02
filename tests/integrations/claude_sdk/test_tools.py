"""Unit tests for Claude SDK MCP tool server wiring."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

import thenvoi.integrations.claude_sdk.tools as tools_module


class _FakeExecution:
    def __init__(self, participants: list[dict[str, Any]] | None = None) -> None:
        self.participants = participants or []

    def add_participant(self, participant: dict[str, Any]) -> None:
        self.participants.append(participant)

    def remove_participant(self, participant_id: str) -> None:
        self.participants = [
            p for p in self.participants if p.get("id") != participant_id
        ]


class _FakeAgentTools:
    mode = "ok"

    def __init__(
        self,
        room_id: str,
        rest: object,
        participants: list[dict[str, Any]],
    ) -> None:
        self.room_id = room_id
        self.rest = rest
        self.participants = participants

    async def send_message(self, content: str, mentions: list[str]) -> dict[str, Any]:
        if _FakeAgentTools.mode == "mention_error":
            raise ValueError("Unknown participant 'missing'")
        return {"content": content, "mentions": mentions}

    async def send_event(self, content: str, message_type: str) -> dict[str, Any]:
        return {"content": content, "message_type": message_type}

    async def add_participant(self, name: str, role: str) -> dict[str, Any]:
        return {"id": "p-123", "name": name, "role": role}

    async def remove_participant(self, name: str) -> dict[str, Any]:
        return {"id": "p-123", "name": name}

    async def get_participants(self) -> list[dict[str, Any]]:
        return [{"id": "p-1", "name": "Alice"}]

    async def lookup_peers(self, page: int, page_size: int) -> dict[str, Any]:
        return {
            "peers": [{"id": "peer-1", "name": "Weather"}],
            "metadata": {"page": page, "page_size": page_size, "total_pages": 1},
        }

    async def create_chatroom(self, task_id: str | None = None) -> str:
        return task_id or "room-new"

    async def execute_tool_call(self, tool_name: str, args: dict[str, Any]) -> Any:
        if tool_name == "thenvoi_send_message":
            return await self.send_message(
                content=args["content"],
                mentions=args.get("mentions", []),
            )
        if tool_name == "thenvoi_send_event":
            return await self.send_event(
                content=args["content"],
                message_type=args["message_type"],
            )
        if tool_name == "thenvoi_add_participant":
            return await self.add_participant(
                name=args["name"],
                role=args.get("role", "member"),
            )
        if tool_name == "thenvoi_remove_participant":
            return await self.remove_participant(name=args["name"])
        if tool_name == "thenvoi_get_participants":
            return await self.get_participants()
        if tool_name == "thenvoi_lookup_peers":
            return await self.lookup_peers(
                page=args["page"],
                page_size=args["page_size"],
            )
        if tool_name == "thenvoi_create_chatroom":
            return await self.create_chatroom(task_id=args.get("task_id"))
        raise ValueError(f"Unknown tool: {tool_name}")


def _parse_content_payload(result: dict[str, Any]) -> dict[str, Any]:
    assert "content" in result
    text = result["content"][0]["text"]
    return json.loads(text)


@pytest.fixture
def patched_tools_runtime(monkeypatch: pytest.MonkeyPatch):
    _FakeAgentTools.mode = "ok"

    def fake_tool(name: str, _description: str, _schema: dict[str, type]):
        def _decorator(func):
            func.__name__ = name
            return func

        return _decorator

    def fake_create_sdk_mcp_server(
        *,
        name: str,
        version: str,
        tools: list[Any],
    ) -> dict[str, Any]:
        return {"name": name, "version": version, "tools": tools}

    monkeypatch.setattr(tools_module, "tool", fake_tool)
    monkeypatch.setattr(tools_module, "create_sdk_mcp_server", fake_create_sdk_mcp_server)
    monkeypatch.setattr("thenvoi.runtime.tools.AgentTools", _FakeAgentTools)

    execution = _FakeExecution(
        participants=[{"id": "u-1", "name": "Bob", "handle": "@bob"}]
    )
    agent = SimpleNamespace(
        link=SimpleNamespace(rest=object()),
        runtime=SimpleNamespace(executions={"room-1": execution}),
    )
    return agent, execution


def test_thenvoi_tools_alias_emits_deprecation_warning() -> None:
    with pytest.warns(DeprecationWarning, match="THENVOI_TOOLS is deprecated"):
        legacy_tools = tools_module.THENVOI_TOOLS

    assert legacy_tools == tools_module.THENVOI_CHAT_TOOLS


@pytest.mark.asyncio
async def test_create_server_registers_all_chat_tools(patched_tools_runtime) -> None:
    agent, _ = patched_tools_runtime

    server = tools_module.create_thenvoi_mcp_server(agent)

    assert server["name"] == "thenvoi"
    assert server["version"] == "1.0.0"
    registered_names = {tool_fn.__name__ for tool_fn in server["tools"]}
    assert registered_names == {
        "thenvoi_send_message",
        "thenvoi_send_event",
        "thenvoi_add_participant",
        "thenvoi_remove_participant",
        "thenvoi_get_participants",
        "thenvoi_lookup_peers",
        "thenvoi_create_chatroom",
    }


@pytest.mark.asyncio
async def test_send_message_reports_available_handles_on_mention_error(
    patched_tools_runtime,
) -> None:
    agent, _ = patched_tools_runtime
    _FakeAgentTools.mode = "mention_error"

    server = tools_module.create_thenvoi_mcp_server(agent)
    handlers = {tool_fn.__name__: tool_fn for tool_fn in server["tools"]}

    result = await handlers["thenvoi_send_message"](
        {"room_id": "room-1", "content": "Hi", "mentions": "[\"@missing\"]"}
    )

    assert result["is_error"] is True
    payload = _parse_content_payload(result)
    assert payload["status"] == "error"
    assert "Available handles" in payload["message"]
    assert "@bob" in payload["message"]


@pytest.mark.asyncio
async def test_add_participant_updates_execution_cache(patched_tools_runtime) -> None:
    agent, execution = patched_tools_runtime
    server = tools_module.create_thenvoi_mcp_server(agent)
    handlers = {tool_fn.__name__: tool_fn for tool_fn in server["tools"]}

    result = await handlers["thenvoi_add_participant"](
        {"room_id": "room-1", "name": "Alice", "role": "member"}
    )

    payload = _parse_content_payload(result)
    assert payload["status"] == "success"
    assert payload["name"] == "Alice"
    assert any(p.get("name") == "Alice" for p in execution.participants)
