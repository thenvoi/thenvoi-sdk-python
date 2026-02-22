"""Integration-level tests for CodexAdapter lifecycle behavior."""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
from typing import Any

import pytest

from thenvoi.adapters.codex import CodexAdapter, CodexAdapterConfig
from thenvoi.core.types import AgentInput, HistoryProvider, PlatformMessage
from thenvoi.integrations.codex import RpcEvent
from thenvoi.testing import FakeAgentTools


def _platform_message(content: str, *, room_id: str = "room-1") -> PlatformMessage:
    return PlatformMessage(
        id=f"msg-{content[:8]}",
        room_id=room_id,
        content=content,
        sender_id="user-1",
        sender_type="User",
        sender_name="Alice",
        message_type="text",
        metadata={},
        created_at=datetime.now(timezone.utc),
    )


def _agent_input(
    content: str,
    tools: FakeAgentTools,
    *,
    raw_history: list[dict[str, Any]] | None = None,
    room_id: str = "room-1",
    is_session_bootstrap: bool = True,
) -> AgentInput:
    return AgentInput(
        msg=_platform_message(content, room_id=room_id),
        tools=tools,
        history=HistoryProvider(raw=raw_history or []),
        participants_msg=None,
        contacts_msg=None,
        is_session_bootstrap=is_session_bootstrap,
        room_id=room_id,
    )


class _ToolSchemaFakeTools(FakeAgentTools):
    def get_openai_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "thenvoi_send_message",
                    "description": "Send a message",
                    "parameters": {
                        "type": "object",
                        "properties": {"content": {"type": "string"}},
                    },
                },
            }
        ]


class _FakeCodexClient:
    def __init__(self, *, events: list[RpcEvent] | None = None) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.responses: list[tuple[int | str, dict[str, Any]]] = []
        self._events = deque(events or [])
        self._thread_counter = 0
        self._turn_counter = 0

    async def connect(self) -> None:
        return None

    async def initialize(
        self,
        *,
        client_name: str,
        client_title: str,
        client_version: str,
        experimental_api: bool = False,
        opt_out_notification_methods: list[str] | None = None,
    ) -> dict[str, Any]:
        return {"userAgent": f"{client_name}/{client_version}"}

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        retry_on_overload: bool = True,
    ) -> dict[str, Any]:
        payload = params or {}
        self.requests.append((method, payload))

        if method == "model/list":
            return {"data": [{"id": "gpt-5.3-codex", "hidden": False}]}
        if method == "thread/resume":
            return {"thread": {"id": str(payload.get("threadId") or "thr-resumed")}}
        if method == "thread/start":
            self._thread_counter += 1
            return {"thread": {"id": f"thr-{self._thread_counter}"}}
        if method == "turn/start":
            self._turn_counter += 1
            return {
                "turn": {
                    "id": f"turn-{self._turn_counter}",
                    "status": "inProgress",
                    "items": [],
                    "error": None,
                }
            }
        return {}

    async def recv_event(self, timeout_s: float | None = None) -> RpcEvent:
        if not self._events:
            raise asyncio.TimeoutError()
        return self._events.popleft()

    async def respond(self, request_id: int | str, result: dict[str, Any]) -> None:
        self.responses.append((request_id, result))

    async def respond_error(
        self,
        request_id: int | str,
        *,
        code: int,
        message: str,
        data: Any | None = None,
    ) -> None:
        self.responses.append(
            (request_id, {"error": {"code": code, "message": message}})
        )

    async def close(self) -> None:
        return None


def _notify(method: str, params: dict[str, Any]) -> RpcEvent:
    return RpcEvent(
        kind="notification",
        method=method,
        params=params,
        id=None,
        raw={"method": method, "params": params},
    )


def _request(request_id: int, method: str, params: dict[str, Any]) -> RpcEvent:
    return RpcEvent(
        kind="request",
        method=method,
        params=params,
        id=request_id,
        raw={"id": request_id, "method": method, "params": params},
    )


@pytest.mark.asyncio
async def test_on_event_uses_converter_history_to_resume_thread() -> None:
    tools = _ToolSchemaFakeTools()
    fake_client = _FakeCodexClient(
        events=[
            _notify(
                "turn/completed",
                {
                    "turn": {
                        "id": "turn-1",
                        "status": "completed",
                        "items": [],
                        "error": None,
                    }
                },
            )
        ]
    )
    adapter = CodexAdapter(
        config=CodexAdapterConfig(transport="ws"),
        client_factory=lambda _cfg: fake_client,
    )
    await adapter.on_started("Codex Agent", "Integration test agent")

    raw_history = [
        {
            "message_type": "task",
            "content": "historical mapping",
            "metadata": {
                "codex_thread_id": "thr-history",
                "codex_room_id": "room-1",
                "codex_created_at": datetime.now(timezone.utc).isoformat(),
            },
        }
    ]
    await adapter.on_event(
        _agent_input(
            "continue work",
            tools,
            raw_history=raw_history,
            room_id="room-1",
            is_session_bootstrap=True,
        )
    )

    methods = [method for method, _ in fake_client.requests]
    assert "thread/resume" in methods
    assert "thread/start" not in methods
    assert any("Status: resumed" in event["content"] for event in tools.events_sent)


@pytest.mark.asyncio
async def test_manual_approval_resolved_by_out_of_band_approve_command() -> None:
    tools = _ToolSchemaFakeTools()
    fake_client = _FakeCodexClient(
        events=[
            _request(
                7,
                "item/commandExecution/requestApproval",
                {"approvalId": "ap-1", "command": ["git", "push"]},
            ),
            _notify(
                "turn/completed",
                {
                    "turn": {
                        "id": "turn-1",
                        "status": "completed",
                        "items": [],
                        "error": None,
                    }
                },
            ),
        ]
    )
    adapter = CodexAdapter(
        config=CodexAdapterConfig(
            transport="ws",
            approval_mode="manual",
            approval_wait_timeout_s=30.0,
        ),
        client_factory=lambda _cfg: fake_client,
    )
    await adapter.on_started("Codex Agent", "Integration test agent")

    first_turn = asyncio.create_task(
        adapter.on_event(
            _agent_input(
                "run protected command",
                tools,
                room_id="room-1",
                is_session_bootstrap=True,
            )
        )
    )

    for _ in range(50):
        if any("Approval id: `ap-1`" in msg["content"] for msg in tools.messages_sent):
            break
        await asyncio.sleep(0.01)

    await adapter.on_event(
        _agent_input(
            "/approve ap-1",
            tools,
            room_id="room-1",
            is_session_bootstrap=False,
        )
    )
    await asyncio.wait_for(first_turn, timeout=2.0)

    assert (7, {"decision": "accept"}) in fake_client.responses
    assert any(
        "Approval `ap-1` resolved as `accept`." in msg["content"]
        for msg in tools.messages_sent
    )


@pytest.mark.asyncio
async def test_restart_rehydrates_mapping_from_previous_task_events() -> None:
    tools_first = _ToolSchemaFakeTools()
    fake_client_first = _FakeCodexClient(
        events=[
            _notify(
                "turn/completed",
                {
                    "turn": {
                        "id": "turn-1",
                        "status": "completed",
                        "items": [],
                        "error": None,
                    }
                },
            )
        ]
    )
    adapter_first = CodexAdapter(
        config=CodexAdapterConfig(transport="ws"),
        client_factory=lambda _cfg: fake_client_first,
    )
    await adapter_first.on_started("Codex Agent", "Integration test agent")
    await adapter_first.on_event(
        _agent_input(
            "first turn",
            tools_first,
            room_id="room-9",
            is_session_bootstrap=True,
        )
    )

    persisted_history = list(tools_first.events_sent)

    tools_second = _ToolSchemaFakeTools()
    fake_client_second = _FakeCodexClient(
        events=[
            _notify(
                "turn/completed",
                {
                    "turn": {
                        "id": "turn-1",
                        "status": "completed",
                        "items": [],
                        "error": None,
                    }
                },
            )
        ]
    )
    adapter_second = CodexAdapter(
        config=CodexAdapterConfig(transport="ws"),
        client_factory=lambda _cfg: fake_client_second,
    )
    await adapter_second.on_started("Codex Agent", "Integration test agent")
    await adapter_second.on_event(
        _agent_input(
            "second turn after restart",
            tools_second,
            raw_history=persisted_history,
            room_id="room-9",
            is_session_bootstrap=True,
        )
    )

    methods_second = [method for method, _ in fake_client_second.requests]
    assert "thread/resume" in methods_second
    assert "thread/start" not in methods_second
