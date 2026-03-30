"""Tests for CodexAdapter."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from datetime import datetime
from typing import Any
from uuid import uuid4

import pytest

from pydantic import BaseModel

from thenvoi.adapters.codex import CodexAdapter, CodexAdapterConfig
from thenvoi.core.types import AgentInput, HistoryProvider, PlatformMessage
from thenvoi.integrations.codex import CodexJsonRpcError, RpcEvent
from thenvoi.integrations.codex.types import CodexSessionState
from thenvoi.runtime.custom_tools import CustomToolDef
from thenvoi.testing import FakeAgentTools


def make_platform_message(
    room_id: str = "room-1", content: str = "hello"
) -> PlatformMessage:
    return PlatformMessage(
        id=str(uuid4()),
        room_id=room_id,
        content=content,
        sender_id="user-1",
        sender_type="User",
        sender_name="Alice",
        message_type="text",
        metadata={},
        created_at=datetime.now(),
    )


class ToolSchemaFakeTools(FakeAgentTools):
    def get_openai_tool_schemas(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "thenvoi_send_message",
                    "description": "Send a message",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "mentions": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["content", "mentions"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "thenvoi_send_event",
                    "description": "Send an event",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "message_type": {"type": "string"},
                        },
                        "required": ["content", "message_type"],
                    },
                },
            },
        ]


class FakeCodexClient:
    """Minimal fake transport client for adapter tests."""

    def __init__(
        self,
        *,
        events: list[RpcEvent] | None = None,
        resume_error: Exception | None = None,
        turn_start_error: Exception | None = None,
        turn_start_error_once: bool = True,
        model_list_result: dict[str, Any] | None = None,
    ) -> None:
        self.connected = False
        self.initialized = False
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.responses: list[tuple[int | str, dict[str, Any]]] = []
        self.response_errors: list[tuple[int | str, int, str]] = []
        self.closed = False
        self._events = deque(events or [])
        self._resume_error = resume_error
        self._turn_start_error = turn_start_error
        self._turn_start_error_once = turn_start_error_once
        self._model_list_result = model_list_result
        self._thread_counter = 0
        self._turn_counter = 0

    async def connect(self) -> None:
        self.connected = True

    async def initialize(
        self,
        *,
        client_name: str,
        client_title: str,
        client_version: str,
        experimental_api: bool = False,
        opt_out_notification_methods: list[str] | None = None,
    ) -> dict[str, Any]:
        self.initialized = True
        return {"userAgent": f"{client_name}/{client_version}"}

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        retry_on_overload: bool = True,
    ) -> dict[str, Any]:
        payload = params or {}
        self.requests.append((method, dict(payload)))

        if method == "model/list":
            if self._model_list_result is not None:
                return self._model_list_result
            return {"data": [{"id": "gpt-5.3-codex", "hidden": False}]}

        if method == "thread/resume":
            if self._resume_error is not None:
                raise self._resume_error
            return {"thread": {"id": payload.get("threadId", "thr-resumed")}}

        if method == "thread/start":
            self._thread_counter += 1
            return {"thread": {"id": f"thr-{self._thread_counter}"}}

        if method == "turn/start":
            if self._turn_start_error is not None:
                err = self._turn_start_error
                if self._turn_start_error_once:
                    self._turn_start_error = None
                raise err
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
            raise asyncio.TimeoutError
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
        self.response_errors.append((request_id, code, message))

    async def close(self) -> None:
        self.closed = True
        return None


def _event_notification(method: str, params: dict[str, Any]) -> RpcEvent:
    return RpcEvent(
        kind="notification",
        method=method,
        params=params,
        id=None,
        raw={"method": method, "params": params},
    )


def _event_request(request_id: int, method: str, params: dict[str, Any]) -> RpcEvent:
    return RpcEvent(
        kind="request",
        method=method,
        params=params,
        id=request_id,
        raw={"id": request_id, "method": method, "params": params},
    )


class TestCodexAdapter:
    def test_config_defaults_are_low_noise_and_manual_approval(self) -> None:
        config = CodexAdapterConfig()
        assert config.emit_turn_task_markers is False
        assert config.emit_thought_events is False
        assert config.approval_mode == "manual"

    @pytest.mark.asyncio
    async def test_bootstrap_starts_thread_and_sends_fallback_message(self) -> None:
        events = [
            _event_notification(
                "item/agentMessage/delta",
                {"itemId": "msg-1", "delta": "harness-ok"},
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        assert any(method == "thread/start" for method, _ in fake_client.requests)
        thread_start = next(
            params
            for method, params in fake_client.requests
            if method == "thread/start"
        )
        assert "dynamicTools" in thread_start
        dynamic_names = [t["name"] for t in thread_start["dynamicTools"]]
        assert "thenvoi_send_message" in dynamic_names
        assert "thenvoi_send_event" in dynamic_names

        assert len(tools.messages_sent) == 1
        assert tools.messages_sent[0]["content"] == "harness-ok"
        assert tools.messages_sent[0]["mentions"][0]["id"] == "user-1"

    @pytest.mark.asyncio
    async def test_system_prompt_retry_after_turn_start_failure(self) -> None:
        """System instructions stay pending until turn/start succeeds."""
        events = [
            _event_notification(
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
        fake_client = FakeCodexClient(
            events=events,
            turn_start_error=CodexJsonRpcError(
                code=-32000,
                message="Model not available",
            ),
            turn_start_error_once=True,
        )
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", model="gpt-5.3-codex"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")

        with pytest.raises(CodexJsonRpcError, match="not available"):
            await adapter.on_message(
                make_platform_message(room_id="room-1", content="first try"),
                tools,
                CodexSessionState(),
                participants_msg=None,
                contacts_msg=None,
                is_session_bootstrap=True,
                room_id="room-1",
            )
        assert "room-1" not in adapter._prompt_injected_rooms

        await adapter.on_message(
            make_platform_message(room_id="room-1", content="second try"),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=False,
            room_id="room-1",
        )
        assert "room-1" in adapter._prompt_injected_rooms

        turn_inputs = [
            params["input"]
            for method, params in fake_client.requests
            if method == "turn/start"
        ]
        assert len(turn_inputs) == 2
        for turn_input in turn_inputs:
            assert any(
                item.get("text", "").startswith("[System Instructions]\n")
                for item in turn_input
            )

    @pytest.mark.asyncio
    async def test_tool_call_request_is_dispatched_and_responded(self) -> None:
        events = [
            _event_request(
                42,
                "item/tool/call",
                {
                    "tool": "thenvoi_lookup_peers",
                    "arguments": {"page": 1, "page_size": 10},
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        assert len(tools.tool_calls) == 1
        assert tools.tool_calls[0]["tool_name"] == "thenvoi_lookup_peers"
        assert fake_client.responses
        response_id, response_payload = fake_client.responses[0]
        assert response_id == 42
        assert response_payload["success"] is True

    @pytest.mark.asyncio
    async def test_fallback_text_not_suppressed_when_send_message_tool_fails(
        self,
    ) -> None:
        """Fallback agent text should still be delivered when send_message fails."""

        class SendMessageFailureTools(ToolSchemaFakeTools):
            async def execute_tool_call(
                self, tool_name: str, arguments: dict[str, Any]
            ) -> Any:
                call = {"tool_name": tool_name, "arguments": arguments}
                self.tool_calls.append(call)
                if tool_name == "thenvoi_send_message":
                    raise RuntimeError("send failed")
                return {"status": "ok"}

        events = [
            _event_request(
                77,
                "item/tool/call",
                {
                    "tool": "thenvoi_send_message",
                    "arguments": {"content": "hi"},
                    "callId": "call-77",
                },
            ),
            _event_notification(
                "item/completed",
                {
                    "item": {
                        "type": "agentMessage",
                        "id": "msg-1",
                        "text": "fallback final text",
                    }
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = SendMessageFailureTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        assert len(tools.messages_sent) == 1
        assert tools.messages_sent[0]["content"] == "fallback final text"
        assert len(fake_client.responses) == 1
        _, payload = fake_client.responses[0]
        assert payload["success"] is False

    @pytest.mark.asyncio
    async def test_resume_failure_falls_back_to_thread_start(self) -> None:
        events = [
            _event_notification(
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
        fake_client = FakeCodexClient(
            events=events,
            resume_error=CodexJsonRpcError(code=-32002, message="Not found"),
        )
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(thread_id="thr-old", room_id="room-1"),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        methods = [method for method, _ in fake_client.requests]
        assert "thread/resume" in methods
        assert "thread/start" in methods

    @pytest.mark.asyncio
    async def test_approval_request_auto_decline(self) -> None:
        events = [
            _event_request(
                7,
                "item/commandExecution/requestApproval",
                {"command": "rm -rf tmp"},
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(
                transport="ws",
                approval_mode="auto_decline",
            ),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        assert fake_client.responses
        response_id, payload = fake_client.responses[0]
        assert response_id == 7
        assert payload["decision"] == "decline"
        assert len(tools.messages_sent) == 1
        assert "Approval requested" in tools.messages_sent[0]["content"]
        assert "rm -rf tmp" in tools.messages_sent[0]["content"]

    @pytest.mark.asyncio
    async def test_auto_approval_responds_even_if_notification_fails(self) -> None:
        class FailingNotifyTools(ToolSchemaFakeTools):
            async def send_message(
                self, content: str, mentions: list[dict[str, str]] | None = None
            ) -> Any:
                raise RuntimeError("notification failed")

        events = [
            _event_request(
                7,
                "item/commandExecution/requestApproval",
                {"command": "rm -rf tmp"},
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(
                transport="ws",
                approval_mode="auto_decline",
                approval_text_notifications=True,
            ),
            client_factory=lambda _config: fake_client,
        )
        tools = FailingNotifyTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        assert fake_client.responses
        response_id, payload = fake_client.responses[0]
        assert response_id == 7
        assert payload["decision"] == "decline"

    @pytest.mark.asyncio
    async def test_manual_approval_responds_with_decline_if_notification_fails(
        self,
    ) -> None:
        class FailingNotifyTools(ToolSchemaFakeTools):
            async def send_message(
                self, content: str, mentions: list[dict[str, str]] | None = None
            ) -> Any:
                raise RuntimeError("notification failed")

        events = [
            _event_request(
                7,
                "item/commandExecution/requestApproval",
                {"command": "rm -rf tmp"},
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(
                transport="ws",
                approval_mode="manual",
            ),
            client_factory=lambda _config: fake_client,
        )
        tools = FailingNotifyTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        assert fake_client.responses
        response_id, payload = fake_client.responses[0]
        assert response_id == 7
        assert payload["decision"] == "decline"
        assert "room-1" not in adapter._pending_approvals

    @pytest.mark.asyncio
    async def test_cleanup_closes_client_when_last_room_removed(self) -> None:
        fake_client = FakeCodexClient(
            events=[
                _event_notification(
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
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(room_id="room-1"),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        assert fake_client.closed is False
        await adapter.on_cleanup("room-1")
        assert fake_client.closed is True

    @pytest.mark.asyncio
    async def test_cleanup_idempotent(self) -> None:
        """Calling on_cleanup twice for the same room should not raise."""
        fake_client = FakeCodexClient(
            events=[
                _event_notification(
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
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(room_id="room-1"),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        await adapter.on_cleanup("room-1")
        assert fake_client.closed is True
        # Second cleanup should not raise
        await adapter.on_cleanup("room-1")

    @pytest.mark.asyncio
    async def test_cleanup_multi_room_keeps_client_until_last(self) -> None:
        """Client stays open until the last room is cleaned up."""
        events_room1 = [
            _event_notification(
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
        events_room2 = [
            _event_notification(
                "turn/completed",
                {
                    "turn": {
                        "id": "turn-2",
                        "status": "completed",
                        "items": [],
                        "error": None,
                    }
                },
            )
        ]
        fake_client = FakeCodexClient(events=events_room1 + events_room2)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(room_id="room-1"),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )
        await adapter.on_message(
            make_platform_message(room_id="room-2"),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-2",
        )

        # Cleaning up room-1 should NOT close the client (room-2 still active)
        await adapter.on_cleanup("room-1")
        assert fake_client.closed is False

        # Cleaning up room-2 should close the client (last room)
        await adapter.on_cleanup("room-2")
        assert fake_client.closed is True

    @pytest.mark.asyncio
    async def test_forwards_raw_codex_task_events(self) -> None:
        events = [
            _event_notification(
                "codex/event/task_started",
                {"taskId": "task-1", "task": {"title": "Inspect repository"}},
            ),
            _event_notification(
                "codex/event/task_complete",
                {"taskId": "task-1", "summary": "Inspection finished"},
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        raw_task_events = [
            event
            for event in tools.events_sent
            if event["metadata"].get("codex_event_method")
            in {
                "codex/event/task_started",
                "codex/event/task_complete",
            }
        ]
        assert len(raw_task_events) == 2
        assert raw_task_events[0]["content"] == (
            "UUID: task-1\nTask: Inspect repository\nStatus: started"
        )
        assert raw_task_events[0]["metadata"]["codex_task_id"] == "task-1"
        assert raw_task_events[1]["content"] == (
            "UUID: task-1\nTask: Inspect repository\nStatus: completed\n"
            "Summary: Inspection finished"
        )
        assert raw_task_events[1]["metadata"]["codex_task_phase"] == "completed"

    @pytest.mark.asyncio
    async def test_can_disable_synthetic_turn_task_markers(self) -> None:
        events = [
            _event_notification(
                "codex/event/task_started",
                {"taskId": "task-1", "task": {"title": "Inspect repository"}},
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(
                transport="ws",
                emit_turn_task_markers=False,
            ),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        turn_marker_events = [
            event
            for event in tools.events_sent
            if "codex_turn_status" in event["metadata"]
        ]
        assert turn_marker_events == []
        assert any(
            event["metadata"].get("codex_event_method") == "codex/event/task_started"
            for event in tools.events_sent
        )

    @pytest.mark.asyncio
    async def test_raw_task_event_without_explicit_task_id_does_not_emit_uuid(
        self,
    ) -> None:
        events = [
            _event_notification(
                "codex/event/task_started",
                {"id": "turn-1"},
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(
                transport="ws",
                emit_turn_task_markers=False,
            ),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        raw_task_event = next(
            event
            for event in tools.events_sent
            if event["metadata"].get("codex_event_method") == "codex/event/task_started"
        )
        assert raw_task_event["content"] == (
            "Task: Codex task lifecycle event\nStatus: started\n"
            "Summary: Method: codex/event/task_started"
        )
        assert "codex_task_id" not in raw_task_event["metadata"]

    @pytest.mark.asyncio
    async def test_status_command_returns_state_without_starting_turn(self) -> None:
        fake_client = FakeCodexClient()
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(content="@AR-2 Darter /status"),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        methods = [method for method, _ in fake_client.requests]
        assert "turn/start" not in methods
        assert "thread/start" not in methods
        assert len(tools.messages_sent) == 1
        assert "Codex status:" in tools.messages_sent[0]["content"]
        assert "thread_id: not mapped" in tools.messages_sent[0]["content"]

    @pytest.mark.asyncio
    async def test_model_command_sets_override_without_starting_turn(self) -> None:
        fake_client = FakeCodexClient()
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(content="/model gpt-5.5-codex"),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        methods = [method for method, _ in fake_client.requests]
        assert "turn/start" not in methods
        assert "thread/start" not in methods
        assert adapter.config.model == "gpt-5.5-codex"
        assert len(tools.messages_sent) == 1
        assert (
            "Model override set to `gpt-5.5-codex`" in tools.messages_sent[0]["content"]
        )

    @pytest.mark.asyncio
    async def test_models_alias_lists_models_without_starting_turn(self) -> None:
        fake_client = FakeCodexClient()
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(content="/models list"),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        methods = [method for method, _ in fake_client.requests]
        assert "turn/start" not in methods
        assert "thread/start" not in methods
        assert methods.count("model/list") >= 1
        assert len(tools.messages_sent) == 1
        assert "Available models" in tools.messages_sent[0]["content"]

    @pytest.mark.asyncio
    async def test_reasoning_effort_passed_in_turn_overrides(self) -> None:
        events = [
            _event_notification(
                "turn/completed",
                {
                    "turn": {
                        "id": "t1",
                        "threadId": "th1",
                        "status": "completed",
                    },
                    "text": "Done",
                },
            ),
        ]
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(
                transport="ws",
                reasoning_effort="high",
                reasoning_summary="concise",
            ),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()
        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(content="hello"),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )
        turn_params = next(
            params for method, params in fake_client.requests if method == "turn/start"
        )
        assert turn_params["effort"] == "high"
        assert turn_params["summary"] == "concise"

    @pytest.mark.asyncio
    async def test_reasoning_effort_omitted_when_none(self) -> None:
        events = [
            _event_notification(
                "turn/completed",
                {
                    "turn": {
                        "id": "t1",
                        "threadId": "th1",
                        "status": "completed",
                    },
                    "text": "Done",
                },
            ),
        ]
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()
        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(content="hello"),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )
        turn_params = next(
            params for method, params in fake_client.requests if method == "turn/start"
        )
        assert "effort" not in turn_params
        assert "summary" not in turn_params

    @pytest.mark.asyncio
    async def test_reasoning_command_sets_effort(self) -> None:
        fake_client = FakeCodexClient()
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()
        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(content="/reasoning high"),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )
        assert adapter.config.reasoning_effort == "high"
        assert len(tools.messages_sent) == 1
        assert "Reasoning effort set to `high`" in tools.messages_sent[0]["content"]

    @pytest.mark.asyncio
    async def test_reasoning_command_rejects_invalid_effort(self) -> None:
        fake_client = FakeCodexClient()
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()
        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(content="/reasoning ultra"),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )
        assert adapter.config.reasoning_effort is None
        assert len(tools.messages_sent) == 1
        assert "Invalid reasoning effort" in tools.messages_sent[0]["content"]

    @pytest.mark.asyncio
    async def test_self_config_tools_registered_when_enabled(self) -> None:
        events = [
            _event_notification(
                "turn/completed",
                {
                    "turn": {
                        "id": "t1",
                        "threadId": "th1",
                        "status": "completed",
                    },
                    "text": "Done",
                },
            ),
        ]
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", enable_self_config_tools=True),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()
        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(content="hello"),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )
        # Check that thread/start included setmodel and setreasoning dynamic tools
        thread_params = next(
            params
            for method, params in fake_client.requests
            if method == "thread/start"
        )
        tool_names = [t["name"] for t in thread_params.get("dynamicTools", [])]
        assert "setmodel" in tool_names
        assert "setreasoning" in tool_names

    @pytest.mark.asyncio
    async def test_self_config_tools_not_registered_when_disabled(self) -> None:
        events = [
            _event_notification(
                "turn/completed",
                {
                    "turn": {
                        "id": "t1",
                        "threadId": "th1",
                        "status": "completed",
                    },
                    "text": "Done",
                },
            ),
        ]
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", enable_self_config_tools=False),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()
        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(content="hello"),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )
        thread_params = next(
            params
            for method, params in fake_client.requests
            if method == "thread/start"
        )
        tool_names = [t["name"] for t in thread_params.get("dynamicTools", [])]
        assert "setmodel" not in tool_names
        assert "setreasoning" not in tool_names

    @pytest.mark.asyncio
    async def test_setmodel_tool_changes_model(self) -> None:
        events = [
            _event_request(
                99,
                "item/tool/call",
                {
                    "tool": "setmodel",
                    "callId": "call-1",
                    "arguments": {"model": "o3"},
                },
            ),
            _event_notification(
                "turn/completed",
                {
                    "turn": {
                        "id": "turn-1",
                        "status": "completed",
                        "items": [],
                        "error": None,
                    },
                },
            ),
        ]
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", enable_self_config_tools=True),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()
        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(content="switch to o3"),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )
        assert adapter.config.model == "o3"
        assert adapter._selected_model == "o3"
        # Verify the tool response was sent back
        tool_responses = [
            (rid, result)
            for rid, result in fake_client.responses
            if isinstance(result, dict) and "contentItems" in result
        ]
        assert len(tool_responses) >= 1
        result_text = tool_responses[0][1]["contentItems"][0]["text"]
        assert "o3" in result_text

    @pytest.mark.asyncio
    async def test_setreasoning_tool_changes_effort(self) -> None:
        events = [
            _event_request(
                99,
                "item/tool/call",
                {
                    "tool": "setreasoning",
                    "callId": "call-2",
                    "arguments": {"effort": "xhigh", "summary": "detailed"},
                },
            ),
            _event_notification(
                "turn/completed",
                {
                    "turn": {
                        "id": "turn-1",
                        "status": "completed",
                        "items": [],
                        "error": None,
                    },
                },
            ),
        ]
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", enable_self_config_tools=True),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()
        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(content="increase reasoning"),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )
        assert adapter.config.reasoning_effort == "xhigh"
        assert adapter.config.reasoning_summary == "detailed"

    @pytest.mark.asyncio
    async def test_setreasoning_tool_rejects_invalid_effort(self) -> None:
        events = [
            _event_request(
                99,
                "item/tool/call",
                {
                    "tool": "setreasoning",
                    "callId": "call-3",
                    "arguments": {"effort": "ultra"},
                },
            ),
            _event_notification(
                "turn/completed",
                {
                    "turn": {
                        "id": "turn-1",
                        "status": "completed",
                        "items": [],
                        "error": None,
                    },
                },
            ),
        ]
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", enable_self_config_tools=True),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()
        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(content="set reasoning ultra"),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )
        # Effort should not have changed
        assert adapter.config.reasoning_effort is None
        # Tool response should contain error message
        tool_responses = [
            (rid, result)
            for rid, result in fake_client.responses
            if isinstance(result, dict) and "contentItems" in result
        ]
        assert len(tool_responses) >= 1
        result_text = tool_responses[0][1]["contentItems"][0]["text"]
        assert "Invalid reasoning effort" in result_text

    @pytest.mark.asyncio
    async def test_sandbox_alias_is_normalized_for_thread_and_turn(self) -> None:
        events = [
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(
                transport="ws",
                sandbox="dangerFullAccess",
            ),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        thread_start = next(
            params
            for method, params in fake_client.requests
            if method == "thread/start"
        )
        turn_start = next(
            params for method, params in fake_client.requests if method == "turn/start"
        )
        # thread/start only accepts the sandbox field (SandboxMode enum)
        assert thread_start["sandbox"] == "danger-full-access"
        # turn/start uses sandboxPolicy (full SandboxPolicy tagged union)
        assert turn_start["sandboxPolicy"]["type"] == "dangerFullAccess"

    @pytest.mark.asyncio
    async def test_external_sandbox_alias_uses_sandbox_policy(self) -> None:
        events = [
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(
                transport="ws",
                sandbox="external-sandbox",
            ),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        thread_start = next(
            params
            for method, params in fake_client.requests
            if method == "thread/start"
        )
        turn_start = next(
            params for method, params in fake_client.requests if method == "turn/start"
        )
        # thread/start has no sandboxPolicy field; externalSandbox is
        # only representable at turn level
        assert "sandbox" not in thread_start
        assert "sandboxPolicy" not in thread_start
        # turn/start can express the full SandboxPolicy tagged union
        assert turn_start["sandboxPolicy"]["type"] == "externalSandbox"

    @pytest.mark.asyncio
    async def test_transport_closed_event_aborts_turn(self) -> None:
        """A transport/closed event should end the turn with a failed status."""
        events = [
            _event_notification(
                "transport/closed",
                {"reason": "Codex process exited unexpectedly"},
            )
        ]
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # Adapter should send a failure message mentioning the disconnect.
        assert any(
            "transport closed" in msg["content"].lower() for msg in tools.messages_sent
        )

    @pytest.mark.asyncio
    async def test_transport_closed_resets_client_state(self) -> None:
        """After transport/closed, _client and _initialized should be reset
        so the next message rebuilds the client via _ensure_client_ready()."""
        events = [
            _event_notification(
                "transport/closed",
                {"reason": "Codex process exited unexpectedly"},
            )
        ]
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # After transport/closed, client state should be reset
        assert adapter._client is None
        assert adapter._initialized is False

    @pytest.mark.asyncio
    async def test_turn_timeout_sends_interrupt_and_clean_error(self) -> None:
        """When recv_event times out, the adapter sends turn/interrupt and reports cleanly."""
        # No events means FakeCodexClient raises asyncio.TimeoutError immediately.
        fake_client = FakeCodexClient(events=[])
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", turn_timeout_s=0.01),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # Adapter should have sent turn/interrupt.
        interrupt_requests = [
            (m, p) for m, p in fake_client.requests if m == "turn/interrupt"
        ]
        assert len(interrupt_requests) == 1

        # Adapter should send a user-facing message about stopping.
        assert any("stopped" in msg["content"].lower() for msg in tools.messages_sent)

    @pytest.mark.asyncio
    async def test_item_completed_text_overrides_accumulated_deltas(self) -> None:
        """item/completed text is authoritative and should replace any accumulated deltas."""
        events = [
            _event_notification(
                "item/agentMessage/delta",
                {"itemId": "msg-1", "delta": "partial "},
            ),
            _event_notification(
                "item/agentMessage/delta",
                {"itemId": "msg-1", "delta": "garbled"},
            ),
            _event_notification(
                "item/completed",
                {
                    "item": {
                        "type": "agentMessage",
                        "id": "msg-1",
                        "text": "authoritative final text",
                    }
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # The authoritative text from item/completed should be used, not the deltas.
        assert any(
            msg["content"] == "authoritative final text" for msg in tools.messages_sent
        )

    @pytest.mark.asyncio
    async def test_custom_tools_schemas_merged_into_dynamic_tools(self) -> None:
        """Custom tool schemas appear in _build_dynamic_tools output."""

        class WeatherInput(BaseModel):
            """Get current weather for a location."""

            city: str

        def get_weather(inp: WeatherInput) -> str:
            return f"Sunny in {inp.city}"

        custom_tools: list[CustomToolDef] = [(WeatherInput, get_weather)]
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            additional_tools=custom_tools,
        )

        tools = ToolSchemaFakeTools()
        dynamic_tools = adapter._build_dynamic_tools(tools)

        names = [t["name"] for t in dynamic_tools]
        assert "weather" in names

        weather_tool = next(t for t in dynamic_tools if t["name"] == "weather")
        assert weather_tool["description"] == "Get current weather for a location."
        assert "inputSchema" in weather_tool
        assert "city" in weather_tool["inputSchema"].get("properties", {})

    @pytest.mark.asyncio
    async def test_custom_tool_dispatched_before_platform_tools(self) -> None:
        """Custom tool is invoked via execute_custom_tool, not platform tools."""

        class CalculatorInput(BaseModel):
            """Simple calculator."""

            expression: str

        call_log: list[str] = []

        async def calculate(inp: CalculatorInput) -> str:
            call_log.append(inp.expression)
            return "42"

        custom_tools: list[CustomToolDef] = [(CalculatorInput, calculate)]
        events = [
            _event_request(
                99,
                "item/tool/call",
                {
                    "tool": "calculator",
                    "arguments": {"expression": "6*7"},
                    "callId": "call-99",
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            additional_tools=custom_tools,
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # Custom tool was called
        assert call_log == ["6*7"]
        # Platform execute_tool_call was NOT called for the custom tool
        assert not any(tc["tool_name"] == "calculator" for tc in tools.tool_calls)
        # Response was sent back to Codex
        assert fake_client.responses
        _, payload = fake_client.responses[0]
        assert payload["success"] is True
        assert payload["contentItems"][0]["text"] == "42"

    @pytest.mark.asyncio
    async def test_execution_reporting_emits_tool_call_and_result_events(self) -> None:
        """With enable_execution_reporting, tool_call and tool_result events are emitted."""
        events = [
            _event_request(
                50,
                "item/tool/call",
                {
                    "tool": "thenvoi_lookup_peers",
                    "arguments": {"page": 1},
                    "callId": "call-50",
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", enable_execution_reporting=True),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        tool_call_events = [
            e for e in tools.events_sent if e["message_type"] == "tool_call"
        ]
        tool_result_events = [
            e for e in tools.events_sent if e["message_type"] == "tool_result"
        ]
        assert len(tool_call_events) == 1
        assert len(tool_result_events) == 1

        call_data = json.loads(tool_call_events[0]["content"])
        assert call_data["name"] == "thenvoi_lookup_peers"
        assert call_data["tool_call_id"] == "call-50"

        result_data = json.loads(tool_result_events[0]["content"])
        assert result_data["name"] == "thenvoi_lookup_peers"
        assert result_data["tool_call_id"] == "call-50"

    @pytest.mark.asyncio
    async def test_execution_reporting_disabled_by_default(self) -> None:
        """Without enable_execution_reporting, no tool_call/tool_result events are emitted."""
        events = [
            _event_request(
                50,
                "item/tool/call",
                {
                    "tool": "thenvoi_lookup_peers",
                    "arguments": {"page": 1},
                    "callId": "call-50",
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        tool_events = [
            e
            for e in tools.events_sent
            if e["message_type"] in {"tool_call", "tool_result"}
        ]
        assert tool_events == []

    @pytest.mark.asyncio
    async def test_execution_reporting_on_tool_error(self) -> None:
        """Execution reporting emits tool_result with error text on failure."""

        class FailInput(BaseModel):
            """A tool that always fails."""

            x: int

        async def fail_func(inp: FailInput) -> str:
            raise RuntimeError("boom")

        custom_tools: list[CustomToolDef] = [(FailInput, fail_func)]
        events = [
            _event_request(
                60,
                "item/tool/call",
                {
                    "tool": "fail",
                    "arguments": {"x": 1},
                    "callId": "call-60",
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", enable_execution_reporting=True),
            additional_tools=custom_tools,
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        tool_result_events = [
            e for e in tools.events_sent if e["message_type"] == "tool_result"
        ]
        assert len(tool_result_events) == 1
        result_data = json.loads(tool_result_events[0]["content"])
        assert result_data["name"] == "fail"
        assert "boom" in result_data["output"]
        assert result_data["tool_call_id"] == "call-60"

        # Codex response should indicate failure
        _, payload = fake_client.responses[0]
        assert payload["success"] is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool_name",
        ["thenvoi_send_event", "thenvoi_send_message"],
    )
    async def test_execution_reporting_suppressed_for_platform_output_tools(
        self, tool_name: str
    ) -> None:
        """Platform tools that produce visible output should not emit reporting events."""
        events = [
            _event_request(
                70,
                "item/tool/call",
                {
                    "tool": tool_name,
                    "arguments": {"content": "test", "message_type": "thought"},
                    "callId": "call-70",
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", enable_execution_reporting=True),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # The tool call itself should still execute
        assert len(tools.tool_calls) == 1
        assert tools.tool_calls[0]["tool_name"] == tool_name

        # But no tool_call/tool_result reporting events should be emitted
        reporting_events = [
            e
            for e in tools.events_sent
            if e["message_type"] in {"tool_call", "tool_result"}
        ]
        assert reporting_events == []


class TestItemCompletedForwarding:
    """Tests for forwarding internal Codex operations as platform events."""

    @pytest.mark.asyncio
    async def test_item_completed_commandExecution_emits_tool_events(self) -> None:
        """commandExecution item emits tool_call + tool_result with command/output."""
        events = [
            _event_notification(
                "item/completed",
                {
                    "item": {
                        "type": "commandExecution",
                        "id": "cmd-1",
                        "command": "ls -la",
                        "cwd": "/workspace",
                        "aggregated_output": "total 42\ndrwxr-xr-x ...",
                        "exitCode": 0,
                        "status": "completed",
                    }
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", enable_execution_reporting=True),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        tool_call_events = [
            e for e in tools.events_sent if e["message_type"] == "tool_call"
        ]
        tool_result_events = [
            e for e in tools.events_sent if e["message_type"] == "tool_result"
        ]
        assert len(tool_call_events) == 1
        assert len(tool_result_events) == 1

        call_data = json.loads(tool_call_events[0]["content"])
        assert call_data["name"] == "exec"
        assert call_data["args"]["command"] == "ls -la"
        assert call_data["args"]["cwd"] == "/workspace"
        assert call_data["tool_call_id"] == "cmd-1"

        result_data = json.loads(tool_result_events[0]["content"])
        assert result_data["name"] == "exec"
        assert "total 42" in result_data["output"]
        assert "exit_code=0" in result_data["output"]
        assert result_data["tool_call_id"] == "cmd-1"

    @pytest.mark.asyncio
    async def test_item_completed_fileChange_emits_tool_events(self) -> None:
        """fileChange emits tool_call + tool_result with file paths."""
        events = [
            _event_notification(
                "item/completed",
                {
                    "item": {
                        "type": "fileChange",
                        "id": "fc-1",
                        "changes": [
                            {"path": "src/main.py"},
                            {"path": "src/utils.py"},
                        ],
                        "status": "applied",
                    }
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", enable_execution_reporting=True),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        tool_call_events = [
            e for e in tools.events_sent if e["message_type"] == "tool_call"
        ]
        tool_result_events = [
            e for e in tools.events_sent if e["message_type"] == "tool_result"
        ]
        assert len(tool_call_events) == 1
        assert len(tool_result_events) == 1

        call_data = json.loads(tool_call_events[0]["content"])
        assert call_data["name"] == "file_edit"
        assert call_data["args"]["files"] == ["src/main.py", "src/utils.py"]

        result_data = json.loads(tool_result_events[0]["content"])
        assert result_data["output"] == "applied"

    @pytest.mark.asyncio
    async def test_item_completed_fileChange_missing_changes_is_safe(self) -> None:
        """fileChange without changes list should not crash and emits empty files."""
        events = [
            _event_notification(
                "item/completed",
                {
                    "item": {
                        "type": "fileChange",
                        "id": "fc-2",
                        "changes": None,
                        "status": "applied",
                    }
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", enable_execution_reporting=True),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        tool_call_events = [
            e for e in tools.events_sent if e["message_type"] == "tool_call"
        ]
        assert len(tool_call_events) == 1
        call_data = json.loads(tool_call_events[0]["content"])
        assert call_data["name"] == "file_edit"
        assert call_data["args"]["files"] == []

    @pytest.mark.asyncio
    async def test_item_completed_imageView_emits_tool_events(self) -> None:
        """imageView emits tool_call + tool_result."""
        events = [
            _event_notification(
                "item/completed",
                {
                    "item": {
                        "type": "imageView",
                        "id": "img-1",
                        "path": "/tmp/screenshot.png",
                        "status": "viewed",
                    }
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", enable_execution_reporting=True),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        tool_call_events = [
            e for e in tools.events_sent if e["message_type"] == "tool_call"
        ]
        tool_result_events = [
            e for e in tools.events_sent if e["message_type"] == "tool_result"
        ]
        assert len(tool_call_events) == 1
        assert len(tool_result_events) == 1

        call_data = json.loads(tool_call_events[0]["content"])
        assert call_data["name"] == "view_image"
        assert call_data["args"]["path"] == "/tmp/screenshot.png"

    @pytest.mark.asyncio
    async def test_item_completed_collabAgentToolCall_emits_tool_events(self) -> None:
        """collabAgentToolCall emits tool_call + tool_result preserving empty result."""
        events = [
            _event_notification(
                "item/completed",
                {
                    "item": {
                        "type": "collabAgentToolCall",
                        "id": "collab-1",
                        "tool": "delegate",
                        "prompt": "Review the changes",
                        "agents": ["Reviewer-1", "Reviewer-2"],
                        "result": {},
                    }
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", enable_execution_reporting=True),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        tool_call_events = [
            e for e in tools.events_sent if e["message_type"] == "tool_call"
        ]
        tool_result_events = [
            e for e in tools.events_sent if e["message_type"] == "tool_result"
        ]
        assert len(tool_call_events) == 1
        assert len(tool_result_events) == 1

        call_data = json.loads(tool_call_events[0]["content"])
        assert call_data["name"] == "collab:delegate"
        assert call_data["args"]["prompt"] == "Review the changes"
        assert call_data["args"]["agents"] == ["Reviewer-1", "Reviewer-2"]

        result_data = json.loads(tool_result_events[0]["content"])
        assert result_data["output"] == "{}"

    @pytest.mark.asyncio
    async def test_item_completed_mcpToolCall_emits_tool_events(self) -> None:
        """mcpToolCall emits tool_call + tool_result with server/tool name."""
        events = [
            _event_notification(
                "item/completed",
                {
                    "item": {
                        "type": "mcpToolCall",
                        "id": "mcp-1",
                        "server": "filesystem",
                        "tool": "read_file",
                        "arguments": {"path": "/etc/hosts"},
                        "result": {"content": "127.0.0.1 localhost"},
                    }
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", enable_execution_reporting=True),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        tool_call_events = [
            e for e in tools.events_sent if e["message_type"] == "tool_call"
        ]
        tool_result_events = [
            e for e in tools.events_sent if e["message_type"] == "tool_result"
        ]
        assert len(tool_call_events) == 1
        assert len(tool_result_events) == 1

        call_data = json.loads(tool_call_events[0]["content"])
        assert call_data["name"] == "mcp:filesystem/read_file"
        assert call_data["args"]["path"] == "/etc/hosts"

        result_data = json.loads(tool_result_events[0]["content"])
        assert "127.0.0.1 localhost" in result_data["output"]

    @pytest.mark.asyncio
    async def test_item_completed_reasoning_emits_thought(self) -> None:
        """reasoning item emits thought event when emit_thought_events=True."""
        events = [
            _event_notification(
                "item/completed",
                {
                    "item": {
                        "type": "reasoning",
                        "id": "reason-1",
                        "summary": [
                            "Analyzing the codebase structure",
                            "Identified key files to modify",
                        ],
                    }
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", emit_thought_events=True),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        thought_events = [
            e for e in tools.events_sent if e["message_type"] == "thought"
        ]
        assert len(thought_events) == 1
        assert "Analyzing the codebase structure" in thought_events[0]["content"]
        assert "Identified key files to modify" in thought_events[0]["content"]

    @pytest.mark.asyncio
    async def test_item_completed_skipped_when_reporting_disabled(self) -> None:
        """No tool events when enable_execution_reporting=False."""
        events = [
            _event_notification(
                "item/completed",
                {
                    "item": {
                        "type": "commandExecution",
                        "id": "cmd-1",
                        "command": "ls",
                        "exitCode": 0,
                    }
                },
            ),
            _event_notification(
                "item/completed",
                {
                    "item": {
                        "type": "reasoning",
                        "id": "reason-1",
                        "summary": ["thinking"],
                    }
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(
                transport="ws",
                enable_execution_reporting=False,
                emit_thought_events=False,
            ),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        tool_events = [
            e
            for e in tools.events_sent
            if e["message_type"] in {"tool_call", "tool_result", "thought"}
        ]
        assert tool_events == []

    @pytest.mark.asyncio
    async def test_item_completed_agentMessage_still_sets_final_text(self) -> None:
        """Existing agentMessage behavior preserved alongside new forwarding."""
        events = [
            _event_notification(
                "item/completed",
                {
                    "item": {
                        "type": "commandExecution",
                        "id": "cmd-1",
                        "command": "pytest",
                        "exitCode": 0,
                        "aggregated_output": "all tests passed",
                    }
                },
            ),
            _event_notification(
                "item/completed",
                {
                    "item": {
                        "type": "agentMessage",
                        "id": "msg-1",
                        "text": "All tests pass!",
                    }
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", enable_execution_reporting=True),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # agentMessage text should still be sent as the final message
        assert any(msg["content"] == "All tests pass!" for msg in tools.messages_sent)
        # commandExecution should also be forwarded as tool events
        tool_call_events = [
            e for e in tools.events_sent if e["message_type"] == "tool_call"
        ]
        assert len(tool_call_events) == 1

    @pytest.mark.asyncio
    async def test_item_completed_webSearch_emits_tool_events(self) -> None:
        """webSearch item emits tool_call + tool_result."""
        events = [
            _event_notification(
                "item/completed",
                {
                    "item": {
                        "type": "webSearch",
                        "id": "ws-1",
                        "query": "python asyncio tutorial",
                        "action": {"url": "https://example.com", "title": "Tutorial"},
                    }
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", enable_execution_reporting=True),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        tool_call_events = [
            e for e in tools.events_sent if e["message_type"] == "tool_call"
        ]
        assert len(tool_call_events) == 1
        call_data = json.loads(tool_call_events[0]["content"])
        assert call_data["name"] == "web_search"
        assert call_data["args"]["query"] == "python asyncio tutorial"

    @pytest.mark.asyncio
    async def test_item_completed_metadata_includes_codex_ids(self) -> None:
        """Forwarded events include codex_room_id, codex_thread_id, codex_turn_id."""
        events = [
            _event_notification(
                "item/completed",
                {
                    "item": {
                        "type": "commandExecution",
                        "id": "cmd-1",
                        "command": "echo hi",
                        "exitCode": 0,
                    }
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", enable_execution_reporting=True),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        tool_call_events = [
            e for e in tools.events_sent if e["message_type"] == "tool_call"
        ]
        assert len(tool_call_events) == 1
        meta = tool_call_events[0]["metadata"]
        assert meta["codex_room_id"] == "room-1"
        assert meta["codex_thread_id"] == "thr-1"
        assert meta["codex_turn_id"] == "turn-1"


class TestHistoryInjection:
    @pytest.mark.asyncio
    async def test_history_injected_on_resume_failure(self) -> None:
        """Resume fails, fresh thread created, first turn input contains history block."""
        events = [
            _event_notification(
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
        fake_client = FakeCodexClient(
            events=events,
            resume_error=CodexJsonRpcError(code=-32002, message="Thread expired"),
        )
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")

        raw_history = [
            {
                "message_type": "task",
                "content": "mapping event",
                "metadata": {"codex_thread_id": "thr-old"},
            },
            {
                "message_type": "text",
                "content": "Can you refactor the auth module?",
                "sender_name": "Alice",
            },
            {
                "message_type": "text",
                "content": "Done — split into auth_handler.py and middleware.py",
                "sender_name": "CodexAgent",
            },
        ]

        inp = AgentInput(
            msg=make_platform_message(
                room_id="room-1", content="Now add rate limiting"
            ),
            tools=tools,
            history=HistoryProvider(raw=raw_history),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )
        await adapter.on_event(inp)

        turn_start = next(
            params for method, params in fake_client.requests if method == "turn/start"
        )
        turn_input = turn_start["input"]
        history_items = [
            item for item in turn_input if "[Conversation History]" in item["text"]
        ]
        assert len(history_items) == 1
        assert "[Alice]: Can you refactor the auth module?" in history_items[0]["text"]
        assert (
            "[CodexAgent]: Done — split into auth_handler.py and middleware.py"
            in history_items[0]["text"]
        )
        # Task events should NOT appear in history context
        assert "mapping event" not in history_items[0]["text"]

    @pytest.mark.asyncio
    async def test_history_not_injected_on_successful_resume(self) -> None:
        """Resume succeeds, no history injection."""
        events = [
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")

        raw_history = [
            {
                "message_type": "task",
                "content": "mapping",
                "metadata": {
                    "codex_thread_id": "thr-existing",
                    "codex_room_id": "room-1",
                },
            },
            {
                "message_type": "text",
                "content": "Hello",
                "sender_name": "Alice",
            },
        ]

        inp = AgentInput(
            msg=make_platform_message(room_id="room-1", content="Continue"),
            tools=tools,
            history=HistoryProvider(raw=raw_history),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )
        await adapter.on_event(inp)

        turn_start = next(
            params for method, params in fake_client.requests if method == "turn/start"
        )
        turn_input = turn_start["input"]
        assert not any("[Conversation History]" in item["text"] for item in turn_input)

    @pytest.mark.asyncio
    async def test_history_not_injected_when_disabled(self) -> None:
        """inject_history_on_resume_failure=False, no injection even on failure."""
        events = [
            _event_notification(
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
        fake_client = FakeCodexClient(
            events=events,
            resume_error=CodexJsonRpcError(code=-32002, message="Thread expired"),
        )
        adapter = CodexAdapter(
            config=CodexAdapterConfig(
                transport="ws",
                inject_history_on_resume_failure=False,
            ),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")

        raw_history = [
            {
                "message_type": "text",
                "content": "Hello",
                "sender_name": "Alice",
            },
        ]

        inp = AgentInput(
            msg=make_platform_message(room_id="room-1", content="Continue"),
            tools=tools,
            history=HistoryProvider(raw=raw_history),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )
        await adapter.on_event(inp)

        turn_start = next(
            params for method, params in fake_client.requests if method == "turn/start"
        )
        turn_input = turn_start["input"]
        assert not any("[Conversation History]" in item["text"] for item in turn_input)

    @pytest.mark.asyncio
    async def test_history_filters_non_text_messages(self) -> None:
        """Only text/message types appear in injected context."""
        events = [
            _event_notification(
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
        fake_client = FakeCodexClient(
            events=events,
            resume_error=CodexJsonRpcError(code=-32002, message="Not found"),
        )
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")

        raw_history = [
            {
                "message_type": "task",
                "content": "task event",
                "sender_name": "System",
                "metadata": {"codex_thread_id": "thr-old", "codex_room_id": "room-1"},
            },
            {
                "message_type": "tool_call",
                "content": '{"name": "foo"}',
                "sender_name": "Agent",
            },
            {
                "message_type": "tool_result",
                "content": "result",
                "sender_name": "Agent",
            },
            {
                "message_type": "thought",
                "content": "thinking...",
                "sender_name": "Agent",
            },
            {"message_type": "error", "content": "oops", "sender_name": "Agent"},
            {
                "message_type": "text",
                "content": "Hello world",
                "sender_name": "Alice",
            },
            {
                "message_type": "message",
                "content": "Hi there",
                "sender_name": "Bob",
            },
        ]

        inp = AgentInput(
            msg=make_platform_message(room_id="room-1", content="Go"),
            tools=tools,
            history=HistoryProvider(raw=raw_history),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )
        await adapter.on_event(inp)

        turn_start = next(
            params for method, params in fake_client.requests if method == "turn/start"
        )
        turn_input = turn_start["input"]
        history_items = [
            item for item in turn_input if "[Conversation History]" in item["text"]
        ]
        assert len(history_items) == 1
        text = history_items[0]["text"]
        assert "[Alice]: Hello world" in text
        assert "[Bob]: Hi there" in text
        assert "task event" not in text
        assert "thinking..." not in text
        assert "oops" not in text
        assert "tool_call" not in text

    @pytest.mark.asyncio
    async def test_history_respects_max_messages(self) -> None:
        """Only last max_history_messages are injected."""
        events = [
            _event_notification(
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
        fake_client = FakeCodexClient(
            events=events,
            resume_error=CodexJsonRpcError(code=-32002, message="Not found"),
        )
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", max_history_messages=3),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")

        raw_history: list[dict[str, Any]] = [
            {
                "message_type": "task",
                "content": "mapping",
                "metadata": {"codex_thread_id": "thr-old", "codex_room_id": "room-1"},
            },
        ]
        raw_history.extend(
            {
                "message_type": "text",
                "content": f"Message {i}",
                "sender_name": "Alice",
            }
            for i in range(10)
        )

        inp = AgentInput(
            msg=make_platform_message(room_id="room-1", content="Go"),
            tools=tools,
            history=HistoryProvider(raw=raw_history),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )
        await adapter.on_event(inp)

        turn_start = next(
            params for method, params in fake_client.requests if method == "turn/start"
        )
        turn_input = turn_start["input"]
        history_items = [
            item for item in turn_input if "[Conversation History]" in item["text"]
        ]
        assert len(history_items) == 1
        text = history_items[0]["text"]
        # Only last 3 messages should be present
        assert "Message 7" in text
        assert "Message 8" in text
        assert "Message 9" in text
        assert "Message 0" not in text
        assert "Message 6" not in text

    @pytest.mark.asyncio
    async def test_history_cleared_after_injection(self) -> None:
        """Raw history removed from memory after first turn."""
        events = [
            _event_notification(
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
        fake_client = FakeCodexClient(
            events=events,
            resume_error=CodexJsonRpcError(code=-32002, message="Not found"),
        )
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Codex Agent", "A coding agent")

        raw_history = [
            {
                "message_type": "text",
                "content": "Hello",
                "sender_name": "Alice",
            },
        ]

        inp = AgentInput(
            msg=make_platform_message(room_id="room-1", content="Go"),
            tools=tools,
            history=HistoryProvider(raw=raw_history),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )
        await adapter.on_event(inp)

        # After injection, stashed data should be cleaned up
        assert "room-1" not in adapter._raw_history_by_room
        assert "room-1" not in adapter._needs_history_injection

    @pytest.mark.asyncio
    async def test_model_fallback_on_auto_selected_model(self) -> None:
        """When auto-selected model fails, adapter falls back to another model."""
        events = [
            _event_notification(
                "turn/completed",
                {"turn": {"id": "turn-1", "status": "completed"}},
            ),
        ]
        fake_client = FakeCodexClient(
            events=events,
            turn_start_error=CodexJsonRpcError(
                code=-32000,
                message="Model gpt-5.3-codex is not available for this account",
            ),
            turn_start_error_once=True,
            model_list_result={
                "data": [
                    {"id": "gpt-5.3-codex", "hidden": False},
                    {"id": "gpt-5.2", "hidden": False},
                ]
            },
        )
        # model=None means auto-select — fallback should trigger
        adapter = CodexAdapter(
            config=CodexAdapterConfig(model=None),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()
        await adapter.on_started("Agent", "An agent")

        msg = make_platform_message(room_id="room-1", content="hello")
        await adapter.on_message(
            msg,
            tools,
            CodexSessionState(),
            None,
            None,
            is_session_bootstrap=False,
            room_id="room-1",
        )

        # Should have retried with fallback model
        turn_start_calls = [
            (m, p) for m, p in fake_client.requests if m == "turn/start"
        ]
        assert len(turn_start_calls) == 2
        # First attempt used auto-selected model
        assert turn_start_calls[0][1]["model"] == "gpt-5.3-codex"
        # Retry used fallback (gpt-5.2 since gpt-5.3-codex is excluded)
        assert turn_start_calls[1][1]["model"] == "gpt-5.2"
        assert adapter._selected_model == "gpt-5.2"

    @pytest.mark.asyncio
    async def test_model_fallback_prefers_gpt_5_2(self) -> None:
        """Fallback prefers gpt-5.2 over other models."""
        events = [
            _event_notification(
                "turn/completed",
                {"turn": {"id": "turn-1", "status": "completed"}},
            ),
        ]
        # Auto-select picks gpt-5.3-codex (first codex model).
        # After turn/start fails, fallback should pick gpt-5.2 (preferred fallback).
        fake_client = FakeCodexClient(
            events=events,
            turn_start_error=CodexJsonRpcError(
                code=-32000, message="Model unavailable"
            ),
            turn_start_error_once=True,
            model_list_result={
                "data": [
                    {"id": "gpt-5.3-codex", "hidden": False},
                    {"id": "gpt-5.1", "hidden": False},
                    {"id": "gpt-5.2", "hidden": False},
                ]
            },
        )
        adapter = CodexAdapter(
            config=CodexAdapterConfig(model=None),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()
        await adapter.on_started("Agent", "An agent")

        msg = make_platform_message(room_id="room-1", content="hello")
        await adapter.on_message(
            msg,
            tools,
            CodexSessionState(),
            None,
            None,
            is_session_bootstrap=False,
            room_id="room-1",
        )

        # Should prefer gpt-5.2 over gpt-5.1
        assert adapter._selected_model == "gpt-5.2"

    @pytest.mark.asyncio
    async def test_explicit_model_error_propagates_without_fallback(self) -> None:
        """When the user explicitly set a model, errors propagate — no silent fallback."""
        fake_client = FakeCodexClient(
            turn_start_error=CodexJsonRpcError(
                code=-32000,
                message="Model gpt-5.3-codex-spark is not available",
            ),
            turn_start_error_once=False,
            model_list_result={
                "data": [
                    {"id": "gpt-5.3-codex", "hidden": False},
                    {"id": "gpt-5.2", "hidden": False},
                ]
            },
        )
        # Explicitly configured model — user chose this, don't override
        adapter = CodexAdapter(
            config=CodexAdapterConfig(model="gpt-5.3-codex-spark"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()
        await adapter.on_started("Agent", "An agent")

        msg = make_platform_message(room_id="room-1", content="hello")
        with pytest.raises(CodexJsonRpcError, match="not available"):
            await adapter.on_message(
                msg,
                tools,
                CodexSessionState(),
                None,
                None,
                is_session_bootstrap=False,
                room_id="room-1",
            )

        # No model/list query for fallback should have been attempted
        model_list_calls = [m for m, _ in fake_client.requests if m == "model/list"]
        assert len(model_list_calls) == 0

    @pytest.mark.asyncio
    async def test_model_fallback_raises_when_no_alternative(self) -> None:
        """When no fallback model is available, the original error propagates."""
        fake_client = FakeCodexClient(
            turn_start_error=CodexJsonRpcError(
                code=-32000, message="Model not available"
            ),
            turn_start_error_once=False,
            model_list_result={"data": []},
        )
        adapter = CodexAdapter(
            config=CodexAdapterConfig(model=None),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()
        await adapter.on_started("Agent", "An agent")

        msg = make_platform_message(room_id="room-1", content="hello")
        with pytest.raises(CodexJsonRpcError, match="not available"):
            await adapter.on_message(
                msg,
                tools,
                CodexSessionState(),
                None,
                None,
                is_session_bootstrap=False,
                room_id="room-1",
            )

    @pytest.mark.asyncio
    async def test_model_fallback_uses_hardcoded_default_when_model_list_fails(
        self,
    ) -> None:
        """When model/list fails during fallback, hardcoded defaults are used."""
        events = [
            _event_notification(
                "turn/completed",
                {"turn": {"id": "turn-1", "status": "completed"}},
            ),
        ]

        class ModelListFailsClient(FakeCodexClient):
            async def request(
                self,
                method: str,
                params: dict[str, Any] | None = None,
                *,
                retry_on_overload: bool = True,
            ) -> dict[str, Any]:
                if method == "model/list" and self.initialized:
                    # First call during init succeeds, subsequent calls fail
                    raise RuntimeError("model/list unavailable")
                return await super().request(
                    method, params, retry_on_overload=retry_on_overload
                )

        fake_client = ModelListFailsClient(
            events=events,
            turn_start_error=CodexJsonRpcError(
                code=-32000, message="Model not available"
            ),
            turn_start_error_once=True,
        )
        adapter = CodexAdapter(
            config=CodexAdapterConfig(model=None),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()
        await adapter.on_started("Agent", "An agent")

        msg = make_platform_message(room_id="room-1", content="hello")
        await adapter.on_message(
            msg,
            tools,
            CodexSessionState(),
            None,
            None,
            is_session_bootstrap=False,
            room_id="room-1",
        )

        # Should fall back to gpt-5.2 from hardcoded defaults
        # (gpt-5.3-codex excluded since it was the auto-selected model that failed)
        assert adapter._selected_model == "gpt-5.2"

    @pytest.mark.asyncio
    async def test_non_model_error_not_caught_by_fallback(self) -> None:
        """Non-model-related errors propagate without fallback attempt."""
        fake_client = FakeCodexClient(
            turn_start_error=CodexJsonRpcError(
                code=-32001, message="Server overloaded"
            ),
            turn_start_error_once=False,
        )
        adapter = CodexAdapter(
            config=CodexAdapterConfig(model=None),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()
        await adapter.on_started("Agent", "An agent")

        msg = make_platform_message(room_id="room-1", content="hello")
        with pytest.raises(CodexJsonRpcError, match="overloaded"):
            await adapter.on_message(
                msg,
                tools,
                CodexSessionState(),
                None,
                None,
                is_session_bootstrap=False,
                room_id="room-1",
            )

    @pytest.mark.asyncio
    async def test_startup_config_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Startup emits a redacted config summary log line."""
        fake_client = FakeCodexClient()
        adapter = CodexAdapter(
            config=CodexAdapterConfig(
                transport="stdio",
                model="gpt-5.3-codex",
                sandbox="workspace-write",
                approval_mode="manual",
            ),
            client_factory=lambda _config: fake_client,
        )

        with caplog.at_level("INFO", logger="thenvoi.adapters.codex"):
            await adapter.on_started("TestBot", "A test agent")

        startup_logs = [
            r for r in caplog.records if "Codex adapter started" in r.message
        ]
        assert len(startup_logs) == 1
        log_msg = startup_logs[0].message
        assert "agent=TestBot" in log_msg
        assert "transport=stdio" in log_msg
        assert "model=gpt-5.3-codex" in log_msg
        assert "sandbox=workspace-write" in log_msg
        assert "approval_mode=manual" in log_msg

    @pytest.mark.asyncio
    async def test_codex_error_emits_event_unconditionally(self) -> None:
        """Non-retryable Codex errors always emit a structured error event."""
        fake_client = FakeCodexClient(
            events=[
                _event_notification(
                    "error",
                    {"error": {"message": "Something went wrong"}, "willRetry": False},
                ),
                _event_notification(
                    "turn/completed",
                    {"turn": {"id": "turn-1", "status": "failed"}},
                ),
            ],
        )
        adapter = CodexAdapter(
            config=CodexAdapterConfig(
                emit_thought_events=False,
            ),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()
        await adapter.on_started("Agent", "An agent")

        msg = make_platform_message(room_id="room-1", content="do something")
        await adapter.on_message(
            msg,
            tools,
            CodexSessionState(),
            None,
            None,
            is_session_bootstrap=False,
            room_id="room-1",
        )

        error_events = [e for e in tools.events_sent if e["message_type"] == "error"]
        assert len(error_events) == 1
        assert "Something went wrong" in error_events[0]["content"]

    @pytest.mark.asyncio
    async def test_cleanup_before_start(self) -> None:
        """Calling on_cleanup on a freshly constructed adapter should not raise."""
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="stdio"),
            client_factory=lambda _config: FakeCodexClient(),
        )
        # No on_started called — cleanup should be safe (idempotent)
        await adapter.on_cleanup("room-x")

    @pytest.mark.asyncio
    async def test_cleanup_clears_pending_approvals(self) -> None:
        """on_cleanup should evict all pending approvals for the given room."""
        fake_client = FakeCodexClient()
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="stdio"),
            client_factory=lambda _config: fake_client,
        )
        await adapter.on_started("Bot", "desc")

        # Manually inject a pending approval for room-1
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()
        adapter._pending_approvals["room-1"] = {
            "tok-1": type(
                "_PA",
                (),
                {
                    "request_id": 1,
                    "method": "item/tool/call",
                    "summary": "test",
                    "created_at": datetime.now(),
                    "future": fut,
                },
            )(),
        }
        # Also register a room thread so the client isn't closed
        adapter._room_threads["room-1"] = "thr-1"
        adapter._room_threads["room-2"] = "thr-2"

        await adapter.on_cleanup("room-1")

        assert "room-1" not in adapter._pending_approvals
        # The future should have been resolved (declined)
        assert fut.done()

    @pytest.mark.asyncio
    async def test_tool_call_validation_error_returns_friendly_message(self) -> None:
        """A ValidationError during tool dispatch returns a user-friendly error."""
        from pydantic import ValidationError as PydanticValidationError

        class ValidationErrorTools(ToolSchemaFakeTools):
            async def execute_tool_call(
                self, tool_name: str, arguments: dict[str, Any]
            ) -> Any:
                call = {"tool_name": tool_name, "arguments": arguments}
                self.tool_calls.append(call)
                # Trigger a real Pydantic ValidationError
                raise PydanticValidationError.from_exception_data(
                    "thenvoi_send_message",
                    [
                        {
                            "type": "missing",
                            "loc": ("content",),
                            "msg": "Field required",
                            "input": arguments,
                        }
                    ],
                )

        events = [
            _event_request(
                99,
                "item/tool/call",
                {
                    "tool": "thenvoi_send_message",
                    "arguments": {},
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = ValidationErrorTools()

        await adapter.on_started("Bot", "desc")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            None,
            None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # The adapter should have responded to the tool call with success=False
        error_responses = [
            (rid, payload)
            for rid, payload in fake_client.responses
            if payload.get("success") is False
        ]
        assert len(error_responses) == 1
        error_text = error_responses[0][1]["contentItems"][0]["text"]
        assert "Invalid arguments for thenvoi_send_message" in error_text


# ===========================================================================
# Phase 1: Structured error reporting
# ===========================================================================


class TestStructuredErrors:
    @pytest.mark.asyncio
    async def test_structured_error_from_error_event(self) -> None:
        """Error events with codexErrorInfo emit structured metadata."""
        events = [
            _event_notification(
                "error",
                {
                    "error": {
                        "message": "Context window exceeded",
                        "codexErrorInfo": {
                            "type": "ContextWindowExceeded",
                            "code": "context_window_exceeded",
                            "retryable": False,
                        },
                    },
                    "willRetry": False,
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", structured_errors=True),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        error_events = [e for e in tools.events_sent if e["message_type"] == "error"]
        assert len(error_events) == 1
        meta = error_events[0]["metadata"]
        assert meta["codex_error_type"] == "ContextWindowExceeded"
        assert meta["codex_suggested_action"] == "compact_context"
        assert meta["codex_is_retryable"] is False
        assert "context window" in error_events[0]["content"].lower()

    @pytest.mark.asyncio
    async def test_structured_error_from_failed_turn(self) -> None:
        """turn/completed with status=failed and codexErrorInfo emits structured error."""
        events = [
            _event_notification(
                "turn/completed",
                {
                    "turn": {
                        "id": "turn-1",
                        "status": "failed",
                        "error": {
                            "message": "Usage limit hit",
                            "codexErrorInfo": {
                                "type": "UsageLimitExceeded",
                                "code": "usage_limit",
                                "retryable": False,
                            },
                        },
                    }
                },
            ),
        ]
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", structured_errors=True),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        error_events = [e for e in tools.events_sent if e["message_type"] == "error"]
        assert len(error_events) == 1
        assert error_events[0]["metadata"]["codex_error_type"] == "UsageLimitExceeded"
        assert (
            error_events[0]["metadata"]["codex_suggested_action"] == "wait_or_upgrade"
        )

    @pytest.mark.asyncio
    async def test_structured_errors_disabled_falls_back_to_plain_text(self) -> None:
        """When structured_errors=False, errors use plain text format."""
        events = [
            _event_notification(
                "error",
                {
                    "error": {
                        "message": "Something failed",
                        "codexErrorInfo": {"type": "ContextWindowExceeded"},
                    },
                    "willRetry": False,
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", structured_errors=False),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        error_events = [e for e in tools.events_sent if e["message_type"] == "error"]
        assert len(error_events) == 1
        assert error_events[0]["content"] == "Codex error: Something failed"
        assert "codex_error_type" not in error_events[0]["metadata"]


# ===========================================================================
# Phase 1: Enriched approvals & session-level acceptance
# ===========================================================================


class TestEnrichedApprovals:
    @pytest.mark.asyncio
    async def test_approve_session_auto_approves_subsequent_requests(self) -> None:
        """After /approve-session, same method type is auto-approved."""
        # First approval request - will be resolved via approve-session
        first_events = [
            _event_request(
                10,
                "item/commandExecution/requestApproval",
                {"command": "npm test"},
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=first_events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", approval_mode="manual"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()
        await adapter.on_started("Agent", "A coding agent")

        # Manually resolve the approval in the background
        async def approve_session_later():
            await asyncio.sleep(0.01)
            await adapter.on_message(
                make_platform_message(content="/approve-session req-10"),
                tools,
                CodexSessionState(),
                participants_msg=None,
                contacts_msg=None,
                is_session_bootstrap=False,
                room_id="room-1",
            )

        task = asyncio.create_task(approve_session_later())
        await adapter.on_message(
            make_platform_message(content="run tests"),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )
        await task

        # Verify session-level was recorded with granular key (command binary)
        assert "commandExecution:npm" in adapter._session_approved.get("room-1", set())
        # Verify approval message mentions session-level
        session_msgs = [
            m for m in tools.messages_sent if "session-level" in m["content"]
        ]
        assert len(session_msgs) >= 1

    @pytest.mark.asyncio
    async def test_approval_audit_trail_emitted(self) -> None:
        """Approval decisions emit audit trail task events."""
        events = [
            _event_request(
                7,
                "item/commandExecution/requestApproval",
                {"command": "rm -rf tmp"},
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(
                transport="ws",
                approval_mode="auto_decline",
            ),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        audit_events = [
            e
            for e in tools.events_sent
            if e["metadata"].get("codex_event_type") == "approval_resolution"
        ]
        assert len(audit_events) == 1
        assert audit_events[0]["metadata"]["codex_approval_decision"] == "decline"
        assert audit_events[0]["metadata"]["codex_decided_by"] == "policy:auto_decline"

    @pytest.mark.asyncio
    async def test_sandbox_command_changes_mode(self) -> None:
        """The /sandbox command sets a per-room override, not mutating global config."""
        fake_client = FakeCodexClient()
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()
        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(content="/sandbox read-only"),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # Per-room override is set, global config is unchanged
        assert adapter._sandbox_overrides.get("room-1") == "read-only"
        assert adapter.config.sandbox is None
        assert "read-only" in tools.messages_sent[0]["content"]

    @pytest.mark.asyncio
    async def test_sandbox_command_is_per_room(self) -> None:
        """Sandbox override in one room does not affect other rooms."""
        fake_client = FakeCodexClient()
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        await adapter.on_started("Agent", "A coding agent")

        adapter._sandbox_overrides["room-1"] = "read-only"

        assert adapter._effective_sandbox("room-1") == "read-only"
        assert adapter._effective_sandbox("room-2") is None

    @pytest.mark.asyncio
    async def test_sandbox_command_rejects_invalid_mode(self) -> None:
        """The /sandbox command rejects invalid modes."""
        fake_client = FakeCodexClient()
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()
        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(content="/sandbox invalid-mode"),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        assert adapter.config.sandbox is None
        assert "room-1" not in adapter._sandbox_overrides
        assert "Invalid sandbox mode" in tools.messages_sent[0]["content"]

    @pytest.mark.asyncio
    async def test_permissions_command_shows_state(self) -> None:
        """/permissions shows current effective permissions."""
        fake_client = FakeCodexClient()
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()
        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(content="/permissions"),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        assert "Effective permissions:" in tools.messages_sent[0]["content"]
        assert "approval_mode: manual" in tools.messages_sent[0]["content"]


# ===========================================================================
# Phase 2: Plan & task lifecycle
# ===========================================================================


class TestPlanAndLifecycle:
    @pytest.mark.asyncio
    async def test_plan_steps_forwarded(self) -> None:
        """turn/plan/updated forwards structured plan steps."""
        events = [
            _event_notification(
                "turn/plan/updated",
                {
                    "plan": {
                        "steps": [
                            {"text": "Read the failing test", "status": "completed"},
                            {"text": "Identify root cause", "status": "inProgress"},
                            {"text": "Apply fix", "status": "pending"},
                        ]
                    }
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", stream_plan_events=True),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        plan_events = [
            e
            for e in tools.events_sent
            if e["metadata"].get("codex_plan_steps") is not None
        ]
        assert len(plan_events) == 1
        steps = plan_events[0]["metadata"]["codex_plan_steps"]
        assert len(steps) == 3
        assert steps[0]["step"] == "Read the failing test"
        assert steps[0]["status"] == "completed"
        assert steps[2]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_plan_steps_not_forwarded_when_disabled(self) -> None:
        """turn/plan/updated is ignored when stream_plan_events=False."""
        events = [
            _event_notification(
                "turn/plan/updated",
                {
                    "plan": {
                        "steps": [
                            {"text": "Step 1", "status": "pending"},
                        ]
                    }
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", stream_plan_events=False),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        plan_events = [
            e
            for e in tools.events_sent
            if e["metadata"].get("codex_plan_steps") is not None
        ]
        assert plan_events == []

    @pytest.mark.asyncio
    async def test_turn_lifecycle_events_emitted(self) -> None:
        """Enriched turn lifecycle events include duration and status."""
        events = [
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(
                transport="ws",
                emit_turn_lifecycle_events=True,
            ),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        lifecycle_events = [
            e
            for e in tools.events_sent
            if e["metadata"].get("codex_event_type") == "turn_lifecycle"
        ]
        assert len(lifecycle_events) == 1
        assert lifecycle_events[0]["metadata"]["codex_turn_status"] == "completed"
        assert "codex_duration_s" in lifecycle_events[0]["metadata"]

    @pytest.mark.asyncio
    async def test_threads_command_lists_mappings(self) -> None:
        """/threads command shows room→thread mappings."""
        events = [
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # Now run /threads
        await adapter.on_message(
            make_platform_message(content="/threads"),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=False,
            room_id="room-1",
        )

        threads_msgs = [
            m
            for m in tools.messages_sent
            if "thread mappings" in m["content"].lower()
            or "active thread" in m["content"].lower()
        ]
        assert len(threads_msgs) >= 1
        assert "room-1" in threads_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_thread_archive_clears_mapping(self) -> None:
        """/thread archive removes the thread mapping."""
        events = [
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )
        assert "room-1" in adapter._room_threads

        await adapter.on_message(
            make_platform_message(content="/thread archive"),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=False,
            room_id="room-1",
        )
        assert "room-1" not in adapter._room_threads
        assert any("archived" in m["content"].lower() for m in tools.messages_sent)


# ===========================================================================
# Phase 3: Real-time streaming
# ===========================================================================


class TestRealtimeStreaming:
    @pytest.mark.asyncio
    async def test_reasoning_delta_streamed_as_thought(self) -> None:
        """item/reasoning/summaryTextDelta forwards as streaming thought."""
        events = [
            _event_notification(
                "item/reasoning/summaryTextDelta",
                {"delta": "Analyzing the code...", "itemId": "item-1"},
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", stream_reasoning_events=True),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        thought_events = [
            e
            for e in tools.events_sent
            if e["message_type"] == "thought" and e["metadata"].get("streaming")
        ]
        assert len(thought_events) == 1
        assert thought_events[0]["content"] == "Analyzing the code..."
        assert thought_events[0]["metadata"]["codex_item_id"] == "item-1"

    @pytest.mark.asyncio
    async def test_reasoning_delta_ignored_when_disabled(self) -> None:
        """Reasoning deltas are skipped when stream_reasoning_events=False."""
        events = [
            _event_notification(
                "item/reasoning/summaryTextDelta",
                {"delta": "Thinking...", "itemId": "item-1"},
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", stream_reasoning_events=False),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        streaming_events = [
            e for e in tools.events_sent if e["metadata"].get("streaming")
        ]
        assert streaming_events == []

    @pytest.mark.asyncio
    async def test_plan_delta_streamed_as_thought(self) -> None:
        """item/plan/delta forwards as streaming thought with plan subtype."""
        events = [
            _event_notification(
                "item/plan/delta",
                {"delta": "Step 1: Read the test", "itemId": "plan-1"},
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", stream_plan_events=True),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        plan_thoughts = [
            e
            for e in tools.events_sent
            if e["message_type"] == "thought" and e["metadata"].get("subtype") == "plan"
        ]
        assert len(plan_thoughts) == 1
        assert plan_thoughts[0]["content"] == "Step 1: Read the test"

    @pytest.mark.asyncio
    async def test_commentary_phase_streamed_as_thought(self) -> None:
        """item/agentMessage/delta with phase=commentary streams as thought."""
        events = [
            _event_notification(
                "item/agentMessage/delta",
                {
                    "delta": "Let me think about this...",
                    "itemId": "msg-1",
                    "phase": "commentary",
                },
            ),
            _event_notification(
                "item/agentMessage/delta",
                {
                    "delta": "Here is the answer.",
                    "itemId": "msg-1",
                    "phase": "final_answer",
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", stream_commentary_events=True),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        commentary_thoughts = [
            e for e in tools.events_sent if e["metadata"].get("subtype") == "commentary"
        ]
        assert len(commentary_thoughts) == 1
        assert commentary_thoughts[0]["content"] == "Let me think about this..."

        # Only the final_answer delta should be in the fallback text
        assert any("Here is the answer." in m["content"] for m in tools.messages_sent)

    @pytest.mark.asyncio
    async def test_commentary_excluded_from_final_text_when_streaming_enabled(
        self,
    ) -> None:
        """When stream_commentary_events=True, commentary is excluded from final_text."""
        events = [
            _event_notification(
                "item/agentMessage/delta",
                {
                    "delta": "thinking...",
                    "itemId": "msg-1",
                    "phase": "commentary",
                },
            ),
            _event_notification(
                "item/agentMessage/delta",
                {
                    "delta": "real answer",
                    "itemId": "msg-1",
                    "phase": "final_answer",
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", stream_commentary_events=True),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # Only the final_answer delta should be in the message text
        assert len(tools.messages_sent) == 1
        assert tools.messages_sent[0]["content"] == "real answer"

    @pytest.mark.asyncio
    async def test_commentary_included_in_final_text_when_streaming_disabled(
        self,
    ) -> None:
        """When stream_commentary_events=False (default), commentary accumulates into final_text."""
        events = [
            _event_notification(
                "item/agentMessage/delta",
                {
                    "delta": "thinking...",
                    "itemId": "msg-1",
                    "phase": "commentary",
                },
            ),
            _event_notification(
                "item/agentMessage/delta",
                {
                    "delta": "real answer",
                    "itemId": "msg-1",
                    "phase": "final_answer",
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", stream_commentary_events=False),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # Both phases should be accumulated (backward compatible)
        assert len(tools.messages_sent) == 1
        assert tools.messages_sent[0]["content"] == "thinking...real answer"


# ===========================================================================
# Phase 4: Diffs + token usage
# ===========================================================================


class TestDiffsAndTokenUsage:
    @pytest.mark.asyncio
    async def test_diff_event_forwarded(self) -> None:
        """turn/diff/updated forwards as tool_result when enabled."""
        events = [
            _event_notification(
                "turn/diff/updated",
                {
                    "diff": "--- a/src/app.py\n+++ b/src/app.py\n@@ ...",
                    "files": ["src/app.py"],
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(
                transport="ws",
                emit_diff_events=True,
                enable_execution_reporting=True,
            ),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        diff_events = [
            e
            for e in tools.events_sent
            if e["metadata"].get("codex_event_type") == "turn_diff"
        ]
        assert len(diff_events) == 1
        assert diff_events[0]["metadata"]["codex_files_changed"] == ["src/app.py"]
        content = json.loads(diff_events[0]["content"])
        assert content["name"] == "codex_diff"
        assert "app.py" in content["output"]

    @pytest.mark.asyncio
    async def test_diff_event_requires_execution_reporting(self) -> None:
        """Diffs are not forwarded without enable_execution_reporting."""
        events = [
            _event_notification(
                "turn/diff/updated",
                {"diff": "some diff", "files": ["f.py"]},
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(
                transport="ws",
                emit_diff_events=True,
                enable_execution_reporting=False,
            ),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        diff_events = [
            e
            for e in tools.events_sent
            if e["metadata"].get("codex_event_type") == "turn_diff"
        ]
        assert diff_events == []

    @pytest.mark.asyncio
    async def test_token_usage_tracked_and_emitted(self) -> None:
        """thread/tokenUsage/updated events are tracked and emitted."""
        events = [
            _event_notification(
                "thread/tokenUsage/updated",
                {
                    "usage": {
                        "inputTokens": 15000,
                        "outputTokens": 3200,
                        "reasoningTokens": 8000,
                        "totalTokens": 26200,
                    }
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", emit_token_usage_events=True),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        usage_events = [
            e
            for e in tools.events_sent
            if e["metadata"].get("codex_event_type") == "token_usage"
        ]
        assert len(usage_events) == 1
        assert usage_events[0]["metadata"]["codex_input_tokens"] == 15000
        assert usage_events[0]["metadata"]["codex_output_tokens"] == 3200
        assert usage_events[0]["metadata"]["codex_total_tokens"] == 26200

    @pytest.mark.asyncio
    async def test_token_usage_ignored_when_disabled(self) -> None:
        """Token usage events are tracked internally but not emitted when disabled."""
        events = [
            _event_notification(
                "thread/tokenUsage/updated",
                {
                    "usage": {
                        "inputTokens": 1000,
                        "outputTokens": 500,
                        "totalTokens": 1500,
                    }
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", emit_token_usage_events=False),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        usage_events = [
            e
            for e in tools.events_sent
            if e["metadata"].get("codex_event_type") == "token_usage"
        ]
        assert usage_events == []

        # But internal tracking still works
        thread_id = adapter._room_threads.get("room-1")
        assert thread_id is not None
        usage = adapter._token_usage.get(thread_id)
        assert usage is not None
        assert usage.input_tokens == 1000

    @pytest.mark.asyncio
    async def test_usage_command_shows_token_usage(self) -> None:
        """/usage command shows accumulated token usage."""
        events = [
            _event_notification(
                "thread/tokenUsage/updated",
                {
                    "usage": {
                        "inputTokens": 5000,
                        "outputTokens": 1000,
                        "reasoningTokens": 2000,
                        "totalTokens": 8000,
                    }
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # Now run /usage
        await adapter.on_message(
            make_platform_message(content="/usage"),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=False,
            room_id="room-1",
        )

        usage_msgs = [
            m for m in tools.messages_sent if "token usage" in m["content"].lower()
        ]
        assert len(usage_msgs) >= 1
        assert "8,000" in usage_msgs[0]["content"]


# ===========================================================================
# Types unit tests
# ===========================================================================


class TestCodexTypes:
    def test_build_structured_error_metadata_known_type(self) -> None:
        from thenvoi.integrations.codex.types import build_structured_error_metadata

        error_obj = {
            "message": "Context overflow",
            "codexErrorInfo": {
                "type": "ContextWindowExceeded",
                "code": "ctx_exceeded",
                "retryable": False,
            },
        }
        content, meta = build_structured_error_metadata(
            error_obj, thread_id="t1", turn_id="turn-1"
        )
        assert "context window" in content.lower()
        assert meta["codex_error_type"] == "ContextWindowExceeded"
        assert meta["codex_suggested_action"] == "compact_context"
        assert meta["codex_thread_id"] == "t1"
        assert meta["codex_turn_id"] == "turn-1"

    def test_build_structured_error_metadata_unknown_type(self) -> None:
        from thenvoi.integrations.codex.types import build_structured_error_metadata

        error_obj = {
            "message": "Something weird happened",
            "codexErrorInfo": {"type": "UnknownError"},
        }
        content, meta = build_structured_error_metadata(error_obj)
        assert content == "Something weird happened"
        assert meta["codex_error_type"] == "UnknownError"
        assert meta["codex_suggested_action"] is None

    def test_parse_plan_steps(self) -> None:
        from thenvoi.integrations.codex.types import parse_plan_steps

        params = {
            "plan": {
                "steps": [
                    {"text": "Step 1", "status": "completed"},
                    {"text": "Step 2", "status": "inProgress"},
                    {"text": "Step 3", "status": "pending"},
                ]
            }
        }
        steps = parse_plan_steps(params)
        assert len(steps) == 3
        assert steps[0].step == "Step 1"
        assert steps[0].status == "completed"
        assert steps[2].status == "pending"

    def test_parse_plan_steps_string_entries(self) -> None:
        from thenvoi.integrations.codex.types import parse_plan_steps

        params = {"plan": {"steps": ["Read code", "Fix bug"]}}
        steps = parse_plan_steps(params)
        assert len(steps) == 2
        assert steps[0].step == "Read code"
        assert steps[0].status == "pending"

    def test_codex_token_usage_update(self) -> None:
        from thenvoi.integrations.codex.types import CodexTokenUsage

        usage = CodexTokenUsage()
        usage.update(
            {
                "usage": {
                    "inputTokens": 1000,
                    "outputTokens": 500,
                    "reasoningTokens": 200,
                    "totalTokens": 1700,
                }
            }
        )
        assert usage.input_tokens == 1000
        assert usage.output_tokens == 500
        assert usage.reasoning_tokens == 200
        assert usage.total_tokens == 1700
        meta = usage.to_metadata()
        assert meta["codex_input_tokens"] == 1000
        assert "1,700" in usage.format_summary()

    def test_config_new_flags_default_false(self) -> None:
        """All new config flags default to False (except structured_errors=True)."""
        config = CodexAdapterConfig()
        assert config.structured_errors is True
        assert config.stream_reasoning_events is False
        assert config.stream_plan_events is False
        assert config.stream_commentary_events is False
        assert config.emit_diff_events is False
        assert config.emit_token_usage_events is False
        assert config.emit_turn_lifecycle_events is False

    def test_session_approval_key_granular_for_commands(self) -> None:
        """Session approval key includes command binary for command executions."""
        key = CodexAdapter._session_approval_key(
            "item/commandExecution/requestApproval", {"command": "npm test"}
        )
        assert key == "commandExecution:npm"

    def test_session_approval_key_empty_for_missing_command(self) -> None:
        """Session approval key returns empty when command is missing (no wildcard)."""
        key = CodexAdapter._session_approval_key(
            "item/commandExecution/requestApproval", {}
        )
        assert key == ""

    def test_session_approval_key_method_for_file_changes(self) -> None:
        """Session approval key uses full method for file changes."""
        key = CodexAdapter._session_approval_key(
            "item/fileChange/requestApproval", {"reason": "update"}
        )
        assert key == "item/fileChange/requestApproval"


class TestSessionAutoApproval:
    @pytest.mark.asyncio
    async def test_session_auto_approves_matching_command_binary(self) -> None:
        """After session-level approval for npm, a new npm command is auto-approved."""
        # Two command execution requests in one turn — first will be manually
        # approved, second should be auto-approved by session policy.
        events = [
            _event_request(
                20,
                "item/commandExecution/requestApproval",
                {"command": "npm install"},
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(
                transport="ws",
                approval_mode="manual",
            ),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()
        await adapter.on_started("Agent", "A coding agent")

        # Pre-seed the session-approved set as if /approve-session was used for npm
        adapter._session_approved["room-1"] = {"commandExecution:npm"}

        await adapter.on_message(
            make_platform_message(content="install deps"),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # The request should have been auto-approved (no manual prompt)
        responses = fake_client.responses
        assert any(result.get("decision") == "accept" for _, result in responses)

    @pytest.mark.asyncio
    async def test_session_does_not_auto_approve_different_binary(self) -> None:
        """Session approval for npm does NOT auto-approve rm commands."""
        events = [
            _event_request(
                30,
                "item/commandExecution/requestApproval",
                {"command": "rm -rf tmp"},
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(
                transport="ws",
                approval_mode="auto_decline",
            ),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()
        await adapter.on_started("Agent", "A coding agent")

        # Pre-seed session approval for npm only
        adapter._session_approved["room-1"] = {"commandExecution:npm"}

        await adapter.on_message(
            make_platform_message(content="clean up"),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # rm should have been declined by the auto_decline policy, not auto-approved
        responses = fake_client.responses
        assert any(result.get("decision") == "decline" for _, result in responses)


class TestCleanup:
    @pytest.mark.asyncio
    async def test_on_cleanup_removes_per_room_token_usage(self) -> None:
        """on_cleanup for a room also removes the thread's token usage."""
        events = [
            _event_notification(
                "thread/tokenUsage/updated",
                {
                    "usage": {
                        "inputTokens": 500,
                        "outputTokens": 100,
                        "totalTokens": 600,
                    }
                },
            ),
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # Verify usage was tracked
        thread_id = adapter._room_threads.get("room-1")
        assert thread_id is not None
        assert thread_id in adapter._token_usage

        # Add a second room so cleanup doesn't close the client entirely
        adapter._room_threads["room-2"] = "other-thread"

        await adapter.on_cleanup("room-1")
        assert thread_id not in adapter._token_usage


class TestAuditCap:
    def test_audit_trail_capped_at_limit(self) -> None:
        """Approval audit trail is capped at max_approval_audit_per_room."""
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws", max_approval_audit_per_room=5),
            client_factory=lambda _config: FakeCodexClient(),
        )
        for i in range(10):
            adapter._record_approval_audit(
                room_id="room-1",
                request_id=str(i),
                method="item/commandExecution/requestApproval",
                decision="accept",
                decided_by="test",
            )
        audit = adapter._approval_audit["room-1"]
        assert len(audit) == 5
        # Should keep the most recent entries
        assert audit[0].request_id == "5"
        assert audit[-1].request_id == "9"


class TestReviewFixes:
    """Tests for issues identified in PR review."""

    @pytest.mark.asyncio
    async def test_sandbox_command_blocked_when_sandbox_policy_set(self) -> None:
        """/sandbox is rejected when sandbox_policy is configured."""
        fake_client = FakeCodexClient()
        adapter = CodexAdapter(
            config=CodexAdapterConfig(
                transport="ws",
                sandbox_policy={"type": "readOnly"},
            ),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()
        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(content="/sandbox workspace-write"),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        assert "room-1" not in adapter._sandbox_overrides
        assert "Cannot override sandbox" in tools.messages_sent[0]["content"]

    @pytest.mark.asyncio
    async def test_thread_archive_clears_raw_history(self) -> None:
        """/thread archive also clears raw history and injection state."""
        events = [
            _event_notification(
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
        fake_client = FakeCodexClient(events=events)
        adapter = CodexAdapter(
            config=CodexAdapterConfig(transport="ws"),
            client_factory=lambda _config: fake_client,
        )
        tools = ToolSchemaFakeTools()

        await adapter.on_started("Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # Seed raw history and injection flag
        adapter._raw_history_by_room["room-1"] = [{"role": "user", "content": "hi"}]
        adapter._needs_history_injection.add("room-1")

        await adapter.on_message(
            make_platform_message(content="/thread archive"),
            tools,
            CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=False,
            room_id="room-1",
        )

        assert "room-1" not in adapter._raw_history_by_room
        assert "room-1" not in adapter._needs_history_injection

    def test_token_usage_update_handles_zero_values(self) -> None:
        """CodexTokenUsage.update() correctly handles explicit zero values."""
        from thenvoi.integrations.codex.types import CodexTokenUsage

        usage = CodexTokenUsage()
        usage.update(
            {
                "usage": {
                    "inputTokens": 0,
                    "outputTokens": 100,
                    "reasoningTokens": 0,
                    "totalTokens": 100,
                }
            }
        )
        assert usage.input_tokens == 0
        assert usage.output_tokens == 100
        assert usage.reasoning_tokens == 0
        assert usage.total_tokens == 100

    def test_session_approval_key_empty_prevents_wildcard_match(self) -> None:
        """Empty session key from missing command cannot match any session set."""
        key = CodexAdapter._session_approval_key(
            "item/commandExecution/requestApproval", {}
        )
        # Empty key is falsy, so `key and key in session_set` is always False
        assert not key
        assert not (key and key in {"commandExecution:npm"})
