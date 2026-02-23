"""Tests for CodexAdapter."""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime
from typing import Any
from uuid import uuid4

import pytest

from thenvoi.adapters.codex import CodexAdapter, CodexAdapterConfig
from thenvoi.core.types import PlatformMessage
from thenvoi.integrations.codex import CodexJsonRpcError, RpcEvent
from thenvoi.integrations.codex.types import CodexSessionState
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
    def get_openai_tool_schemas(self) -> list[dict[str, Any]]:
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
            }
        ]


class FakeCodexClient:
    """Minimal fake transport client for adapter tests."""

    def __init__(
        self,
        *,
        events: list[RpcEvent] | None = None,
        resume_error: Exception | None = None,
    ) -> None:
        self.connected = False
        self.initialized = False
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.responses: list[tuple[int | str, dict[str, Any]]] = []
        self.response_errors: list[tuple[int | str, int, str]] = []
        self.closed = False
        self._events = deque(events or [])
        self._resume_error = resume_error
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
        self.requests.append((method, payload))

        if method == "model/list":
            return {"data": [{"id": "gpt-5.3-codex", "hidden": False}]}

        if method == "thread/resume":
            if self._resume_error is not None:
                raise self._resume_error
            return {"thread": {"id": payload.get("threadId", "thr-resumed")}}

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
        assert thread_start["dynamicTools"][0]["name"] == "thenvoi_send_message"

        assert len(tools.messages_sent) == 1
        assert tools.messages_sent[0]["content"] == "harness-ok"
        assert tools.messages_sent[0]["mentions"][0]["id"] == "user-1"

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
        assert methods.count("model/list") >= 2
        assert len(tools.messages_sent) == 1
        assert "Available models" in tools.messages_sent[0]["content"]

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
