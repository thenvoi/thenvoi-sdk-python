"""Codex app-server adapter."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Protocol

from thenvoi.converters.codex import CodexHistoryConverter
from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.core.simple_adapter import SimpleAdapter
from thenvoi.core.types import AgentInput, PlatformMessage
from thenvoi.integrations.codex import (
    CodexJsonRpcError,
    CodexStdioClient,
    CodexWebSocketClient,
    RpcEvent,
)
from thenvoi.integrations.codex.types import CodexSessionState
from thenvoi.runtime.custom_tools import (
    CustomToolDef,
    custom_tool_to_openai_schema,
    execute_custom_tool,
    find_custom_tool,
)
from thenvoi.runtime.prompts import render_system_prompt

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

TransportKind = Literal["stdio", "ws"]
ApprovalMode = Literal["auto_accept", "auto_decline", "manual"]
ApprovalDecision = Literal["accept", "decline"]
RoleProfile = Literal["coding", "planner", "reviewer"]

_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
_REASONING_SUMMARIES = {"auto", "concise", "detailed", "none"}

# Platform tools whose execution should not be reported as tool_call/tool_result
# events — they already produce visible output (messages or events) on the platform.
_SILENT_REPORTING_TOOLS: frozenset[str] = frozenset(
    {
        "thenvoi_send_message",
        "thenvoi_send_event",
        "setmodel",
        "setreasoning",
    }
)


# ---------------------------------------------------------------------------
# Self-configuration tools — let Codex change its own model/reasoning at runtime
# ---------------------------------------------------------------------------


class SetModelInput(BaseModel):
    """Switch the model used for subsequent turns. Call this when a different model would be more appropriate for the task (e.g. a faster model for simple queries, a stronger model for complex reasoning)."""

    model: str = Field(description="Model ID to use (e.g. 'gpt-5.3-codex', 'gpt-5.2').")


class SetReasoningInput(BaseModel):
    """Adjust reasoning effort and summary detail for subsequent turns. Use higher effort for complex problems and lower effort for straightforward tasks."""

    effort: str | None = Field(
        default=None,
        description="Reasoning effort level: none, minimal, low, medium, high, or xhigh. Omit to keep current.",
    )
    summary: str | None = Field(
        default=None,
        description="Reasoning summary detail: auto, concise, detailed, or none. Omit to keep current.",
    )


# Hardcoded default — update when OpenAI rotates model IDs.
# Override at runtime via CodexAdapterConfig.model or CODEX_MODEL env var.
_DEFAULT_MODEL = "gpt-5.3-codex"

_ROLE_SECTION: dict[RoleProfile, str] = {
    "coding": "Primary mode: implement and validate code changes end-to-end.",
    "planner": "Primary mode: produce execution plans, constraints, and phased delivery.",
    "reviewer": "Primary mode: review behavior risks, regressions, and missing tests.",
}


class _CodexClientProtocol(Protocol):
    async def connect(self) -> None: ...

    async def initialize(
        self,
        *,
        client_name: str,
        client_title: str,
        client_version: str,
        experimental_api: bool = False,
        opt_out_notification_methods: list[str] | None = None,
    ) -> dict[str, Any]: ...

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        retry_on_overload: bool = True,
    ) -> dict[str, Any]: ...

    async def recv_event(self, timeout_s: float | None = None) -> RpcEvent: ...

    async def respond(self, request_id: int | str, result: dict[str, Any]) -> None: ...

    async def respond_error(
        self,
        request_id: int | str,
        *,
        code: int,
        message: str,
        data: Any | None = None,
    ) -> None: ...

    async def close(self) -> None: ...


@dataclass
class _PendingApproval:
    request_id: int | str
    method: str
    summary: str
    created_at: datetime
    future: asyncio.Future[str]


@dataclass
class CodexAdapterConfig:
    """Runtime configuration for Codex adapter sessions."""

    transport: TransportKind = "stdio"
    model: str | None = None
    reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None
    ) = None
    reasoning_summary: Literal["auto", "concise", "detailed", "none"] | None = None
    cwd: str | None = None
    approval_policy: str = "never"
    personality: Literal["friendly", "pragmatic", "none"] = "pragmatic"
    sandbox: str | None = None
    sandbox_policy: dict[str, Any] | None = None
    role: RoleProfile = "coding"
    system_prompt: str | None = None
    custom_section: str = ""
    include_base_instructions: bool = True
    experimental_api: bool = True
    enable_task_events: bool = True
    emit_turn_task_markers: bool = False
    emit_thought_events: bool = False
    fallback_send_agent_text: bool = True
    approval_mode: ApprovalMode = "manual"
    approval_text_notifications: bool = True
    approval_wait_timeout_s: float = 300.0
    approval_timeout_decision: ApprovalDecision = "decline"
    turn_timeout_s: float = 180.0
    client_name: str = "thenvoi_codex_adapter"
    client_title: str = "Thenvoi Codex Adapter"
    client_version: str = "0.1.0"
    codex_command: tuple[str, ...] | None = None
    codex_env: dict[str, str] | None = None
    codex_ws_url: str = "ws://127.0.0.1:8765"
    enable_execution_reporting: bool = False
    enable_self_config_tools: bool = False
    additional_dynamic_tools: list[dict[str, Any]] = field(default_factory=list)
    inject_history_on_resume_failure: bool = True
    max_history_messages: int = 50
    fallback_models: tuple[str, ...] = ("gpt-5.2", "gpt-5.3-codex")
    max_pending_approvals_per_room: int = 50


class CodexAdapter(SimpleAdapter[CodexSessionState]):
    """
    Codex adapter backed by codex app-server (stdio or websocket transport).

    One Thenvoi room maps to one Codex thread. Mapping is persisted in task
    events metadata and restored via CodexHistoryConverter on bootstrap.
    """

    def __init__(
        self,
        config: CodexAdapterConfig | None = None,
        *,
        additional_tools: list[CustomToolDef] | None = None,
        history_converter: CodexHistoryConverter | None = None,
        client_factory: Callable[[CodexAdapterConfig], _CodexClientProtocol]
        | None = None,
    ) -> None:
        super().__init__(history_converter=history_converter or CodexHistoryConverter())
        self.config = config or CodexAdapterConfig()
        self._custom_tools: list[CustomToolDef] = list(additional_tools or [])
        if self.config.enable_self_config_tools:
            self._custom_tools.extend(self._build_self_config_tools())
        self._client_factory = client_factory
        self._client: _CodexClientProtocol | None = None
        self._initialized = False
        self._selected_model: str | None = None
        self._model_explicitly_set: bool = bool(self.config.model)
        self._system_prompt: str = ""
        self._room_threads: dict[str, str] = {}
        self._prompt_injected_rooms: set[str] = set()
        self._task_titles_by_id: dict[str, str] = {}
        self._pending_approvals: dict[str, dict[str, _PendingApproval]] = {}
        self._raw_history_by_room: dict[str, list[dict[str, Any]]] = {}
        self._needs_history_injection: set[str] = set()
        # Single client receive queue means turn processing must be serialized.
        self._rpc_lock = asyncio.Lock()

    def _build_self_config_tools(self) -> list[CustomToolDef]:
        """Build custom tools that let Codex change its own model/reasoning.

        Note: ``_handle_set_model`` and ``_handle_set_reasoning`` closures
        mutate adapter state (``config.model``, ``_selected_model``, etc.).
        They are safe because they are always called inside the
        ``_handle_server_request`` path which holds ``_rpc_lock``.
        """
        adapter = self

        def _handle_set_model(inp: SetModelInput) -> str:
            adapter.config.model = inp.model
            adapter._selected_model = inp.model
            adapter._model_explicitly_set = True
            return f"Model changed to {inp.model} for subsequent turns."

        def _handle_set_reasoning(inp: SetReasoningInput) -> str:
            parts: list[str] = []
            if inp.effort is not None:
                if inp.effort not in _REASONING_EFFORTS:
                    return (
                        f"Invalid reasoning effort '{inp.effort}'. "
                        f"Valid: {', '.join(sorted(_REASONING_EFFORTS))}."
                    )
                adapter.config.reasoning_effort = inp.effort  # type: ignore[assignment]  # Literal narrowed by Pydantic validation
                parts.append(f"effort={inp.effort}")
            if inp.summary is not None:
                if inp.summary not in _REASONING_SUMMARIES:
                    return (
                        f"Invalid reasoning summary '{inp.summary}'. "
                        f"Valid: {', '.join(sorted(_REASONING_SUMMARIES))}."
                    )
                adapter.config.reasoning_summary = inp.summary  # type: ignore[assignment]  # Literal narrowed by Pydantic validation
                parts.append(f"summary={inp.summary}")
            if not parts:
                return (
                    f"No changes. Current: effort={adapter.config.reasoning_effort or 'default'}, "
                    f"summary={adapter.config.reasoning_summary or 'default'}."
                )
            return f"Reasoning updated: {', '.join(parts)}."

        return [
            (SetModelInput, _handle_set_model),
            (SetReasoningInput, _handle_set_reasoning),
        ]

    async def on_started(self, agent_name: str, agent_description: str) -> None:
        await super().on_started(agent_name, agent_description)
        self._build_system_prompt()
        async with self._rpc_lock:
            await self._ensure_client_ready()
        self._log_startup_config(agent_name)

    def _log_startup_config(self, agent_name: str) -> None:
        logger.info(
            "Codex adapter started: agent=%s, transport=%s, model=%s, "
            "sandbox=%s, approval_mode=%s, role=%s, "
            "execution_reporting=%s, self_config_tools=%s, "
            "task_events=%s, turn_markers=%s, thought_events=%s",
            agent_name,
            self.config.transport,
            self._selected_model or self.config.model or "auto",
            self.config.sandbox or "default",
            self.config.approval_mode,
            self.config.role,
            self.config.enable_execution_reporting,
            self.config.enable_self_config_tools,
            self.config.enable_task_events,
            self.config.emit_turn_task_markers,
            self.config.emit_thought_events,
        )

    async def on_event(self, inp: AgentInput) -> None:
        if (
            self.config.inject_history_on_resume_failure
            and inp.is_session_bootstrap
            and inp.history.raw
        ):
            self._raw_history_by_room[inp.room_id] = inp.history.raw
        await super().on_event(inp)

    async def on_message(
        self,
        msg: PlatformMessage,
        tools: AgentToolsProtocol,
        history: CodexSessionState,
        participants_msg: str | None,
        contacts_msg: str | None,
        *,
        is_session_bootstrap: bool,
        room_id: str,
    ) -> None:
        command = self._extract_local_command(msg.content)
        if command is not None and command[0] in {"approve", "decline", "approvals"}:
            handled = await self._handle_approval_command(
                tools=tools,
                msg=msg,
                room_id=room_id,
                command=command[0],
                args=command[1],
            )
            if handled:
                return

        async with self._rpc_lock:
            await self._ensure_client_ready()
            assert self._client is not None

            if command is not None:
                handled = await self._handle_local_command(
                    tools=tools,
                    msg=msg,
                    history=history,
                    room_id=room_id,
                    command=command[0],
                    args=command[1],
                )
                if handled:
                    return

            thread_id = await self._ensure_thread(
                room_id=room_id,
                history=history,
                tools=tools,
                is_session_bootstrap=is_session_bootstrap,
            )

            turn_input, has_pending_prompt_injection = self._build_turn_input(
                msg=msg,
                participants_msg=participants_msg,
                contacts_msg=contacts_msg,
                room_id=room_id,
            )

            turn_params: dict[str, Any] = {
                "threadId": thread_id,
                "input": turn_input,
            }
            self._apply_turn_overrides(turn_params)

            turn_started = await self._start_turn_with_model_fallback(turn_params)
            if has_pending_prompt_injection:
                self._prompt_injected_rooms.add(room_id)
            turn = turn_started.get("turn") if isinstance(turn_started, dict) else {}
            turn_id = str((turn or {}).get("id") or "")

            if self.config.enable_task_events and self.config.emit_turn_task_markers:
                await tools.send_event(
                    content=self._build_task_event_content(
                        task_id=turn_id or None,
                        task="Codex turn",
                        status="started",
                        summary=f"Thread: {thread_id}",
                    ),
                    message_type="task",
                    metadata={
                        "codex_thread_id": thread_id,
                        "codex_turn_id": turn_id or None,
                        "codex_room_id": room_id,
                    },
                )

            saw_send_message_tool = False
            final_text = ""
            turn_status = "failed"
            turn_error = ""

            try:
                while True:
                    event = await self._client.recv_event(
                        timeout_s=self.config.turn_timeout_s
                    )
                    if event.kind == "request":
                        used_send_message = await self._handle_server_request(
                            tools=tools,
                            msg=msg,
                            room_id=room_id,
                            event=event,
                        )
                        saw_send_message_tool = (
                            saw_send_message_tool or used_send_message
                        )
                        continue

                    params = event.params if isinstance(event.params, dict) else {}

                    if event.method in {
                        "codex/event/task_started",
                        "codex/event/task_complete",
                    }:
                        await self._forward_raw_task_event(
                            tools=tools,
                            room_id=room_id,
                            thread_id=thread_id,
                            turn_id=turn_id or None,
                            method=event.method,
                            params=params,
                        )
                        continue

                    if event.method == "error":
                        error_obj = params.get("error") or {}
                        if isinstance(error_obj, dict):
                            error_msg = error_obj.get("message", "")
                        else:
                            error_msg = str(error_obj)
                        will_retry = bool(params.get("willRetry", False))
                        if will_retry:
                            logger.warning(
                                "Codex transient error (will retry): %s",
                                error_msg,
                            )
                        else:
                            logger.error("Codex error: %s", error_msg)
                            await tools.send_event(
                                content=f"Codex error: {error_msg}",
                                message_type="error",
                                metadata={
                                    "codex_room_id": room_id,
                                    "codex_thread_id": thread_id,
                                    "codex_turn_id": turn_id or None,
                                },
                            )
                        continue

                    if event.method == "item/agentMessage/delta":
                        delta = params.get("delta")
                        if isinstance(delta, str):
                            final_text += delta
                        continue

                    if event.method == "item/completed":
                        item = params.get("item") if isinstance(params, dict) else {}
                        if isinstance(item, dict):
                            item_type = item.get("type")
                            if item_type == "agentMessage":
                                text = item.get("text")
                                if isinstance(text, str) and text:
                                    final_text = text
                            else:
                                await self._emit_item_completed_events(
                                    tools=tools,
                                    item=item,
                                    room_id=room_id,
                                    thread_id=thread_id,
                                    turn_id=turn_id or None,
                                )
                        continue

                    if event.method == "transport/closed":
                        turn_status = "failed"
                        turn_error = "Codex transport closed unexpectedly"
                        break

                    if event.method == "turn/completed":
                        turn_payload = (
                            params.get("turn")
                            if isinstance(params.get("turn"), dict)
                            else {}
                        )
                        event_turn_id = str(turn_payload.get("id") or "")
                        if turn_id and event_turn_id and event_turn_id != turn_id:
                            continue
                        turn_status = str(turn_payload.get("status") or "failed")
                        turn_error = self._extract_turn_error(turn_payload)
                        break
            except asyncio.TimeoutError:
                logger.error(
                    "Codex turn timed out after %ss (thread=%s, turn=%s)",
                    self.config.turn_timeout_s,
                    thread_id,
                    turn_id,
                )
                if turn_id:
                    try:
                        await self._client.request(
                            "turn/interrupt", {"turnId": turn_id}
                        )
                    except Exception:
                        logger.warning("Failed to send turn/interrupt after timeout")
                turn_status = "interrupted"
                turn_error = "Turn timed out"

            await self._emit_turn_outcome(
                tools=tools,
                msg=msg,
                room_id=room_id,
                thread_id=thread_id,
                turn_id=turn_id or None,
                turn_status=turn_status,
                turn_error=turn_error,
                final_text=final_text,
                saw_send_message_tool=saw_send_message_tool,
            )

    async def on_cleanup(self, room_id: str) -> None:
        async with self._rpc_lock:
            self._room_threads.pop(room_id, None)
            self._prompt_injected_rooms.discard(room_id)
            self._raw_history_by_room.pop(room_id, None)
            self._needs_history_injection.discard(room_id)
            self._clear_pending_approvals_for_room(room_id)
            if self._room_threads:
                return
            if self._client is None:
                return
            try:
                await self._client.close()
            finally:
                self._client = None
                self._initialized = False
                self._selected_model = None
                self._task_titles_by_id.clear()
                self._pending_approvals.clear()

    async def _ensure_client_ready(self) -> None:
        if self._client is None:
            self._client = self._build_client(self.config)

        if not self._initialized:
            await self._client.connect()
            await self._client.initialize(
                client_name=self.config.client_name,
                client_title=self.config.client_title,
                client_version=self.config.client_version,
                experimental_api=self.config.experimental_api,
            )
            self._selected_model = await self._select_model()
            self._initialized = True

    def _build_client(self, config: CodexAdapterConfig) -> _CodexClientProtocol:
        if self._client_factory is not None:
            return self._client_factory(config)

        if config.transport == "ws":
            return CodexWebSocketClient(ws_url=config.codex_ws_url)

        return CodexStdioClient(
            command=config.codex_command,
            cwd=config.cwd,
            env=config.codex_env,
        )

    async def _select_model(self) -> str:
        if self.config.model:
            return self.config.model

        assert self._client is not None
        try:
            result = await self._client.request("model/list", {})
        except Exception:
            logger.warning("model/list failed; using fallback model id", exc_info=True)
            return _DEFAULT_MODEL

        data = result.get("data") if isinstance(result, dict) else None
        if not isinstance(data, list):
            return _DEFAULT_MODEL

        visible_models = [
            entry
            for entry in data
            if isinstance(entry, dict)
            and isinstance(entry.get("id"), str)
            and not bool(entry.get("hidden", False))
        ]
        for entry in visible_models:
            model_id = str(entry["id"])
            if "codex" in model_id:
                return model_id
        if visible_models:
            return str(visible_models[0]["id"])
        return _DEFAULT_MODEL

    async def _ensure_thread(
        self,
        *,
        room_id: str,
        history: CodexSessionState,
        tools: AgentToolsProtocol,
        is_session_bootstrap: bool,
    ) -> str:
        thread_id = self._room_threads.get(room_id)
        if thread_id:
            return thread_id

        assert self._client is not None

        if is_session_bootstrap and history.has_thread():
            try:
                result = await self._client.request(
                    "thread/resume",
                    {
                        "threadId": history.thread_id,
                        "personality": self.config.personality,
                    },
                )
                resumed = result.get("thread", {}) if isinstance(result, dict) else {}
                thread_id = str(resumed.get("id") or history.thread_id or "")
                if thread_id:
                    self._room_threads[room_id] = thread_id
                    self._raw_history_by_room.pop(room_id, None)
                    if self.config.enable_task_events:
                        await tools.send_event(
                            content=self._build_task_event_content(
                                task_id=thread_id,
                                task="Codex thread",
                                status="resumed",
                                summary=f"Room: {room_id}",
                            ),
                            message_type="task",
                            metadata={
                                "codex_thread_id": thread_id,
                                "codex_room_id": room_id,
                                "codex_resumed": True,
                            },
                        )
                    return thread_id
            except CodexJsonRpcError as exc:
                logger.warning(
                    "thread/resume failed for room %s thread %s: %s",
                    room_id,
                    history.thread_id,
                    exc,
                )
                if self.config.inject_history_on_resume_failure:
                    self._needs_history_injection.add(room_id)
        else:
            # Not a bootstrap resume — clean up any stashed history
            self._raw_history_by_room.pop(room_id, None)

        dynamic_tools = self._build_dynamic_tools(tools)
        start_params: dict[str, Any] = {
            "model": self._selected_model,
            "cwd": self.config.cwd,
            "approvalPolicy": self.config.approval_policy,
            "personality": self.config.personality,
            "dynamicTools": dynamic_tools,
        }
        self._apply_thread_sandbox(start_params)

        started = await self._client.request("thread/start", start_params)
        thread = started.get("thread") if isinstance(started, dict) else {}
        thread_id = str((thread or {}).get("id") or "")
        if not thread_id:
            raise RuntimeError("Codex thread/start returned no thread id")

        self._room_threads[room_id] = thread_id

        if self.config.enable_task_events:
            await tools.send_event(
                content=self._build_task_event_content(
                    task_id=thread_id,
                    task="Codex thread",
                    status="mapped",
                    summary=f"Transport: {self.config.transport}",
                ),
                message_type="task",
                metadata={
                    "codex_thread_id": thread_id,
                    "codex_room_id": room_id,
                    "codex_created_at": datetime.now(timezone.utc).isoformat(),
                    "codex_transport": self.config.transport,
                },
            )

        return thread_id

    def _build_dynamic_tools(self, tools: AgentToolsProtocol) -> list[dict[str, Any]]:
        dynamic_tools: list[dict[str, Any]] = []
        seen: set[str] = set()

        for schema in tools.get_openai_tool_schemas():
            if not isinstance(schema, dict):
                continue

            if schema.get("type") == "function":
                function = schema.get("function") if isinstance(schema, dict) else {}
                if not isinstance(function, dict):
                    continue
                name = function.get("name")
                if not isinstance(name, str) or not name or name in seen:
                    continue
                dynamic_tools.append(
                    {
                        "name": name,
                        "description": str(function.get("description") or ""),
                        "inputSchema": function.get("parameters")
                        if isinstance(function.get("parameters"), dict)
                        else {"type": "object", "properties": {}},
                    }
                )
                seen.add(name)
                continue

            name = schema.get("name")
            input_schema = schema.get("inputSchema") or schema.get("input_schema")
            if not isinstance(name, str) or not name or name in seen:
                continue
            if not isinstance(input_schema, dict):
                input_schema = {"type": "object", "properties": {}}
            dynamic_tools.append(
                {
                    "name": name,
                    "description": str(schema.get("description") or ""),
                    "inputSchema": input_schema,
                }
            )
            seen.add(name)

        for tool in self.config.additional_dynamic_tools:
            if not isinstance(tool, dict):
                continue
            name = tool.get("name")
            if not isinstance(name, str) or not name or name in seen:
                continue
            dynamic_tools.append(tool)
            seen.add(name)

        for model_cls, _ in self._custom_tools:
            schema = custom_tool_to_openai_schema(model_cls)
            function = schema.get("function", {})
            name = function.get("name", "")
            if not name or name in seen:
                continue
            dynamic_tools.append(
                {
                    "name": name,
                    "description": str(function.get("description") or ""),
                    "inputSchema": function.get("parameters")
                    or {"type": "object", "properties": {}},
                }
            )
            seen.add(name)

        return dynamic_tools

    def _build_turn_input(
        self,
        *,
        msg: PlatformMessage,
        participants_msg: str | None,
        contacts_msg: str | None,
        room_id: str,
    ) -> tuple[list[dict[str, str]], bool]:
        items: list[dict[str, str]] = []
        injected_system_prompt = False

        if room_id not in self._prompt_injected_rooms and self._system_prompt:
            items.append(
                {
                    "type": "text",
                    "text": "[System Instructions]\n" + self._system_prompt,
                }
            )
            injected_system_prompt = True

        if room_id in self._needs_history_injection:
            self._needs_history_injection.discard(room_id)
            raw_history = self._raw_history_by_room.pop(room_id, None)
            if raw_history:
                context = self._format_history_context(raw_history)
                if context:
                    items.append({"type": "text", "text": context})

        if participants_msg:
            items.append({"type": "text", "text": f"[System]: {participants_msg}"})

        if contacts_msg:
            items.append({"type": "text", "text": f"[System]: {contacts_msg}"})

        items.append({"type": "text", "text": msg.format_for_llm()})
        return items, injected_system_prompt

    def _format_history_context(self, raw: list[dict[str, Any]]) -> str | None:
        text_messages: list[str] = []
        for entry in raw:
            msg_type = entry.get("message_type", "")
            if msg_type not in {"text", "message"}:
                continue
            content = entry.get("content", "")
            if not isinstance(content, str) or not content.strip():
                continue
            sender = entry.get("sender_name") or entry.get("sender_type") or "Unknown"
            text_messages.append(f"[{sender}]: {content}")

        if not text_messages:
            return None

        truncated = text_messages[-self.config.max_history_messages :]
        header = (
            "[Conversation History]\n"
            "The following is the conversation history from a previous session. "
            "Use it to maintain continuity.\n"
        )
        return header + "\n".join(truncated)

    async def _handle_server_request(
        self,
        *,
        tools: AgentToolsProtocol,
        msg: PlatformMessage,
        room_id: str,
        event: RpcEvent,
    ) -> bool:
        assert self._client is not None
        if event.id is None:
            return False

        params = event.params if isinstance(event.params, dict) else {}

        if event.method == "item/tool/call":
            tool_name = str(params.get("tool") or "")
            arguments = params.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            call_id = str(params.get("callId") or "")
            tool_call_succeeded = False

            # Don't emit reporting for platform tools that already produce
            # visible output (messages/events) — reporting them is redundant.
            should_report = (
                self.config.enable_execution_reporting
                and tool_name not in _SILENT_REPORTING_TOOLS
            )

            if should_report:
                await tools.send_event(
                    content=json.dumps(
                        {"name": tool_name, "args": arguments, "tool_call_id": call_id}
                    ),
                    message_type="tool_call",
                )

            try:
                custom_tool = find_custom_tool(self._custom_tools, tool_name)
                if custom_tool:
                    result = await execute_custom_tool(custom_tool, arguments)
                else:
                    result = await tools.execute_tool_call(tool_name, arguments)
                text_result = (
                    result
                    if isinstance(result, str)
                    else json.dumps(result, default=str)
                )
                success = True
                await self._client.respond(
                    event.id,
                    {
                        "contentItems": [{"type": "inputText", "text": text_result}],
                        "success": success,
                    },
                )
                tool_call_succeeded = True
                if should_report:
                    await tools.send_event(
                        content=json.dumps(
                            {
                                "name": tool_name,
                                "output": text_result,
                                "tool_call_id": call_id,
                            }
                        ),
                        message_type="tool_result",
                    )
            except ValidationError as exc:
                errors = "; ".join(
                    f"{err['loc'][0]}: {err['msg']}" for err in exc.errors()
                )
                error_text = f"Invalid arguments for {tool_name}: {errors}"
                logger.error("Validation error for tool %s: %s", tool_name, exc)
                await self._client.respond(
                    event.id,
                    {
                        "contentItems": [{"type": "inputText", "text": error_text}],
                        "success": False,
                    },
                )
                if should_report:
                    await tools.send_event(
                        content=json.dumps(
                            {
                                "name": tool_name,
                                "output": error_text,
                                "tool_call_id": call_id,
                            }
                        ),
                        message_type="tool_result",
                    )
            except Exception as exc:
                error_text = f"Error: {exc}"
                logger.exception("Tool execution failed for %s", tool_name)
                await self._client.respond(
                    event.id,
                    {
                        "contentItems": [{"type": "inputText", "text": error_text}],
                        "success": False,
                    },
                )
                if should_report:
                    await tools.send_event(
                        content=json.dumps(
                            {
                                "name": tool_name,
                                "output": error_text,
                                "tool_call_id": call_id,
                            }
                        ),
                        message_type="tool_result",
                    )

            return tool_name == "thenvoi_send_message" and tool_call_succeeded

        if event.method in {
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
        }:
            summary = self._approval_summary(event.method, params)
            if self.config.approval_mode == "manual":
                decision = await self._resolve_manual_approval(
                    tools=tools,
                    msg=msg,
                    room_id=room_id,
                    event=event,
                    summary=summary,
                    params=params,
                )
            else:
                decision = (
                    "accept"
                    if self.config.approval_mode == "auto_accept"
                    else "decline"
                )
                if self.config.approval_text_notifications:
                    mention = [
                        {
                            "id": msg.sender_id,
                            "name": msg.sender_name or msg.sender_type,
                        }
                    ]
                    await tools.send_message(
                        f"Approval requested ({summary}). Policy decision: {decision}.",
                        mentions=mention,
                    )

            await self._client.respond(event.id, {"decision": decision})

            if self.config.emit_thought_events:
                await tools.send_event(
                    content=f"Codex approval request handled automatically ({decision}).",
                    message_type="thought",
                    metadata={"codex_approval_method": event.method},
                )
            return False

        await self._client.respond_error(
            event.id,
            code=-32601,
            message=f"Unhandled server request: {event.method}",
        )
        return False

    async def _emit_turn_outcome(
        self,
        *,
        tools: AgentToolsProtocol,
        msg: PlatformMessage,
        room_id: str,
        thread_id: str,
        turn_id: str | None,
        turn_status: str,
        turn_error: str,
        final_text: str,
        saw_send_message_tool: bool,
    ) -> None:
        if self.config.enable_task_events and self.config.emit_turn_task_markers:
            summary = f"Thread: {thread_id}"
            if turn_error:
                summary += f" | Error: {turn_error}"
            await tools.send_event(
                content=self._build_task_event_content(
                    task_id=turn_id,
                    task="Codex turn",
                    status=turn_status,
                    summary=summary,
                ),
                message_type="task",
                metadata={
                    "codex_room_id": room_id,
                    "codex_thread_id": thread_id,
                    "codex_turn_id": turn_id,
                    "codex_turn_status": turn_status,
                    "codex_error": turn_error or None,
                },
            )

        mention = [{"id": msg.sender_id, "name": msg.sender_name or msg.sender_type}]

        if turn_status == "completed":
            if (
                self.config.fallback_send_agent_text
                and final_text.strip()
                and not saw_send_message_tool
            ):
                await tools.send_message(final_text.strip(), mentions=mention)
            return

        if turn_status == "interrupted":
            await tools.send_message(
                "I stopped before completing this request.",
                mentions=mention,
            )
            return

        error_text = (
            f"I couldn't complete this request ({turn_status})."
            if not turn_error
            else f"I couldn't complete this request ({turn_status}): {turn_error}"
        )
        await tools.send_message(error_text, mentions=mention)

    async def _emit_item_completed_events(
        self,
        *,
        tools: AgentToolsProtocol,
        item: dict[str, Any],
        room_id: str,
        thread_id: str,
        turn_id: str | None,
    ) -> None:
        """Forward internal Codex operations as platform events."""
        item_type = item.get("type", "")
        item_id = str(item.get("id") or "")
        metadata = {
            "codex_room_id": room_id,
            "codex_thread_id": thread_id,
            "codex_turn_id": turn_id,
        }

        # Tool-like items gated on enable_execution_reporting
        if item_type in {
            "commandExecution",
            "fileChange",
            "mcpToolCall",
            "webSearch",
            "imageView",
            "collabAgentToolCall",
        }:
            if not self.config.enable_execution_reporting:
                return
            name, args, output = self._extract_tool_item(item_type, item)
            await tools.send_event(
                content=json.dumps(
                    {"name": name, "args": args, "tool_call_id": item_id}
                ),
                message_type="tool_call",
                metadata=metadata,
            )
            await tools.send_event(
                content=json.dumps(
                    {"name": name, "output": output, "tool_call_id": item_id}
                ),
                message_type="tool_result",
                metadata=metadata,
            )
            return

        # Thought-like items gated on emit_thought_events
        if item_type in {
            "reasoning",
            "plan",
            "contextCompaction",
            "enteredReviewMode",
            "exitedReviewMode",
        }:
            if not self.config.emit_thought_events:
                return
            text = self._extract_thought_text(item_type, item)
            await tools.send_event(
                content=text,
                message_type="thought",
                metadata=metadata,
            )
            return

        # Skip known non-actionable types
        if item_type in {"userMessage", "agentMessage"}:
            return

        logger.debug("Unhandled item/completed type: %s", item_type)

    @staticmethod
    def _extract_tool_item(
        item_type: str, item: dict[str, Any]
    ) -> tuple[str, dict[str, Any], str]:
        """Extract (name, args, output) for a tool-like item."""
        if item_type == "commandExecution":
            command = item.get("command", "")
            cwd = item.get("cwd", "")
            args: dict[str, Any] = {"command": command, "cwd": cwd}
            output_parts: list[str] = []
            if item.get("aggregated_output"):
                output_parts.append(str(item["aggregated_output"]))
            exit_code = item.get("exitCode")
            if exit_code is not None:
                output_parts.append(f"exit_code={exit_code}")
            status = item.get("status", "")
            output = "\n".join(output_parts) if output_parts else str(status)
            return "exec", args, output

        if item_type == "fileChange":
            changes = item.get("changes", [])
            if not isinstance(changes, list):
                changes = []
            file_paths = [c.get("path", "") for c in changes if isinstance(c, dict)]
            return (
                "file_edit",
                {"files": file_paths},
                str(item.get("status", "applied")),
            )

        if item_type == "mcpToolCall":
            server = item.get("server", "")
            tool = item.get("tool", "")
            name = f"mcp:{server}/{tool}"
            mcp_args = item.get("arguments", {})
            if not isinstance(mcp_args, dict):
                mcp_args = {}
            result = item.get("result")
            error = item.get("error")
            if result is not None:
                output = json.dumps(result, default=str)
            elif error is not None:
                output = json.dumps(error, default=str)
            else:
                output = "completed"
            return name, mcp_args, output

        if item_type == "webSearch":
            query = item.get("query", "")
            action = item.get("action")
            output = json.dumps(action, default=str) if action else "completed"
            return "web_search", {"query": query}, output

        if item_type == "imageView":
            path = item.get("path", "")
            return "view_image", {"path": path}, str(item.get("status", "viewed"))

        if item_type == "collabAgentToolCall":
            collab_tool = item.get("tool", "")
            name = f"collab:{collab_tool}"
            collab_args: dict[str, Any] = {}
            if item.get("prompt"):
                collab_args["prompt"] = item["prompt"]
            if item.get("agents"):
                collab_args["agents"] = item["agents"]
            result = item.get("result")
            output = (
                json.dumps(result, default=str) if result is not None else "completed"
            )
            return name, collab_args, output

        return item_type, {}, "completed"

    @staticmethod
    def _extract_thought_text(item_type: str, item: dict[str, Any]) -> str:
        """Extract display text for a thought-like item."""
        if item_type == "reasoning":
            summary = item.get("summary", [])
            if isinstance(summary, list):
                return "\n".join(str(s) for s in summary) or "(reasoning)"
            return str(summary) or "(reasoning)"

        if item_type == "plan":
            return str(item.get("text", "")) or "(plan)"

        if item_type == "contextCompaction":
            return "Context compaction performed"

        if item_type in {"enteredReviewMode", "exitedReviewMode"}:
            text = item.get("text", "")
            if text:
                return str(text)
            return f"Review mode: {item_type}"

        return str(item.get("text", "")) or item_type

    async def _resolve_manual_approval(
        self,
        *,
        tools: AgentToolsProtocol,
        msg: PlatformMessage,
        room_id: str,
        event: RpcEvent,
        summary: str,
        params: dict[str, Any],
    ) -> ApprovalDecision:
        mention = [{"id": msg.sender_id, "name": msg.sender_name or msg.sender_type}]
        assert event.id is not None, "approval request must have an id"
        token = self._approval_token(event.id, params)
        loop = asyncio.get_running_loop()
        pending = _PendingApproval(
            request_id=event.id,
            method=event.method,
            summary=summary,
            created_at=datetime.now(timezone.utc),
            future=loop.create_future(),
        )
        room_pending = self._pending_approvals.setdefault(room_id, {})
        if len(room_pending) >= self.config.max_pending_approvals_per_room:
            oldest_token = min(room_pending, key=lambda t: room_pending[t].created_at)
            evicted = room_pending.pop(oldest_token)
            if not evicted.future.done():
                evicted.future.set_result("decline")
            logger.warning(
                "Evicted oldest pending approval %s in room %s (limit %s)",
                oldest_token,
                room_id,
                self.config.max_pending_approvals_per_room,
            )
        room_pending[token] = pending
        await tools.send_message(
            "Approval requested "
            f"({summary}). Approval id: `{token}`. "
            f"Reply `/approve {token}` or `/decline {token}`. "
            "Use `/approvals` to list pending approvals.",
            mentions=mention,
        )

        try:
            decision_raw = await asyncio.wait_for(
                pending.future,
                timeout=self.config.approval_wait_timeout_s,
            )
            decision: ApprovalDecision = (
                "accept" if decision_raw == "accept" else "decline"
            )
            return decision
        except asyncio.TimeoutError:
            timeout_decision = self.config.approval_timeout_decision
            await tools.send_message(
                f"Approval `{token}` timed out. Applied `{timeout_decision}`.",
                mentions=mention,
            )
            return timeout_decision
        finally:
            self._clear_pending_approval(room_id, token)

    async def _forward_raw_task_event(
        self,
        *,
        tools: AgentToolsProtocol,
        room_id: str,
        thread_id: str,
        turn_id: str | None,
        method: str,
        params: dict[str, Any],
    ) -> None:
        if not self.config.enable_task_events:
            return

        is_started = method == "codex/event/task_started"
        task_phase = "started" if is_started else "completed"
        task_id = self._task_event_id(params)
        title = self._task_event_title(params)
        if task_id and title and is_started:
            self._task_titles_by_id[task_id] = title
        if task_id and not title:
            title = self._task_titles_by_id.get(task_id)
        summary = self._task_event_summary(params)
        if not title:
            title = "Codex task lifecycle event"
            if not summary:
                summary = f"Method: {method}"
        content = self._build_task_event_content(
            task_id=task_id,
            task=title,
            status=task_phase,
            summary=summary,
        )

        metadata: dict[str, Any] = {
            "codex_room_id": room_id,
            "codex_thread_id": thread_id,
            "codex_turn_id": turn_id,
            "codex_event_method": method,
            "codex_task_phase": task_phase,
        }
        if task_id:
            metadata["codex_task_id"] = task_id
        if params:
            metadata["codex_event_params"] = params

        await tools.send_event(
            content=content,
            message_type="task",
            metadata=metadata,
        )
        if not is_started and task_id:
            self._task_titles_by_id.pop(task_id, None)

    async def _handle_local_command(
        self,
        *,
        tools: AgentToolsProtocol,
        msg: PlatformMessage,
        history: CodexSessionState,
        room_id: str,
        command: str,
        args: str,
    ) -> bool:
        mention = [{"id": msg.sender_id, "name": msg.sender_name or msg.sender_type}]

        if command == "help":
            await tools.send_message(
                "Codex commands: "
                "`/status`, `/model`, `/models`, `/model list`, `/models list`, `/model <id>`, "
                "`/reasoning [none|minimal|low|medium|high|xhigh]`, "
                "`/approvals`, `/approve <id>`, `/decline <id>`, `/help`.",
                mentions=mention,
            )
            return True

        if command == "status":
            mapped_thread = self._room_threads.get(room_id) or history.thread_id or None
            status_text = (
                "Codex status:\n"
                f"- transport: {self.config.transport}\n"
                f"- selected_model: {self._selected_model or 'unknown'}\n"
                f"- configured_model: {self.config.model or 'auto'}\n"
                f"- room_id: {room_id}\n"
                f"- thread_id: {mapped_thread or 'not mapped'}\n"
                f"- approval_policy: {self.config.approval_policy}\n"
                f"- approval_mode: {self.config.approval_mode}\n"
                f"- sandbox: {self.config.sandbox or 'default'}\n"
                f"- reasoning_effort: {self.config.reasoning_effort or 'default'}\n"
                f"- reasoning_summary: {self.config.reasoning_summary or 'default'}\n"
                f"- pending_approvals: {len(self._pending_approvals.get(room_id, {}))}\n"
                f"- turn_task_markers: {self.config.emit_turn_task_markers}"
            )
            await tools.send_message(status_text, mentions=mention)
            return True

        if command in {"model", "models"}:
            model_arg = args.strip()
            if not model_arg:
                await tools.send_message(
                    "Current model: "
                    f"`{self._selected_model or 'unknown'}` "
                    f"(configured: `{self.config.model or 'auto'}`). "
                    "Use `/model list` to view available models or `/model <id>` to override.",
                    mentions=mention,
                )
                return True

            if model_arg.lower() in {"list", "ls"}:
                assert self._client is not None
                result = await self._client.request("model/list", {})
                models = self._visible_model_ids(result)
                if models:
                    preview = ", ".join(models[:10])
                    if len(models) > 10:
                        preview += ", ..."
                    await tools.send_message(
                        f"Available models ({len(models)}): {preview}",
                        mentions=mention,
                    )
                else:
                    await tools.send_message(
                        "No visible models returned by Codex app-server.",
                        mentions=mention,
                    )
                return True

            self.config.model = model_arg
            self._selected_model = model_arg
            self._model_explicitly_set = True
            await tools.send_message(
                f"Model override set to `{model_arg}` for subsequent turns.",
                mentions=mention,
            )
            return True

        if command == "reasoning":
            effort_arg = args.strip().lower()
            if not effort_arg:
                await tools.send_message(
                    f"Current reasoning effort: `{self.config.reasoning_effort or 'default'}`. "
                    f"Summary: `{self.config.reasoning_summary or 'default'}`. "
                    f"Use `/reasoning <{'|'.join(sorted(_REASONING_EFFORTS))}>` to override.",
                    mentions=mention,
                )
                return True
            if effort_arg not in _REASONING_EFFORTS:
                await tools.send_message(
                    f"Invalid reasoning effort `{effort_arg}`. "
                    f"Valid values: {', '.join(sorted(_REASONING_EFFORTS))}.",
                    mentions=mention,
                )
                return True
            self.config.reasoning_effort = effort_arg  # type: ignore[assignment]  # Literal narrowed by Pydantic validation
            await tools.send_message(
                f"Reasoning effort set to `{effort_arg}` for subsequent turns.",
                mentions=mention,
            )
            return True

        return False

    async def _handle_approval_command(
        self,
        *,
        tools: AgentToolsProtocol,
        msg: PlatformMessage,
        room_id: str,
        command: str,
        args: str,
    ) -> bool:
        mention = [{"id": msg.sender_id, "name": msg.sender_name or msg.sender_type}]
        pending = self._pending_approvals.get(room_id, {})

        if command == "approvals":
            if not pending:
                await tools.send_message("No pending approvals.", mentions=mention)
                return True
            lines = ["Pending approvals:"]
            now = datetime.now(timezone.utc)
            for token, item in pending.items():
                age_s = int((now - item.created_at).total_seconds())
                lines.append(f"- {token}: {item.summary} ({age_s}s)")
            await tools.send_message("\n".join(lines), mentions=mention)
            return True

        if command not in {"approve", "decline"}:
            return False

        if not pending:
            await tools.send_message(
                "No pending approvals to resolve.",
                mentions=mention,
            )
            return True

        token = args.strip().split(" ", 1)[0] if args.strip() else ""
        if token:
            selected = pending.get(token)
            if selected is None:
                available = ", ".join(sorted(pending.keys()))
                await tools.send_message(
                    f"Unknown approval id `{token}`. Pending: {available}",
                    mentions=mention,
                )
                return True
        elif len(pending) == 1:
            token, selected = next(iter(pending.items()))
        else:
            available = ", ".join(sorted(pending.keys()))
            await tools.send_message(
                "Multiple approvals pending. "
                f"Use `/{command} <id>`. Pending: {available}",
                mentions=mention,
            )
            return True

        assert selected is not None
        decision: ApprovalDecision = "accept" if command == "approve" else "decline"
        if not selected.future.done():
            selected.future.set_result(decision)
        await tools.send_message(
            f"Approval `{token}` resolved as `{decision}`.",
            mentions=mention,
        )
        return True

    def _build_system_prompt(self) -> None:
        if self.config.system_prompt:
            self._system_prompt = self.config.system_prompt
            return

        role_section = _ROLE_SECTION.get(self.config.role, "")
        combined_custom = "\n".join(
            section for section in (role_section, self.config.custom_section) if section
        )
        self._system_prompt = render_system_prompt(
            agent_name=self.agent_name or "Agent",
            agent_description=self.agent_description or "An AI assistant",
            custom_section=combined_custom,
            include_base_instructions=self.config.include_base_instructions,
        )

    def _apply_turn_overrides(self, params: dict[str, Any]) -> None:
        params["model"] = self._selected_model
        params["cwd"] = self.config.cwd
        params["approvalPolicy"] = self.config.approval_policy
        params["personality"] = self.config.personality
        if self.config.reasoning_effort is not None:
            params["effort"] = self.config.reasoning_effort
        if self.config.reasoning_summary is not None:
            params["summary"] = self.config.reasoning_summary
        self._apply_turn_sandbox(params)

    async def _start_turn_with_model_fallback(
        self, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Start a turn, falling back to an available model if the auto-selected one is unavailable.

        Only attempts fallback when the model was auto-selected (config.model was None).
        When the user explicitly configured a model, the error propagates — they may
        be using unlisted models that model/list doesn't report.
        """
        assert self._client is not None
        try:
            return await self._client.request("turn/start", params)
        except CodexJsonRpcError as exc:
            if self._model_explicitly_set:
                raise
            if not self._is_model_unavailable_error(exc):
                raise
            original_model = params.get("model", self._selected_model)
            logger.warning(
                "Model %s unavailable (code=%s): %s. Querying available models...",
                original_model,
                exc.code,
                exc.message,
            )
            fallback = await self._find_fallback_model(exclude=original_model)
            if fallback is None:
                raise
            logger.warning(
                "Falling back from %s to %s",
                original_model,
                fallback,
            )
            self._selected_model = fallback
            params["model"] = fallback
            return await self._client.request("turn/start", params)

    async def _find_fallback_model(self, exclude: Any = None) -> str | None:
        """Query model/list and return a fallback model, or None if unavailable."""
        assert self._client is not None
        fallbacks = self.config.fallback_models
        try:
            result = await self._client.request("model/list", {})
        except Exception:
            logger.warning("model/list failed during fallback lookup")
            for model_id in fallbacks:
                if model_id != exclude:
                    return model_id
            return None
        models = self._visible_model_ids(result)
        # Prefer configured fallback models if available
        for model_id in models:
            if model_id != exclude and model_id in fallbacks:
                return model_id
        for model_id in models:
            if model_id != exclude:
                return model_id
        # No visible models — try configured defaults
        for model_id in fallbacks:
            if model_id != exclude:
                return model_id
        return None

    @staticmethod
    def _is_model_unavailable_error(exc: CodexJsonRpcError) -> bool:
        """Check if the error indicates the requested model is not available."""
        msg = exc.message.lower()
        return any(
            phrase in msg
            for phrase in (
                "model not found",
                "model not available",
                "is not available",
                "model_not_found",
                "model unavailable",
                "does not have access",
                "no access to model",
            )
        )

    def _apply_thread_sandbox(self, params: dict[str, Any]) -> None:
        """Apply sandbox to thread/start params (only SandboxMode is accepted)."""
        if self.config.sandbox_policy is not None:
            # thread/start only has a `sandbox` field (SandboxMode enum).
            # Extract mode from sandbox_policy when possible.
            policy_type = self.config.sandbox_policy.get("type")
            if isinstance(policy_type, str):
                mode = self._normalize_sandbox_mode(policy_type)
                if mode is not None:
                    params["sandbox"] = mode
                    return
            logger.warning(
                "sandbox_policy type %s is not representable on thread/start; "
                "it will be applied at turn level instead",
                policy_type,
            )
            return

        if self.config.sandbox is None:
            return

        sandbox_mode = self._normalize_sandbox_mode(self.config.sandbox)
        if sandbox_mode is not None:
            params["sandbox"] = sandbox_mode
            return

        if self._canonical_sandbox_key(self.config.sandbox) == "external-sandbox":
            # externalSandbox is only representable via sandboxPolicy on
            # turn/start; thread/start does not accept it.
            logger.debug(
                "external-sandbox will be applied at turn level, not on thread/start"
            )
            return

        logger.warning(
            "Ignoring unsupported Codex sandbox value: %s", self.config.sandbox
        )

    # SandboxMode (thread/start) uses kebab-case: "read-only", "workspace-write", etc.
    # SandboxPolicy (turn/start) uses camelCase type tags: "readOnly", "workspaceWrite", etc.
    _SANDBOX_MODE_TO_POLICY_TYPE: dict[str, str] = {
        "read-only": "readOnly",
        "workspace-write": "workspaceWrite",
        "danger-full-access": "dangerFullAccess",
    }

    def _apply_turn_sandbox(self, params: dict[str, Any]) -> None:
        """Apply sandbox to turn/start params (full SandboxPolicy is accepted)."""
        if self.config.sandbox_policy is not None:
            params["sandboxPolicy"] = self._normalize_sandbox_policy(
                self.config.sandbox_policy
            )
            return

        if self.config.sandbox is None:
            return

        sandbox_mode = self._normalize_sandbox_mode(self.config.sandbox)
        if sandbox_mode is not None:
            policy_type = self._SANDBOX_MODE_TO_POLICY_TYPE.get(sandbox_mode)
            if policy_type:
                params["sandboxPolicy"] = {"type": policy_type}
            return

        if self._canonical_sandbox_key(self.config.sandbox) == "external-sandbox":
            params["sandboxPolicy"] = {"type": "externalSandbox"}
            return

        logger.warning(
            "Ignoring unsupported Codex sandbox value: %s", self.config.sandbox
        )

    @classmethod
    def _normalize_sandbox_mode(cls, sandbox: str) -> str | None:
        key = cls._canonical_sandbox_key(sandbox)
        if key in {"read-only", "workspace-write", "danger-full-access"}:
            return key
        return None

    @classmethod
    def _normalize_sandbox_policy(
        cls, sandbox_policy: dict[str, Any]
    ) -> dict[str, Any]:
        normalized = dict(sandbox_policy)
        policy_type = normalized.get("type")
        if not isinstance(policy_type, str):
            return normalized

        key = cls._canonical_sandbox_key(policy_type)
        camel = cls._SANDBOX_MODE_TO_POLICY_TYPE.get(key)
        if camel:
            normalized["type"] = camel
        elif key == "external-sandbox":
            # externalSandbox is represented only via sandboxPolicy.
            normalized["type"] = "externalSandbox"
        return normalized

    @staticmethod
    def _canonical_sandbox_key(value: str) -> str:
        compact = value.strip().lower().replace("_", "-").replace(" ", "")
        aliases = {
            "readonly": "read-only",
            "read-only": "read-only",
            "workspacewrite": "workspace-write",
            "workspace-write": "workspace-write",
            "dangerfullaccess": "danger-full-access",
            "danger-full-access": "danger-full-access",
            "externalsandbox": "external-sandbox",
            "external-sandbox": "external-sandbox",
        }
        return aliases.get(compact, compact)

    @staticmethod
    def _extract_turn_error(turn_payload: dict[str, Any]) -> str:
        error = turn_payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str):
                return message
        return ""

    @staticmethod
    def _approval_summary(method: str, params: dict[str, Any]) -> str:
        if method == "item/commandExecution/requestApproval":
            command = params.get("command")
            if isinstance(command, str) and command:
                return f"command: {command}"
            return "command execution"
        if method == "item/fileChange/requestApproval":
            reason = params.get("reason")
            if isinstance(reason, str) and reason:
                return f"file changes: {reason}"
            return "file changes"
        return method

    @staticmethod
    def _task_event_id(params: dict[str, Any]) -> str | None:
        for key in ("taskId", "task_id"):
            value = params.get(key)
            if isinstance(value, str) and value:
                return value

        task_value = params.get("task")
        if isinstance(task_value, dict):
            for key in ("taskId", "task_id", "id"):
                nested = task_value.get(key)
                if isinstance(nested, str) and nested:
                    return nested
        return None

    @staticmethod
    def _task_event_title(params: dict[str, Any]) -> str | None:
        for key in ("title", "name"):
            value = params.get(key)
            if isinstance(value, str) and value:
                return value

        task_value = params.get("task")
        if isinstance(task_value, str) and task_value:
            return task_value
        if isinstance(task_value, dict):
            for key in ("title", "name", "description"):
                nested = task_value.get(key)
                if isinstance(nested, str) and nested:
                    return nested
        return None

    @staticmethod
    def _task_event_summary(params: dict[str, Any]) -> str | None:
        for key in ("summary", "result", "message", "description"):
            value = params.get(key)
            if isinstance(value, str) and value:
                return value

        task_value = params.get("task")
        if isinstance(task_value, dict):
            for key in ("summary", "result", "message", "description"):
                nested = task_value.get(key)
                if isinstance(nested, str) and nested:
                    return nested
        return None

    @staticmethod
    def _build_task_event_content(
        *,
        task_id: str | None,
        task: str,
        status: str,
        summary: str | None = None,
    ) -> str:
        lines: list[str] = []
        if task_id:
            lines.append(f"UUID: {task_id}")
        lines.append(f"Task: {task}")
        lines.append(f"Status: {status}")
        if summary and summary != task:
            lines.append(f"Summary: {summary}")
        return "\n".join(lines)

    @staticmethod
    def _extract_local_command(content: str) -> tuple[str, str] | None:
        tokens = content.strip().split()
        for idx, token in enumerate(tokens):
            if not token.startswith("/") or len(token) == 1:
                continue
            command = token[1:].lower()
            if command not in {
                "help",
                "status",
                "model",
                "models",
                "reasoning",
                "approvals",
                "approve",
                "decline",
            }:
                continue
            args = " ".join(tokens[idx + 1 :]).strip()
            return command, args
        return None

    @staticmethod
    def _approval_token(request_id: int | str, params: dict[str, Any]) -> str:
        for key in ("approvalId", "approval_id", "itemId", "callId"):
            value = params.get(key)
            if isinstance(value, str) and value:
                return value
        return f"req-{request_id}"

    def _clear_pending_approval(self, room_id: str, token: str) -> None:
        room_pending = self._pending_approvals.get(room_id)
        if not room_pending:
            return
        room_pending.pop(token, None)
        if not room_pending:
            self._pending_approvals.pop(room_id, None)

    def _clear_pending_approvals_for_room(self, room_id: str) -> None:
        room_pending = self._pending_approvals.pop(room_id, {})
        for item in room_pending.values():
            if not item.future.done():
                item.future.set_result("decline")

    @staticmethod
    def _visible_model_ids(result: dict[str, Any]) -> list[str]:
        data = result.get("data") if isinstance(result, dict) else None
        if not isinstance(data, list):
            return []
        models: list[str] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            model_id = entry.get("id")
            if not isinstance(model_id, str) or not model_id:
                continue
            if bool(entry.get("hidden", False)):
                continue
            models.append(model_id)
        return models
