"""Codex app-server adapter."""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

from pydantic import BaseModel, Field

from thenvoi.converters.codex import CodexHistoryConverter
from thenvoi.core.nonfatal import NonFatalErrorRecorder
from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.core.simple_adapter import SimpleAdapter, legacy_chat_turn_compat
from thenvoi.core.types import AgentInput, ChatMessageTurnContext, PlatformMessage
from thenvoi.integrations.codex import (
    CodexJsonRpcError,
    CodexStdioClient,
    CodexWebSocketClient,
    RpcEvent,
)
from thenvoi.integrations.codex.model_selection import (
    find_fallback_model as find_codex_fallback_model,
    is_model_unavailable_error as codex_is_model_unavailable_error,
    start_turn_with_model_fallback as codex_start_turn_with_model_fallback,
    visible_model_ids as codex_visible_model_ids,
)
from thenvoi.integrations.codex.sandbox import CodexSandboxConfigurator
from thenvoi.integrations.codex.types import CodexSessionState
from thenvoi.runtime.tooling.custom_tools import CustomToolDef
from thenvoi.runtime.prompts import render_system_prompt

from .adapter_commands import (
    clear_pending_approval,
    clear_pending_approvals_for_room,
    forward_raw_task_event,
    handle_approval_command,
    handle_local_command,
    resolve_manual_approval,
)
from .adapter_room import (
    build_dynamic_tools,
    build_turn_input,
    ensure_thread,
    format_history_context,
)
from .adapter_tooling import (
    emit_item_completed_events,
    emit_item_event,
    extract_thought_text,
    extract_tool_item,
    handle_approval_request,
    handle_server_request,
    handle_tool_call_request,
    report_tool_call_result,
)
from .adapter_turn_processing import (
    consume_turn_events,
    emit_turn_outcome,
    extract_turn_error,
    prepare_turn,
)
from .approval import ApprovalDecision, ApprovalMode, PendingApproval
from .turns import extract_local_command

logger = logging.getLogger(__name__)

TransportKind = Literal["stdio", "ws"]
_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
_REASONING_SUMMARIES = {"auto", "concise", "detailed", "none"}
_SANDBOX_CONFIG = CodexSandboxConfigurator(logger)

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
    # Fallback models tried when model/list fails or returns empty.
    # Update when OpenAI rotates model IDs.
    fallback_models: tuple[str, ...] = ("gpt-5.2", "gpt-5.3-codex")
    max_pending_approvals_per_room: int = 50


class CodexAdapter(NonFatalErrorRecorder, SimpleAdapter[CodexSessionState]):
    """
    Codex adapter backed by codex app-server (stdio or websocket transport).

    One Thenvoi room maps to one Codex thread. Mapping is persisted in task
    events metadata and restored via CodexHistoryConverter on bootstrap.
    """

    _nonfatal_log_level = logging.ERROR

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
        self._task_titles_by_id: OrderedDict[str, str] = OrderedDict()
        self._max_task_titles: int = 500
        self._pending_approvals: dict[str, dict[str, PendingApproval]] = {}
        self._raw_history_by_room: dict[str, list[dict[str, Any]]] = {}
        self._needs_history_injection: set[str] = set()
        self._init_nonfatal_errors()
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
            if not adapter._rpc_lock.locked():
                raise RuntimeError("_handle_set_model must run under _rpc_lock")
            adapter.config.model = inp.model
            adapter._selected_model = inp.model
            adapter._model_explicitly_set = True
            return f"Model changed to {inp.model} for subsequent turns."

        def _handle_set_reasoning(inp: SetReasoningInput) -> str:
            if not adapter._rpc_lock.locked():
                raise RuntimeError("_handle_set_reasoning must run under _rpc_lock")
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
            "sandbox=%s, approval_mode=%s, "
            "execution_reporting=%s, self_config_tools=%s, "
            "task_events=%s, turn_markers=%s, thought_events=%s",
            agent_name,
            self.config.transport,
            self._selected_model or self.config.model or "auto",
            self.config.sandbox or "default",
            self.config.approval_mode,
            self.config.enable_execution_reporting,
            self.config.enable_self_config_tools,
            self.config.enable_task_events,
            self.config.emit_turn_task_markers,
            self.config.emit_thought_events,
        )

    async def on_event(self, inp: AgentInput) -> None:
        if self.config.inject_history_on_resume_failure:
            self.stage_bootstrap_payload(
                self._raw_history_by_room,
                room_id=inp.room_id,
                is_session_bootstrap=inp.is_session_bootstrap,
                payload=inp.history.raw or None,
            )
        await super().on_event(inp)

    @legacy_chat_turn_compat
    async def on_message(
        self,
        turn: ChatMessageTurnContext[CodexSessionState, AgentToolsProtocol],
    ) -> None:
        msg = turn.msg
        tools = turn.tools
        history = turn.history
        participants_msg = turn.participants_msg
        contacts_msg = turn.contacts_msg
        is_session_bootstrap = turn.is_session_bootstrap
        room_id = turn.room_id

        command = extract_local_command(msg.content)
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
            if self._client is None:
                raise RuntimeError(
                    "Codex client not initialized after _ensure_client_ready"
                )

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

            thread_id, turn_id = await self._prepare_turn(
                room_id=room_id,
                history=history,
                tools=tools,
                msg=msg,
                participants_msg=participants_msg,
                contacts_msg=contacts_msg,
                is_session_bootstrap=is_session_bootstrap,
            )

            (
                saw_send_message_tool,
                final_text,
                turn_status,
                turn_error,
            ) = await self._consume_turn_events(
                tools=tools,
                msg=msg,
                room_id=room_id,
                thread_id=thread_id,
                turn_id=turn_id,
            )

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

    async def _prepare_turn(
        self,
        *,
        room_id: str,
        history: CodexSessionState,
        tools: AgentToolsProtocol,
        msg: PlatformMessage,
        participants_msg: str | None,
        contacts_msg: str | None,
        is_session_bootstrap: bool,
    ) -> tuple[str, str]:
        return await prepare_turn(
            self,
            room_id=room_id,
            history=history,
            tools=tools,
            msg=msg,
            participants_msg=participants_msg,
            contacts_msg=contacts_msg,
            is_session_bootstrap=is_session_bootstrap,
        )

    async def _consume_turn_events(
        self,
        *,
        tools: AgentToolsProtocol,
        msg: PlatformMessage,
        room_id: str,
        thread_id: str,
        turn_id: str,
    ) -> tuple[bool, str, str, str]:
        return await consume_turn_events(
            self,
            tools=tools,
            msg=msg,
            room_id=room_id,
            thread_id=thread_id,
            turn_id=turn_id,
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

        if self._client is None:
            raise RuntimeError("Codex client not initialized")
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
        return await ensure_thread(
            self,
            room_id=room_id,
            history=history,
            tools=tools,
            is_session_bootstrap=is_session_bootstrap,
        )

    def _build_dynamic_tools(self, tools: AgentToolsProtocol) -> list[dict[str, Any]]:
        return build_dynamic_tools(self, tools)

    def _build_turn_input(
        self,
        *,
        msg: PlatformMessage,
        participants_msg: str | None,
        contacts_msg: str | None,
        room_id: str,
    ) -> tuple[list[dict[str, str]], bool]:
        return build_turn_input(
            self,
            msg=msg,
            participants_msg=participants_msg,
            contacts_msg=contacts_msg,
            room_id=room_id,
        )

    def _format_history_context(self, raw: list[dict[str, Any]]) -> str | None:
        return format_history_context(self, raw)

    async def _report_tool_call_result(
        self,
        *,
        tools: AgentToolsProtocol,
        tool_name: str,
        call_id: str,
        output: str,
        should_report: bool,
    ) -> None:
        await report_tool_call_result(
            tools=tools,
            tool_name=tool_name,
            call_id=call_id,
            output=output,
            should_report=should_report,
        )

    async def _handle_tool_call_request(
        self,
        *,
        tools: AgentToolsProtocol,
        event: RpcEvent,
        params: dict[str, Any],
    ) -> bool:
        return await handle_tool_call_request(
            self,
            tools=tools,
            event=event,
            params=params,
            silent_reporting_tools=_SILENT_REPORTING_TOOLS,
        )

    async def _handle_approval_request(
        self,
        *,
        tools: AgentToolsProtocol,
        msg: PlatformMessage,
        room_id: str,
        event: RpcEvent,
        params: dict[str, Any],
    ) -> None:
        await handle_approval_request(
            self,
            tools=tools,
            msg=msg,
            room_id=room_id,
            event=event,
            params=params,
        )

    async def _handle_server_request(
        self,
        *,
        tools: AgentToolsProtocol,
        msg: PlatformMessage,
        room_id: str,
        event: RpcEvent,
    ) -> bool:
        return await handle_server_request(
            self,
            tools=tools,
            msg=msg,
            room_id=room_id,
            event=event,
            silent_reporting_tools=_SILENT_REPORTING_TOOLS,
        )

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
        await emit_turn_outcome(
            self,
            tools=tools,
            msg=msg,
            room_id=room_id,
            thread_id=thread_id,
            turn_id=turn_id,
            turn_status=turn_status,
            turn_error=turn_error,
            final_text=final_text,
            saw_send_message_tool=saw_send_message_tool,
        )

    async def _emit_item_completed_events(
        self,
        *,
        tools: AgentToolsProtocol,
        item: dict[str, Any],
        room_id: str,
        thread_id: str,
        turn_id: str | None,
    ) -> None:
        await emit_item_completed_events(
            self,
            tools=tools,
            item=item,
            room_id=room_id,
            thread_id=thread_id,
            turn_id=turn_id,
        )

    async def _emit_item_event(
        self,
        *,
        tools: AgentToolsProtocol,
        item_type: str,
        item_id: str,
        item: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        await emit_item_event(
            self,
            tools=tools,
            item_type=item_type,
            item_id=item_id,
            item=item,
            metadata=metadata,
        )

    @staticmethod
    def _extract_tool_item(
        item_type: str, item: dict[str, Any]
    ) -> tuple[str, dict[str, Any], str]:
        return extract_tool_item(item_type, item)

    @staticmethod
    def _extract_thought_text(item_type: str, item: dict[str, Any]) -> str:
        return extract_thought_text(item_type, item)

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
        return await resolve_manual_approval(
            self,
            tools=tools,
            msg=msg,
            room_id=room_id,
            event=event,
            summary=summary,
            params=params,
        )

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
        await forward_raw_task_event(
            self,
            tools=tools,
            room_id=room_id,
            thread_id=thread_id,
            turn_id=turn_id,
            method=method,
            params=params,
        )

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
        return await handle_local_command(
            self,
            tools=tools,
            msg=msg,
            history=history,
            room_id=room_id,
            command=command,
            args=args,
            reasoning_efforts=_REASONING_EFFORTS,
        )

    async def _handle_approval_command(
        self,
        *,
        tools: AgentToolsProtocol,
        msg: PlatformMessage,
        room_id: str,
        command: str,
        args: str,
    ) -> bool:
        return await handle_approval_command(
            self,
            tools=tools,
            msg=msg,
            room_id=room_id,
            command=command,
            args=args,
        )

    def _build_system_prompt(self) -> None:
        if self.config.system_prompt:
            self._system_prompt = self.config.system_prompt
            return

        self._system_prompt = render_system_prompt(
            agent_name=self.agent_name or "Agent",
            agent_description=self.agent_description or "An AI assistant",
            custom_section=self.config.custom_section,
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
        """Start a turn, retrying with fallback model policy when appropriate."""
        if self._client is None:
            raise RuntimeError("CodexAdapter client is None — was on_started() called?")
        return await codex_start_turn_with_model_fallback(
            request=self._client.request,
            params=params,
            model_explicitly_set=self._model_explicitly_set,
            fallback_models=self.config.fallback_models,
            update_selected_model=lambda model: setattr(self, "_selected_model", model),
            logger=logger,
        )

    async def _find_fallback_model(self, exclude: Any = None) -> str | None:
        """Query model/list and return a fallback model, or None if unavailable."""
        if self._client is None:
            raise RuntimeError("CodexAdapter client is None — was on_started() called?")
        return await find_codex_fallback_model(
            request=self._client.request,
            exclude=exclude,
            fallback_models=self.config.fallback_models,
            logger=logger,
        )

    @staticmethod
    def _is_model_unavailable_error(exc: CodexJsonRpcError) -> bool:
        """Check if the error indicates the requested model is not available."""
        return codex_is_model_unavailable_error(exc)

    def _apply_thread_sandbox(self, params: dict[str, Any]) -> None:
        """Apply sandbox to thread/start params (only SandboxMode is accepted)."""
        _SANDBOX_CONFIG.apply_thread_sandbox(
            params=params,
            sandbox=self.config.sandbox,
            sandbox_policy=self.config.sandbox_policy,
        )

    def _apply_turn_sandbox(self, params: dict[str, Any]) -> None:
        """Apply sandbox to turn/start params (full SandboxPolicy is accepted)."""
        _SANDBOX_CONFIG.apply_turn_sandbox(
            params=params,
            sandbox=self.config.sandbox,
            sandbox_policy=self.config.sandbox_policy,
        )

    @classmethod
    def _normalize_sandbox_mode(cls, sandbox: str) -> str | None:
        del cls
        return _SANDBOX_CONFIG.normalize_sandbox_mode(sandbox)

    @classmethod
    def _normalize_sandbox_policy(
        cls, sandbox_policy: dict[str, Any]
    ) -> dict[str, Any]:
        del cls
        return _SANDBOX_CONFIG.normalize_sandbox_policy(sandbox_policy)

    @staticmethod
    def _canonical_sandbox_key(value: str) -> str:
        return _SANDBOX_CONFIG.canonical_sandbox_key(value)

    @staticmethod
    def _extract_turn_error(turn_payload: dict[str, Any]) -> str:
        return extract_turn_error(turn_payload)

    def _clear_pending_approval(self, room_id: str, token: str) -> None:
        clear_pending_approval(self, room_id, token)

    def _clear_pending_approvals_for_room(self, room_id: str) -> None:
        clear_pending_approvals_for_room(self, room_id)

    @staticmethod
    def _visible_model_ids(result: dict[str, Any]) -> list[str]:
        return codex_visible_model_ids(result)
