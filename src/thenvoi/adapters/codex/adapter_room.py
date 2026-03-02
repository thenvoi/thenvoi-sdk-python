"""Room and thread lifecycle helpers for Codex adapter."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Protocol

from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.core.types import PlatformMessage
from thenvoi.integrations.codex import CodexJsonRpcError
from thenvoi.integrations.codex.types import CodexSessionState
from thenvoi.runtime.tooling.custom_tools import CustomToolDef, custom_tool_to_openai_schema

from .events import build_task_event_content

logger = logging.getLogger(__name__)


class CodexRoomClientProtocol(Protocol):
    """Subset of Codex client API used by room/thread helpers."""

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        retry_on_overload: bool = True,
    ) -> dict[str, Any]: ...


class CodexRoomStateProtocol(Protocol):
    """Adapter state required for room/thread helper functions."""

    config: Any
    _client: CodexRoomClientProtocol | None
    _selected_model: str | None
    _custom_tools: list[CustomToolDef]
    _room_threads: dict[str, str]
    _prompt_injected_rooms: set[str]
    _raw_history_by_room: dict[str, list[dict[str, Any]]]
    _needs_history_injection: set[str]
    _system_prompt: str

    def _apply_thread_sandbox(self, params: dict[str, Any]) -> None: ...

    def build_metadata_updates(
        self,
        *,
        participants_msg: str | None,
        contacts_msg: str | None,
    ) -> list[str]: ...


async def ensure_thread(
    adapter: CodexRoomStateProtocol,
    *,
    room_id: str,
    history: CodexSessionState,
    tools: AgentToolsProtocol,
    is_session_bootstrap: bool,
) -> str:
    """Return mapped thread id for the room, creating/resuming when needed."""
    thread_id = adapter._room_threads.get(room_id)
    if thread_id:
        return thread_id

    if adapter._client is None:
        raise RuntimeError("Codex client not initialized")

    if is_session_bootstrap and history.has_thread():
        try:
            result = await adapter._client.request(
                "thread/resume",
                {
                    "threadId": history.thread_id,
                    "personality": adapter.config.personality,
                },
            )
            resumed = result.get("thread", {}) if isinstance(result, dict) else {}
            thread_id = str(resumed.get("id") or history.thread_id or "")
            if thread_id:
                adapter._room_threads[room_id] = thread_id
                adapter._raw_history_by_room.pop(room_id, None)
                if adapter.config.enable_task_events:
                    await tools.send_event(
                        content=build_task_event_content(
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
            if adapter.config.inject_history_on_resume_failure:
                adapter._needs_history_injection.add(room_id)
    else:
        # Not a bootstrap resume — clean up any stashed history.
        adapter._raw_history_by_room.pop(room_id, None)

    dynamic_tools = build_dynamic_tools(adapter, tools)
    start_params: dict[str, Any] = {
        "model": adapter._selected_model,
        "cwd": adapter.config.cwd,
        "approvalPolicy": adapter.config.approval_policy,
        "personality": adapter.config.personality,
        "dynamicTools": dynamic_tools,
    }
    adapter._apply_thread_sandbox(start_params)

    started = await adapter._client.request("thread/start", start_params)
    thread = started.get("thread") if isinstance(started, dict) else {}
    thread_id = str((thread or {}).get("id") or "")
    if not thread_id:
        raise RuntimeError("Codex thread/start returned no thread id")

    adapter._room_threads[room_id] = thread_id

    if adapter.config.enable_task_events:
        await tools.send_event(
            content=build_task_event_content(
                task_id=thread_id,
                task="Codex thread",
                status="mapped",
                summary=f"Transport: {adapter.config.transport}",
            ),
            message_type="task",
            metadata={
                "codex_thread_id": thread_id,
                "codex_room_id": room_id,
                "codex_created_at": datetime.now(timezone.utc).isoformat(),
                "codex_transport": adapter.config.transport,
            },
        )

    return thread_id


def build_dynamic_tools(
    adapter: CodexRoomStateProtocol, tools: AgentToolsProtocol
) -> list[dict[str, Any]]:
    """Return dynamic tool schema list for `thread/start`."""
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

    for tool in adapter.config.additional_dynamic_tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name or name in seen:
            continue
        dynamic_tools.append(tool)
        seen.add(name)

    for model_cls, _ in adapter._custom_tools:
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


def build_turn_input(
    adapter: CodexRoomStateProtocol,
    *,
    msg: PlatformMessage,
    participants_msg: str | None,
    contacts_msg: str | None,
    room_id: str,
) -> tuple[list[dict[str, str]], bool]:
    """Build Codex turn input payload and indicate prompt-injection intent."""
    items: list[dict[str, str]] = []
    injected_system_prompt = False

    if room_id not in adapter._prompt_injected_rooms and adapter._system_prompt:
        items.append(
            {
                "type": "text",
                "text": "[System Instructions]\n" + adapter._system_prompt,
            }
        )
        injected_system_prompt = True

    if room_id in adapter._needs_history_injection:
        adapter._needs_history_injection.discard(room_id)
        raw_history = adapter._raw_history_by_room.pop(room_id, None)
        if raw_history:
            context = format_history_context(adapter, raw_history)
            if context:
                items.append({"type": "text", "text": context})

    for update in adapter.build_metadata_updates(
        participants_msg=participants_msg,
        contacts_msg=contacts_msg,
    ):
        items.append({"type": "text", "text": update})

    items.append({"type": "text", "text": msg.format_for_llm()})
    return items, injected_system_prompt


def format_history_context(
    adapter: CodexRoomStateProtocol, raw: list[dict[str, Any]]
) -> str | None:
    """Render previously stashed history into a compact continuity block."""
    text_messages: list[str] = []
    for entry in raw:
        msg_type = entry.get("message_type", "")
        if msg_type not in {"text", "message"}:
            continue
        content = entry.get("content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        sender = (
            entry.get("sender_name")
            or entry.get("type")
            or entry.get("sender_type")
            or "Unknown"
        )
        text_messages.append(f"[{sender}]: {content}")

    if not text_messages:
        return None

    truncated = text_messages[-adapter.config.max_history_messages :]
    header = (
        "[Conversation History]\n"
        "The following is the conversation history from a previous session. "
        "Use it to maintain continuity.\n"
    )
    return header + "\n".join(truncated)
