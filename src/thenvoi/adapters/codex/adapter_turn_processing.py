"""Turn lifecycle helpers for Codex adapter."""

from __future__ import annotations

import asyncio
import logging
import time as _time
from typing import Any, Protocol

from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.core.types import PlatformMessage
from thenvoi.integrations.codex import RpcEvent
from thenvoi.integrations.codex.types import CodexSessionState

from .events import build_task_event_content

logger = logging.getLogger(__name__)


class CodexTurnClientProtocol(Protocol):
    """Subset of Codex client API used by turn-processing helpers."""

    async def recv_event(self, timeout_s: float | None = None) -> RpcEvent: ...

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        retry_on_overload: bool = True,
    ) -> dict[str, Any]: ...


class CodexTurnStateProtocol(Protocol):
    """Adapter state surface required by turn-processing helpers."""

    config: Any
    _client: CodexTurnClientProtocol | None
    _prompt_injected_rooms: set[str]

    async def _ensure_thread(
        self,
        *,
        room_id: str,
        history: CodexSessionState,
        tools: AgentToolsProtocol,
        is_session_bootstrap: bool,
    ) -> str: ...

    def _build_turn_input(
        self,
        *,
        msg: PlatformMessage,
        participants_msg: str | None,
        contacts_msg: str | None,
        room_id: str,
    ) -> tuple[list[dict[str, str]], bool]: ...

    def _apply_turn_overrides(self, params: dict[str, Any]) -> None: ...

    async def _start_turn_with_model_fallback(
        self, params: dict[str, Any]
    ) -> dict[str, Any]: ...

    async def _handle_server_request(
        self,
        *,
        tools: AgentToolsProtocol,
        msg: PlatformMessage,
        room_id: str,
        event: RpcEvent,
    ) -> bool: ...

    async def _forward_raw_task_event(
        self,
        *,
        tools: AgentToolsProtocol,
        room_id: str,
        thread_id: str,
        turn_id: str | None,
        method: str,
        params: dict[str, Any],
    ) -> None: ...

    async def _emit_item_completed_events(
        self,
        *,
        tools: AgentToolsProtocol,
        item: dict[str, Any],
        room_id: str,
        thread_id: str,
        turn_id: str | None,
    ) -> None: ...

    @staticmethod
    def _extract_turn_error(turn_payload: dict[str, Any]) -> str: ...


async def prepare_turn(
    adapter: CodexTurnStateProtocol,
    *,
    room_id: str,
    history: CodexSessionState,
    tools: AgentToolsProtocol,
    msg: PlatformMessage,
    participants_msg: str | None,
    contacts_msg: str | None,
    is_session_bootstrap: bool,
) -> tuple[str, str]:
    """Prepare and start a Codex turn, returning `(thread_id, turn_id)`."""
    thread_id = await adapter._ensure_thread(
        room_id=room_id,
        history=history,
        tools=tools,
        is_session_bootstrap=is_session_bootstrap,
    )

    turn_input, has_pending_prompt_injection = adapter._build_turn_input(
        msg=msg,
        participants_msg=participants_msg,
        contacts_msg=contacts_msg,
        room_id=room_id,
    )

    turn_params: dict[str, Any] = {
        "threadId": thread_id,
        "input": turn_input,
    }
    adapter._apply_turn_overrides(turn_params)

    turn_started = await adapter._start_turn_with_model_fallback(turn_params)
    if has_pending_prompt_injection:
        adapter._prompt_injected_rooms.add(room_id)
    turn = turn_started.get("turn") if isinstance(turn_started, dict) else {}
    turn_id = str((turn or {}).get("id") or "")

    if adapter.config.enable_task_events and adapter.config.emit_turn_task_markers:
        await tools.send_event(
            content=build_task_event_content(
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

    return thread_id, turn_id


async def consume_turn_events(
    adapter: CodexTurnStateProtocol,
    *,
    tools: AgentToolsProtocol,
    msg: PlatformMessage,
    room_id: str,
    thread_id: str,
    turn_id: str,
) -> tuple[bool, str, str, str]:
    """Consume stream events for a turn and return outcome state tuple."""
    if adapter._client is None:
        raise RuntimeError("Codex client not initialized")

    saw_send_message_tool = False
    final_text = ""
    turn_status = "failed"
    turn_error = ""
    turn_start = _time.monotonic()

    try:
        while True:
            remaining = max(
                0.0,
                adapter.config.turn_timeout_s - (_time.monotonic() - turn_start),
            )
            event = await adapter._client.recv_event(timeout_s=remaining)
            if event.kind == "request":
                used_send_message = await adapter._handle_server_request(
                    tools=tools,
                    msg=msg,
                    room_id=room_id,
                    event=event,
                )
                saw_send_message_tool = saw_send_message_tool or used_send_message
                continue

            params = event.params if isinstance(event.params, dict) else {}

            if event.method in {
                "codex/event/task_started",
                "codex/event/task_complete",
            }:
                await adapter._forward_raw_task_event(
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
                error_msg = (
                    error_obj.get("message", "")
                    if isinstance(error_obj, dict)
                    else str(error_obj)
                )
                will_retry = bool(params.get("willRetry", False))
                if will_retry:
                    logger.warning("Codex transient error (will retry): %s", error_msg)
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
                        await adapter._emit_item_completed_events(
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
                    params.get("turn") if isinstance(params.get("turn"), dict) else {}
                )
                event_turn_id = str(turn_payload.get("id") or "")
                if turn_id and event_turn_id and event_turn_id != turn_id:
                    continue
                turn_status = str(turn_payload.get("status") or "failed")
                turn_error = adapter._extract_turn_error(turn_payload)
                break
    except asyncio.TimeoutError:
        logger.error(
            "Codex turn timed out after %ss (thread=%s, turn=%s)",
            adapter.config.turn_timeout_s,
            thread_id,
            turn_id,
        )
        if turn_id:
            try:
                await adapter._client.request("turn/interrupt", {"turnId": turn_id})
            except Exception:
                logger.warning(
                    "Failed to send turn/interrupt after timeout",
                    exc_info=True,
                )
        turn_status = "interrupted"
        turn_error = "Turn timed out"

    return saw_send_message_tool, final_text, turn_status, turn_error


async def emit_turn_outcome(
    adapter: CodexTurnStateProtocol,
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
    """Send user-visible completion/failure messages for a processed turn."""
    if adapter.config.enable_task_events and adapter.config.emit_turn_task_markers:
        summary = f"Thread: {thread_id}"
        if turn_error:
            summary += f" | Error: {turn_error}"
        await tools.send_event(
            content=build_task_event_content(
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
            adapter.config.fallback_send_agent_text
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


def extract_turn_error(turn_payload: dict[str, Any]) -> str:
    """Extract human-readable error text from completed-turn payload."""
    error = turn_payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str):
            return message
    return ""

