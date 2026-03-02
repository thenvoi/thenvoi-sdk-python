"""Command and approval workflow helpers for Codex adapter."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Protocol

from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.core.types import PlatformMessage
from thenvoi.integrations.codex import RpcEvent
from thenvoi.integrations.codex.types import CodexSessionState

from .approval import ApprovalDecision, PendingApproval, approval_token
from .events import (
    build_task_event_content,
    task_event_id,
    task_event_summary,
    task_event_title,
)

logger = logging.getLogger(__name__)


class CodexCommandClientProtocol(Protocol):
    """Subset of Codex client API used by command handlers."""

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        retry_on_overload: bool = True,
    ) -> dict[str, Any]: ...


class CodexCommandStateProtocol(Protocol):
    """Adapter state surface required by command helpers."""

    config: Any
    _client: CodexCommandClientProtocol | None
    _room_threads: dict[str, str]
    _selected_model: str | None
    _model_explicitly_set: bool
    _pending_approvals: dict[str, dict[str, PendingApproval]]
    _task_titles_by_id: dict[str, str]
    _max_task_titles: int

    @staticmethod
    def _visible_model_ids(result: dict[str, Any]) -> list[str]: ...

    def _record_nonfatal_error(
        self,
        category: str,
        error: Exception,
        **context: Any,
    ) -> None: ...


async def resolve_manual_approval(
    adapter: CodexCommandStateProtocol,
    *,
    tools: AgentToolsProtocol,
    msg: PlatformMessage,
    room_id: str,
    event: RpcEvent,
    summary: str,
    params: dict[str, Any],
) -> ApprovalDecision:
    """Run manual approval workflow and await user decision."""
    mention = [{"id": msg.sender_id, "name": msg.sender_name or msg.sender_type}]
    if event.id is None:
        raise RuntimeError("approval request must have an id")
    token = approval_token(event.id, params)
    loop = asyncio.get_running_loop()
    pending = PendingApproval(
        request_id=event.id,
        method=event.method,
        summary=summary,
        created_at=datetime.now(timezone.utc),
        future=loop.create_future(),
    )
    room_pending = adapter._pending_approvals.setdefault(room_id, {})
    if len(room_pending) >= adapter.config.max_pending_approvals_per_room:
        oldest_token = min(room_pending, key=lambda t: room_pending[t].created_at)
        evicted = room_pending.pop(oldest_token)
        if not evicted.future.done():
            evicted.future.set_result("decline")
        logger.warning(
            "Evicted oldest pending approval %s in room %s (limit %s)",
            oldest_token,
            room_id,
            adapter.config.max_pending_approvals_per_room,
        )
    room_pending[token] = pending
    try:
        await tools.send_message(
            "Approval requested "
            f"({summary}). Approval id: `{token}`. "
            f"Reply `/approve {token}` or `/decline {token}`. "
            "Use `/approvals` to list pending approvals.",
            mentions=mention,
        )
        decision_raw = await asyncio.wait_for(
            pending.future,
            timeout=adapter.config.approval_wait_timeout_s,
        )
        decision: ApprovalDecision = "accept" if decision_raw == "accept" else "decline"
        return decision
    except asyncio.TimeoutError:
        timeout_decision = adapter.config.approval_timeout_decision
        try:
            await tools.send_message(
                f"Approval `{token}` timed out. Applied `{timeout_decision}`.",
                mentions=mention,
            )
        except Exception as error:
            adapter._record_nonfatal_error(
                "approval_timeout_notification",
                error,
                room_id=room_id,
                token=token,
            )
        return timeout_decision
    finally:
        clear_pending_approval(adapter, room_id, token)


async def forward_raw_task_event(
    adapter: CodexCommandStateProtocol,
    *,
    tools: AgentToolsProtocol,
    room_id: str,
    thread_id: str,
    turn_id: str | None,
    method: str,
    params: dict[str, Any],
) -> None:
    """Forward raw Codex task lifecycle events onto platform task events."""
    if not adapter.config.enable_task_events:
        return

    is_started = method == "codex/event/task_started"
    task_phase = "started" if is_started else "completed"
    task_id = task_event_id(params)
    title = task_event_title(params)
    if task_id and title and is_started:
        adapter._task_titles_by_id[task_id] = title
        if len(adapter._task_titles_by_id) > adapter._max_task_titles:
            first_key = next(iter(adapter._task_titles_by_id))
            adapter._task_titles_by_id.pop(first_key, None)
    if task_id and not title:
        title = adapter._task_titles_by_id.get(task_id)
    summary = task_event_summary(params)
    if not title:
        title = "Codex task lifecycle event"
        if not summary:
            summary = f"Method: {method}"
    content = build_task_event_content(
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
        adapter._task_titles_by_id.pop(task_id, None)


async def handle_local_command(
    adapter: CodexCommandStateProtocol,
    *,
    tools: AgentToolsProtocol,
    msg: PlatformMessage,
    history: CodexSessionState,
    room_id: str,
    command: str,
    args: str,
    reasoning_efforts: set[str],
) -> bool:
    """Handle slash-commands that are local to adapter state."""
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
        mapped_thread = adapter._room_threads.get(room_id) or history.thread_id or None
        status_text = (
            "Codex status:\n"
            f"- transport: {adapter.config.transport}\n"
            f"- selected_model: {adapter._selected_model or 'unknown'}\n"
            f"- configured_model: {adapter.config.model or 'auto'}\n"
            f"- room_id: {room_id}\n"
            f"- thread_id: {mapped_thread or 'not mapped'}\n"
            f"- approval_policy: {adapter.config.approval_policy}\n"
            f"- approval_mode: {adapter.config.approval_mode}\n"
            f"- sandbox: {adapter.config.sandbox or 'default'}\n"
            f"- reasoning_effort: {adapter.config.reasoning_effort or 'default'}\n"
            f"- reasoning_summary: {adapter.config.reasoning_summary or 'default'}\n"
            f"- pending_approvals: {len(adapter._pending_approvals.get(room_id, {}))}\n"
            f"- turn_task_markers: {adapter.config.emit_turn_task_markers}"
        )
        await tools.send_message(status_text, mentions=mention)
        return True

    if command in {"model", "models"}:
        model_arg = args.strip()
        if not model_arg:
            await tools.send_message(
                "Current model: "
                f"`{adapter._selected_model or 'unknown'}` "
                f"(configured: `{adapter.config.model or 'auto'}`). "
                "Use `/model list` to view available models or `/model <id>` to override.",
                mentions=mention,
            )
            return True

        if model_arg.lower() in {"list", "ls"}:
            if adapter._client is None:
                raise RuntimeError("Codex client not initialized")
            result = await adapter._client.request("model/list", {})
            models = adapter._visible_model_ids(result)
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

        adapter.config.model = model_arg
        adapter._selected_model = model_arg
        adapter._model_explicitly_set = True
        await tools.send_message(
            f"Model override set to `{model_arg}` for subsequent turns.",
            mentions=mention,
        )
        return True

    if command == "reasoning":
        effort_arg = args.strip().lower()
        if not effort_arg:
            await tools.send_message(
                f"Current reasoning effort: `{adapter.config.reasoning_effort or 'default'}`. "
                f"Summary: `{adapter.config.reasoning_summary or 'default'}`. "
                f"Use `/reasoning <{'|'.join(sorted(reasoning_efforts))}>` to override.",
                mentions=mention,
            )
            return True
        if effort_arg not in reasoning_efforts:
            await tools.send_message(
                f"Invalid reasoning effort `{effort_arg}`. "
                f"Valid values: {', '.join(sorted(reasoning_efforts))}.",
                mentions=mention,
            )
            return True
        adapter.config.reasoning_effort = effort_arg  # type: ignore[assignment]
        await tools.send_message(
            f"Reasoning effort set to `{effort_arg}` for subsequent turns.",
            mentions=mention,
        )
        return True

    return False


async def handle_approval_command(
    adapter: CodexCommandStateProtocol,
    *,
    tools: AgentToolsProtocol,
    msg: PlatformMessage,
    room_id: str,
    command: str,
    args: str,
) -> bool:
    """Handle `/approvals`, `/approve`, and `/decline` commands."""
    mention = [{"id": msg.sender_id, "name": msg.sender_name or msg.sender_type}]
    pending = adapter._pending_approvals.get(room_id, {})

    if command == "approvals":
        if not pending:
            await tools.send_message("No pending approvals.", mentions=mention)
            return True
        lines = ["Pending approvals:"]
        now = datetime.now(timezone.utc)
        for token, item in list(pending.items()):
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

    if selected is None:
        raise RuntimeError("No matching pending approval after token lookup")
    decision: ApprovalDecision = "accept" if command == "approve" else "decline"
    if not selected.future.done():
        selected.future.set_result(decision)
    await tools.send_message(
        f"Approval `{token}` resolved as `{decision}`.",
        mentions=mention,
    )
    return True


def clear_pending_approval(
    adapter: CodexCommandStateProtocol, room_id: str, token: str
) -> None:
    """Remove one pending approval token and clean up empty room buckets."""
    room_pending = adapter._pending_approvals.get(room_id)
    if not room_pending:
        return
    room_pending.pop(token, None)
    if not room_pending:
        adapter._pending_approvals.pop(room_id, None)


def clear_pending_approvals_for_room(
    adapter: CodexCommandStateProtocol, room_id: str
) -> None:
    """Resolve all room approvals as declined and remove room bucket."""
    room_pending = adapter._pending_approvals.pop(room_id, {})
    for item in room_pending.values():
        if not item.future.done():
            item.future.set_result("decline")

