"""Tool execution and reporting helpers for Codex adapter."""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from pydantic import ValidationError

from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.core.types import PlatformMessage
from thenvoi.integrations.codex import RpcEvent
from thenvoi.runtime.tool_bridge import ToolExecutionError, invoke_platform_tool
from thenvoi.runtime.tooling.custom_tools import (
    CustomToolDef,
    execute_custom_tool,
    find_custom_tool,
)

from .approval import ApprovalDecision, approval_summary

logger = logging.getLogger(__name__)


class CodexToolingClientProtocol(Protocol):
    """Subset of Codex client API used by tooling helpers."""

    async def respond(self, request_id: int | str, result: dict[str, Any]) -> None: ...

    async def respond_error(
        self,
        request_id: int | str,
        *,
        code: int,
        message: str,
        data: Any | None = None,
    ) -> None: ...


class CodexToolingStateProtocol(Protocol):
    """Adapter state surface required by tooling helpers."""

    config: Any
    _client: CodexToolingClientProtocol | None
    _custom_tools: list[CustomToolDef]

    async def _resolve_manual_approval(
        self,
        *,
        tools: AgentToolsProtocol,
        msg: PlatformMessage,
        room_id: str,
        event: RpcEvent,
        summary: str,
        params: dict[str, Any],
    ) -> ApprovalDecision: ...

    def _record_nonfatal_error(
        self,
        category: str,
        error: Exception,
        **context: Any,
    ) -> None: ...


async def report_tool_call_result(
    *,
    tools: AgentToolsProtocol,
    tool_name: str,
    call_id: str,
    output: str,
    should_report: bool,
) -> None:
    """Emit tool-result event when execution reporting is enabled."""
    if not should_report:
        return
    await tools.send_event(
        content=json.dumps(
            {
                "name": tool_name,
                "output": output,
                "tool_call_id": call_id,
            }
        ),
        message_type="tool_result",
    )


async def handle_tool_call_request(
    adapter: CodexToolingStateProtocol,
    *,
    tools: AgentToolsProtocol,
    event: RpcEvent,
    params: dict[str, Any],
    silent_reporting_tools: frozenset[str],
) -> bool:
    """Execute a server-initiated tool call and respond to Codex."""
    if adapter._client is None or event.id is None:
        return False

    tool_name = str(params.get("tool") or "")
    arguments = params.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}
    call_id = str(params.get("callId") or "")
    tool_call_succeeded = False

    should_report = (
        adapter.config.enable_execution_reporting
        and tool_name not in silent_reporting_tools
    )
    if should_report:
        await tools.send_event(
            content=json.dumps(
                {"name": tool_name, "args": arguments, "tool_call_id": call_id}
            ),
            message_type="tool_call",
        )

    try:
        custom_tool = find_custom_tool(adapter._custom_tools, tool_name)
        if custom_tool:
            result = await execute_custom_tool(custom_tool, arguments)
        else:
            result = await invoke_platform_tool(tools, tool_name, arguments)
        text_result = result if isinstance(result, str) else json.dumps(result, default=str)
        await adapter._client.respond(
            event.id,
            {
                "contentItems": [{"type": "inputText", "text": text_result}],
                "success": True,
            },
        )
        tool_call_succeeded = True
        await report_tool_call_result(
            tools=tools,
            tool_name=tool_name,
            call_id=call_id,
            output=text_result,
            should_report=should_report,
        )
    except ToolExecutionError as exc:
        error_text = exc.failure.message
        logger.error("Tool execution failed for %s: %s", tool_name, error_text)
        await adapter._client.respond(
            event.id,
            {
                "contentItems": [{"type": "inputText", "text": error_text}],
                "success": False,
            },
        )
        await report_tool_call_result(
            tools=tools,
            tool_name=tool_name,
            call_id=call_id,
            output=error_text,
            should_report=should_report,
        )
    except ValidationError as exc:
        errors = "; ".join(f"{err['loc'][0]}: {err['msg']}" for err in exc.errors())
        error_text = f"Invalid arguments for {tool_name}: {errors}"
        logger.error("Validation error for tool %s: %s", tool_name, exc)
        await adapter._client.respond(
            event.id,
            {
                "contentItems": [{"type": "inputText", "text": error_text}],
                "success": False,
            },
        )
        await report_tool_call_result(
            tools=tools,
            tool_name=tool_name,
            call_id=call_id,
            output=error_text,
            should_report=should_report,
        )
    except Exception as exc:
        error_text = f"Error: {exc}"
        logger.exception("Tool execution failed for %s", tool_name)
        await adapter._client.respond(
            event.id,
            {
                "contentItems": [{"type": "inputText", "text": error_text}],
                "success": False,
            },
        )
        await report_tool_call_result(
            tools=tools,
            tool_name=tool_name,
            call_id=call_id,
            output=error_text,
            should_report=should_report,
        )

    return tool_name == "thenvoi_send_message" and tool_call_succeeded


async def handle_approval_request(
    adapter: CodexToolingStateProtocol,
    *,
    tools: AgentToolsProtocol,
    msg: PlatformMessage,
    room_id: str,
    event: RpcEvent,
    params: dict[str, Any],
) -> None:
    """Resolve file/command approval request from Codex."""
    if adapter._client is None or event.id is None:
        return

    summary = approval_summary(event.method, params)
    if adapter.config.approval_mode == "manual":
        try:
            decision = await adapter._resolve_manual_approval(
                tools=tools,
                msg=msg,
                room_id=room_id,
                event=event,
                summary=summary,
                params=params,
            )
        except Exception:
            logger.exception("Manual approval flow failed; defaulting to decline")
            decision = "decline"
    else:
        decision = "accept" if adapter.config.approval_mode == "auto_accept" else "decline"
    await adapter._client.respond(event.id, {"decision": decision})

    if (
        adapter.config.approval_mode != "manual"
        and adapter.config.approval_text_notifications
    ):
        mention = [{"id": msg.sender_id, "name": msg.sender_name or msg.sender_type}]
        try:
            await tools.send_message(
                f"Approval requested ({summary}). Policy decision: {decision}.",
                mentions=mention,
            )
        except Exception as error:
            adapter._record_nonfatal_error(
                "approval_policy_notification",
                error,
                room_id=room_id,
                decision=decision,
            )

    if adapter.config.emit_thought_events:
        try:
            await tools.send_event(
                content=f"Codex approval request handled automatically ({decision}).",
                message_type="thought",
                metadata={"codex_approval_method": event.method},
            )
        except Exception as error:
            adapter._record_nonfatal_error(
                "approval_thought_event",
                error,
                room_id=room_id,
                method=event.method,
            )


async def handle_server_request(
    adapter: CodexToolingStateProtocol,
    *,
    tools: AgentToolsProtocol,
    msg: PlatformMessage,
    room_id: str,
    event: RpcEvent,
    silent_reporting_tools: frozenset[str],
) -> bool:
    """Dispatch server `request` events (tool calls, approvals, unknown)."""
    if adapter._client is None:
        raise RuntimeError("CodexAdapter client is None — was on_started() called?")
    if event.id is None:
        return False

    params = event.params if isinstance(event.params, dict) else {}

    if event.method == "item/tool/call":
        return await handle_tool_call_request(
            adapter,
            tools=tools,
            event=event,
            params=params,
            silent_reporting_tools=silent_reporting_tools,
        )

    if event.method in {
        "item/commandExecution/requestApproval",
        "item/fileChange/requestApproval",
    }:
        await handle_approval_request(
            adapter,
            tools=tools,
            msg=msg,
            room_id=room_id,
            event=event,
            params=params,
        )
        return False

    await adapter._client.respond_error(
        event.id,
        code=-32601,
        message=f"Unhandled server request: {event.method}",
    )
    return False


async def emit_item_completed_events(
    adapter: CodexToolingStateProtocol,
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

    try:
        await emit_item_event(
            adapter,
            tools=tools,
            item_type=item_type,
            item_id=item_id,
            item=item,
            metadata=metadata,
        )
    except Exception:
        logger.exception(
            "Failed to emit %s event for item %s (best-effort)",
            item_type,
            item_id,
        )


async def emit_item_event(
    adapter: CodexToolingStateProtocol,
    *,
    tools: AgentToolsProtocol,
    item_type: str,
    item_id: str,
    item: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Inner dispatch for item-completed event shapes."""
    if item_type in {
        "commandExecution",
        "fileChange",
        "mcpToolCall",
        "webSearch",
        "imageView",
        "collabAgentToolCall",
    }:
        if not adapter.config.enable_execution_reporting:
            return
        name, args, output = extract_tool_item(item_type, item)
        await tools.send_event(
            content=json.dumps({"name": name, "args": args, "tool_call_id": item_id}),
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

    if item_type in {
        "reasoning",
        "plan",
        "contextCompaction",
        "enteredReviewMode",
        "exitedReviewMode",
    }:
        if not adapter.config.emit_thought_events:
            return
        text = extract_thought_text(item_type, item)
        await tools.send_event(
            content=text,
            message_type="thought",
            metadata=metadata,
        )
        return

    if item_type in {"userMessage", "agentMessage"}:
        return

    logger.debug("Unhandled item/completed type: %s", item_type)


def extract_tool_item(item_type: str, item: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    """Extract `(name, args, output)` tuple for tool-like items."""
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
        return "file_edit", {"files": file_paths}, str(item.get("status", "applied"))

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
        output = json.dumps(result, default=str) if result is not None else "completed"
        return name, collab_args, output

    return item_type, {}, "completed"


def extract_thought_text(item_type: str, item: dict[str, Any]) -> str:
    """Extract display text for a thought-like item payload."""
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

