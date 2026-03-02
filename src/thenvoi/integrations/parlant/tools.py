"""
Parlant tool definitions that wrap Thenvoi platform tools.

These tools are defined at server startup and use a session-keyed registry
to access the current room's tools during execution.

NOTE: We intentionally do NOT use ``from __future__ import annotations`` here
because Parlant's ``@p.tool`` decorator checks annotation types at runtime.
"""

import logging
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Optional

from thenvoi.core.seams import BoundaryResult
from thenvoi.runtime.tool_binding_factory import (
    ToolBindingFactory,
    ToolBindingLookupError,
)
from thenvoi.runtime.tool_bridge import build_tool_failure
from thenvoi.runtime.tool_sessions import ToolSessionRegistry

logger = logging.getLogger(__name__)
_TOOL_BINDINGS = ToolBindingFactory(binding_logger=logger)

_SESSION_REGISTRY = ToolSessionRegistry[Any]()

# Backward-compatible aliases for tests/introspection of session caches.
_session_tools = _SESSION_REGISTRY._tools_by_session
_session_message_sent = _SESSION_REGISTRY._message_sent_by_session


def set_session_tools(session_id: str, tools: Optional[Any]) -> None:
    """Set the tools for a specific Parlant session."""
    _SESSION_REGISTRY.set_tools(session_id, tools)
    logger.debug("Set tools for session %s: %s", session_id, tools is not None)


def get_session_tools(session_id: str) -> Optional[Any]:
    """Get the tools for a specific Parlant session."""
    tools = _SESSION_REGISTRY.get_tools(session_id)
    logger.debug(
        "Get tools for session_id=%s: found=%s, available_sessions=%s",
        session_id,
        tools is not None,
        _SESSION_REGISTRY.active_sessions(),
    )
    return tools


def mark_message_sent(session_id: str) -> None:
    """Mark that a message was sent via the send_message tool for this session."""
    _SESSION_REGISTRY.mark_message_sent(session_id)
    logger.debug("Marked message sent for session %s", session_id)


def was_message_sent(session_id: str) -> bool:
    """Check if a message was sent via the send_message tool for this session."""
    return _SESSION_REGISTRY.was_message_sent(session_id)


def set_current_tools(tools: Optional[Any]) -> None:
    """Deprecated: Use set_session_tools instead."""
    warnings.warn(
        "set_current_tools is deprecated, use set_session_tools instead",
        DeprecationWarning,
        stacklevel=2,
    )


def get_current_tools() -> Optional[Any]:
    """Deprecated: Use get_session_tools instead."""
    warnings.warn(
        "get_current_tools is deprecated, use get_session_tools instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return None


ToolRunner = Callable[
    [Any, str, dict[str, Any]],
    Awaitable[BoundaryResult[Any]],
]


def _create_tool_runner() -> ToolRunner:
    async def _run_tool(
        context: Any,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> BoundaryResult[Any]:
        tools = get_session_tools(context.session_id)
        if not tools:
            logger.error(
                "[Parlant Tool] %s: No tools available for session %s",
                tool_name,
                context.session_id,
            )
            return BoundaryResult.failure(
                code="tools_unavailable",
                message="Error: No tools available in current context",
            )

        try:
            result = await _TOOL_BINDINGS.invoke_session_tool(
                session_id=context.session_id,
                registry=_SESSION_REGISTRY,
                tool_name=tool_name,
                arguments=arguments,
            )
            return BoundaryResult.success(result)
        except ToolBindingLookupError as error:
            return BoundaryResult.failure(
                code="tool_lookup_failed",
                message=str(error),
                details={"tool_name": tool_name},
            )
        except Exception as error:
            failure = build_tool_failure(tool_name, arguments, error)
            logger.error("[Parlant Tool] %s failed: %s", tool_name, error, exc_info=True)
            return BoundaryResult.failure(
                code="tool_execution_failed",
                message=failure.message,
                details={"tool_name": tool_name},
            )

    return _run_tool


def _parse_mentions(mentions: str) -> list[str]:
    return [mention.strip() for mention in mentions.split(",") if mention.strip()]


def _format_lookup_peers_result(tool_result_type: Any, result: Any) -> Any:
    if not isinstance(result, dict):
        return tool_result_type(data=str(result))

    peers = result.get("peers", [])
    metadata = result.get("metadata", {})
    if not peers:
        return tool_result_type(data="No available agents found")

    lines = [
        (
            "Available agents "
            f"(page {metadata.get('page', 1)} of {metadata.get('total_pages', 1)}):"
        )
    ]
    for peer in peers:
        name = peer.get("name", "Unknown")
        description = peer.get("description", "No description")
        peer_type = peer.get("type", "Agent")
        lines.append(f"- {name} ({peer_type}): {description}")
    return tool_result_type(data="\n".join(lines))


def _format_get_participants_result(tool_result_type: Any, result: Any) -> Any:
    if not isinstance(result, list):
        return tool_result_type(data=str(result))
    if not result:
        return tool_result_type(data="No participants in the room")

    lines = ["Current participants:"]
    for participant in result:
        name = participant.get("name", "Unknown")
        participant_type = participant.get("type", "Unknown")
        lines.append(f"- {name} ({participant_type})")
    return tool_result_type(data="\n".join(lines))


async def _execute_descriptor_tool(
    context: Any,
    descriptor: "ParlantToolDescriptor",
    run_tool: ToolRunner,
    arguments: dict[str, Any],
) -> BoundaryResult[Any]:
    return await run_tool(context, descriptor.name, arguments)


def _to_tool_result_error(tool_result_type: Any, outcome: BoundaryResult[Any]) -> Any:
    error_message = (
        outcome.error.message
        if outcome.error is not None
        else "Error: Tool execution failed"
    )
    return tool_result_type(data=error_message)


@dataclass(frozen=True)
class ParlantToolDescriptor:
    """Declarative tool table for Parlant wrapper generation."""

    name: str
    description: str
    kind: str


_PARLANT_CHAT_DESCRIPTORS: tuple[ParlantToolDescriptor, ...] = (
    ParlantToolDescriptor(
        name="thenvoi_send_message",
        description="Send a message to the chat room using required mentions.",
        kind="send_message",
    ),
    ParlantToolDescriptor(
        name="thenvoi_send_event",
        description="Send an event update to the chat room.",
        kind="send_event",
    ),
    ParlantToolDescriptor(
        name="thenvoi_add_participant",
        description="Invite an agent or user to join the current room.",
        kind="add_participant",
    ),
    ParlantToolDescriptor(
        name="thenvoi_remove_participant",
        description="Remove a participant from the current room.",
        kind="remove_participant",
    ),
    ParlantToolDescriptor(
        name="thenvoi_lookup_peers",
        description="List available peers that can be added to the room.",
        kind="lookup_peers",
    ),
    ParlantToolDescriptor(
        name="thenvoi_get_participants",
        description="List participants currently present in the room.",
        kind="get_participants",
    ),
    ParlantToolDescriptor(
        name="thenvoi_create_chatroom",
        description="Create a new chat room, optionally linked to a task.",
        kind="create_chatroom",
    ),
)


def _build_send_message_tool(
    p: Any,
    descriptor: ParlantToolDescriptor,
    tool_context_type: Any,
    tool_result_type: Any,
    run_tool: ToolRunner,
) -> Any:
    @p.tool
    async def thenvoi_send_message(
        context: tool_context_type,
        content: str,
        mentions: str,
    ) -> tool_result_type:
        """Send a message to the chat room using required mentions."""
        mention_list = _parse_mentions(mentions)
        if not mention_list:
            return tool_result_type(data="Error: At least one mention is required")

        outcome = await _execute_descriptor_tool(
            context,
            descriptor,
            run_tool,
            {"content": content, "mentions": mention_list},
        )
        if not outcome.is_ok:
            return _to_tool_result_error(tool_result_type, outcome)
        result = outcome.value

        mark_message_sent(context.session_id)
        logger.info("[Parlant Tool] Message sent via %s: %s", descriptor.name, result)
        return tool_result_type(data=f"Message sent to {', '.join(mention_list)}")

    return thenvoi_send_message


def _build_send_event_tool(
    p: Any,
    descriptor: ParlantToolDescriptor,
    tool_context_type: Any,
    tool_result_type: Any,
    run_tool: ToolRunner,
) -> Any:
    @p.tool
    async def thenvoi_send_event(
        context: tool_context_type,
        content: str,
        message_type: str,
    ) -> tool_result_type:
        """Send an event update to the chat room."""
        if message_type not in ("thought", "error", "task"):
            return tool_result_type(
                data=(
                    f"Error: Invalid message_type '{message_type}'. "
                    "Use 'thought', 'error', or 'task'"
                )
            )

        outcome = await _execute_descriptor_tool(
            context,
            descriptor,
            run_tool,
            {"content": content, "message_type": message_type, "metadata": None},
        )
        if not outcome.is_ok:
            return _to_tool_result_error(tool_result_type, outcome)
        return tool_result_type(data=f"Event ({message_type}) sent successfully")

    return thenvoi_send_event


def _build_add_participant_tool(
    p: Any,
    descriptor: ParlantToolDescriptor,
    tool_context_type: Any,
    tool_result_type: Any,
    run_tool: ToolRunner,
) -> Any:
    @p.tool
    async def thenvoi_add_participant(
        context: tool_context_type,
        name: str,
    ) -> tool_result_type:
        """Invite an agent or user to join the current room."""
        outcome = await _execute_descriptor_tool(
            context,
            descriptor,
            run_tool,
            {"name": name, "role": "member"},
        )
        if not outcome.is_ok:
            return _to_tool_result_error(tool_result_type, outcome)
        result = outcome.value

        status = result.get("status", "added") if isinstance(result, dict) else "added"
        if status == "already_in_room":
            return tool_result_type(data=f"'{name}' is already in the room - no action needed")
        return tool_result_type(data=f"Successfully added '{name}' to the room")

    return thenvoi_add_participant


def _build_remove_participant_tool(
    p: Any,
    descriptor: ParlantToolDescriptor,
    tool_context_type: Any,
    tool_result_type: Any,
    run_tool: ToolRunner,
) -> Any:
    @p.tool
    async def thenvoi_remove_participant(
        context: tool_context_type,
        name: str,
    ) -> tool_result_type:
        """Remove a participant from the current room."""
        outcome = await _execute_descriptor_tool(
            context,
            descriptor,
            run_tool,
            {"name": name},
        )
        if not outcome.is_ok:
            return _to_tool_result_error(tool_result_type, outcome)
        return tool_result_type(data=f"Successfully removed '{name}' from the room")

    return thenvoi_remove_participant


def _build_lookup_peers_tool(
    p: Any,
    descriptor: ParlantToolDescriptor,
    tool_context_type: Any,
    tool_result_type: Any,
    run_tool: ToolRunner,
) -> Any:
    @p.tool
    async def thenvoi_lookup_peers(
        context: tool_context_type,
    ) -> tool_result_type:
        """List available peers that can be added to the room."""
        outcome = await _execute_descriptor_tool(
            context,
            descriptor,
            run_tool,
            {"page": 1, "page_size": 50},
        )
        if not outcome.is_ok:
            return _to_tool_result_error(tool_result_type, outcome)
        result = outcome.value
        return _format_lookup_peers_result(tool_result_type, result)

    return thenvoi_lookup_peers


def _build_get_participants_tool(
    p: Any,
    descriptor: ParlantToolDescriptor,
    tool_context_type: Any,
    tool_result_type: Any,
    run_tool: ToolRunner,
) -> Any:
    @p.tool
    async def thenvoi_get_participants(
        context: tool_context_type,
    ) -> tool_result_type:
        """List participants currently present in the room."""
        outcome = await _execute_descriptor_tool(
            context,
            descriptor,
            run_tool,
            {},
        )
        if not outcome.is_ok:
            return _to_tool_result_error(tool_result_type, outcome)
        result = outcome.value
        return _format_get_participants_result(tool_result_type, result)

    return thenvoi_get_participants


def _build_create_chatroom_tool(
    p: Any,
    descriptor: ParlantToolDescriptor,
    tool_context_type: Any,
    tool_result_type: Any,
    run_tool: ToolRunner,
) -> Any:
    @p.tool
    async def thenvoi_create_chatroom(
        context: tool_context_type,
        task_id: str = "",
    ) -> tool_result_type:
        """Create a new chat room, optionally linked to a task."""
        outcome = await _execute_descriptor_tool(
            context,
            descriptor,
            run_tool,
            {"task_id": task_id if task_id else None},
        )
        if not outcome.is_ok:
            return _to_tool_result_error(tool_result_type, outcome)
        result = outcome.value
        return tool_result_type(data=f"Created new chat room: {result}")

    return thenvoi_create_chatroom


def _build_descriptor_tool(
    p: Any,
    descriptor: ParlantToolDescriptor,
    tool_context_type: Any,
    tool_result_type: Any,
    run_tool: ToolRunner,
) -> Any:
    if descriptor.kind == "send_message":
        return _build_send_message_tool(
            p,
            descriptor,
            tool_context_type,
            tool_result_type,
            run_tool,
        )
    if descriptor.kind == "send_event":
        return _build_send_event_tool(
            p,
            descriptor,
            tool_context_type,
            tool_result_type,
            run_tool,
        )
    if descriptor.kind == "add_participant":
        return _build_add_participant_tool(
            p,
            descriptor,
            tool_context_type,
            tool_result_type,
            run_tool,
        )
    if descriptor.kind == "remove_participant":
        return _build_remove_participant_tool(
            p,
            descriptor,
            tool_context_type,
            tool_result_type,
            run_tool,
        )
    if descriptor.kind == "lookup_peers":
        return _build_lookup_peers_tool(
            p,
            descriptor,
            tool_context_type,
            tool_result_type,
            run_tool,
        )
    if descriptor.kind == "get_participants":
        return _build_get_participants_tool(
            p,
            descriptor,
            tool_context_type,
            tool_result_type,
            run_tool,
        )
    if descriptor.kind == "create_chatroom":
        return _build_create_chatroom_tool(
            p,
            descriptor,
            tool_context_type,
            tool_result_type,
            run_tool,
        )
    raise ValueError(f"Unknown descriptor kind: {descriptor.kind}")


def create_parlant_tools() -> list[Any]:
    """Create Parlant tool definitions backed by session-scoped Thenvoi tools."""
    try:
        import parlant.sdk as p
        from parlant.core.tools import ToolContext, ToolResult
    except ImportError:
        logger.warning("Parlant SDK not installed, skipping tool creation")
        return []

    run_tool = _create_tool_runner()
    return [
        _build_descriptor_tool(p, descriptor, ToolContext, ToolResult, run_tool)
        for descriptor in _PARLANT_CHAT_DESCRIPTORS
    ]
