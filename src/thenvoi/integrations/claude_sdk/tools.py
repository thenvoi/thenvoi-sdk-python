"""
Thenvoi platform tools exposed as MCP SDK server for Claude Agent SDK.

This module creates an in-process MCP SDK server that exposes Thenvoi
platform tools to Claude. Tools receive room_id from Claude (which knows
it from the system prompt) and execute real API calls.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

try:
    from claude_agent_sdk import create_sdk_mcp_server, tool
except ImportError as e:
    raise ImportError(
        "claude-agent-sdk is required for Claude SDK examples.\n"
        "Install with: pip install claude-agent-sdk\n"
        "Or: uv add claude-agent-sdk"
    ) from e

from thenvoi.runtime.tool_binding_factory import (
    ToolBindingFactory,
    ToolBindingLookupError,
)
from thenvoi.runtime.tool_bridge import ToolExecutionError
from thenvoi.runtime.tool_sessions import mcp_text_error, mcp_text_result
from thenvoi.runtime.tools import CHAT_TOOL_NAMES, mcp_tool_names
from thenvoi.integrations.claude_sdk.tool_state import (
    get_participant_handles,
    get_tools,
    parse_mention_handles,
    update_participants_cache_for_add,
    update_participants_cache_for_remove,
)

logger = logging.getLogger(__name__)

# Tool names as constants (MCP naming convention: mcp__{server}__{tool})
# Derived from TOOL_MODELS — single source of truth (chat tools only, no contacts/memory)
THENVOI_CHAT_TOOLS: list[str] = mcp_tool_names(CHAT_TOOL_NAMES)

_THENVOI_TOOLS: list[str] = THENVOI_CHAT_TOOLS


def __getattr__(name: str) -> Any:
    if name == "THENVOI_TOOLS":
        warnings.warn(
            "THENVOI_TOOLS is deprecated, use THENVOI_CHAT_TOOLS instead. "
            f"Note: this contains only chat tools ({len(_THENVOI_TOOLS)}). "
            "For all tools including contacts and memory, use "
            "thenvoi.adapters.claude_sdk.THENVOI_ALL_TOOLS.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _THENVOI_TOOLS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass(frozen=True)
class ClaudeToolDescriptor:
    """Declarative mapping from MCP payloads to platform tool execution."""

    name: str
    description: str
    schema: dict[str, type]
    map_args: Callable[[dict[str, Any]], dict[str, Any]]
    format_success: Callable[[Any, dict[str, Any]], dict[str, Any]]
    on_success: Callable[[Any, str, Any], None] | None = None
    add_handle_guidance_on_error: bool = False


def _build_claude_tool(
    agent: Any,
    descriptor: ClaudeToolDescriptor,
    tool_bindings: ToolBindingFactory,
) -> Any:
    @tool(descriptor.name, descriptor.description, descriptor.schema)
    async def _handler(args: dict[str, Any]) -> dict[str, Any]:
        room_id = str(args.get("room_id", ""))
        tool_args = descriptor.map_args(args)

        try:
            result = await tool_bindings.invoke_room_tool(
                room_id=room_id,
                get_tools=lambda current_room_id: get_tools(agent, current_room_id),
                tool_name=descriptor.name,
                arguments=tool_args,
                missing_tools_message=f"No tools available for room {room_id}",
            )
        except ToolBindingLookupError as error:
            return mcp_text_error(str(error))
        except ToolExecutionError as error:
            message = error.failure.message
            if descriptor.add_handle_guidance_on_error:
                available = get_participant_handles(agent, room_id)
                message = (
                    f"{message}. Available handles: {available}. "
                    "Use participant handles from the list."
                )
            return mcp_text_error(message)

        if descriptor.on_success is not None:
            descriptor.on_success(agent, room_id, result)

        return mcp_text_result(descriptor.format_success(result, args))

    return _handler


_CLAUDE_TOOL_DESCRIPTORS: tuple[ClaudeToolDescriptor, ...] = (
    ClaudeToolDescriptor(
        name="thenvoi_send_message",
        description=(
            "Send a message to the Thenvoi chat room. Use this to respond to users. "
            "You MUST use this tool to communicate - text responses without this tool "
            "won't reach users."
        ),
        schema={"room_id": str, "content": str, "mentions": str},
        map_args=lambda args: {
            "content": str(args.get("content", "")),
            "mentions": parse_mention_handles(args.get("mentions", "[]")),
        },
        format_success=lambda _result, _args: {
            "status": "success",
            "message": "Message sent",
        },
        add_handle_guidance_on_error=True,
    ),
    ClaudeToolDescriptor(
        name="thenvoi_send_event",
        description=(
            "Send an event (thought, tool_call, tool_result, error) to the chat room "
            "for transparency."
        ),
        schema={"room_id": str, "content": str, "message_type": str},
        map_args=lambda args: {
            "content": str(args.get("content", "")),
            "message_type": str(args.get("message_type", "thought")),
        },
        format_success=lambda _result, _args: {
            "status": "success",
            "message": "Event sent",
        },
    ),
    ClaudeToolDescriptor(
        name="thenvoi_add_participant",
        description=(
            "Add a participant (user or agent) to the chat room by name. "
            "Use thenvoi_lookup_peers first to see available participants."
        ),
        schema={"room_id": str, "name": str, "role": str},
        map_args=lambda args: {
            "name": str(args.get("name", "")),
            "role": str(args.get("role", "member")),
        },
        format_success=lambda result, args: {
            "status": "success",
            "message": (
                f"Participant '{args.get('name', '')}' added as "
                f"{args.get('role', 'member')}"
            ),
            **result,
        },
        on_success=lambda agent, room_id, result: update_participants_cache_for_add(
            agent,
            room_id,
            result,
        ),
    ),
    ClaudeToolDescriptor(
        name="thenvoi_remove_participant",
        description="Remove a participant from the chat room by name.",
        schema={"room_id": str, "name": str},
        map_args=lambda args: {"name": str(args.get("name", ""))},
        format_success=lambda result, args: {
            "status": "success",
            "message": f"Participant '{args.get('name', '')}' removed",
            **result,
        },
        on_success=lambda agent, room_id, result: update_participants_cache_for_remove(
            agent,
            room_id,
            str(result.get("id", "")),
        ),
    ),
    ClaudeToolDescriptor(
        name="thenvoi_get_participants",
        description="Get list of participants currently in the chat room.",
        schema={"room_id": str},
        map_args=lambda _args: {},
        format_success=lambda result, _args: {
            "status": "success",
            "participants": result,
            "count": len(result),
        },
    ),
    ClaudeToolDescriptor(
        name="thenvoi_lookup_peers",
        description=(
            "Look up available users and agents that can be added to the chat room. "
            "Returns peers NOT already in the room."
        ),
        schema={"room_id": str, "page": int, "page_size": int},
        map_args=lambda args: {
            "page": int(args.get("page", 1)),
            "page_size": int(args.get("page_size", 50)),
        },
        format_success=lambda result, _args: {"status": "success", **result},
    ),
    ClaudeToolDescriptor(
        name="thenvoi_create_chatroom",
        description=(
            "Create a new chat room. Optionally link it to a task by providing a task_id."
        ),
        schema={"room_id": str, "task_id": str},
        map_args=lambda args: {"task_id": args.get("task_id") or None},
        format_success=lambda result, _args: {
            "status": "success",
            "message": "Chat room created",
            "room_id": result,
        },
    ),
)


def create_thenvoi_mcp_server(agent: Any):
    """
    Create MCP SDK server for Thenvoi platform chat tools.

    Tools receive room_id from Claude and execute real API calls via AgentTools.
    """
    tool_bindings = ToolBindingFactory(binding_logger=logger)
    server_tools = [
        _build_claude_tool(agent, descriptor, tool_bindings)
        for descriptor in _CLAUDE_TOOL_DESCRIPTORS
    ]
    server = create_sdk_mcp_server(name="thenvoi", version="1.0.0", tools=server_tools)
    logger.info(
        "Thenvoi MCP SDK server created with %d real tools",
        len(THENVOI_CHAT_TOOLS),
    )
    return server
