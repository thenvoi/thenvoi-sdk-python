"""
Convert AgentTools to Parlant SDK tool format.

This module provides the bridge between the SDK's AgentTools and Parlant's
tool format for use with the Parlant SDK.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Literal

from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.runtime.tools import get_tool_description

logger = logging.getLogger(__name__)


@dataclass
class ParlantToolContext:
    """
    Context provided to Parlant tools for accessing room-specific resources.

    This mimics Parlant's ToolContext but provides access to Thenvoi's
    AgentToolsProtocol for the current room.
    """

    room_id: str
    tools: AgentToolsProtocol
    session_id: str | None = None


class ParlantToolResult:
    """
    Result of a Parlant tool execution.

    Matches Parlant SDK's ToolResult structure.
    """

    def __init__(
        self,
        data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.data = data or {}
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "data": self.data,
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        return json.dumps(self.data, default=str)


# Type alias for tool functions
ParlantToolFunc = Callable[
    [ParlantToolContext, dict[str, Any]],
    Coroutine[Any, Any, ParlantToolResult],
]


@dataclass
class ParlantToolDefinition:
    """Definition of a Parlant-compatible tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: ParlantToolFunc


async def _send_message_tool(
    ctx: ParlantToolContext, args: dict[str, Any]
) -> ParlantToolResult:
    """Send a message to the chat room."""
    try:
        content = args.get("content", "")
        mentions_raw = args.get("mentions", [])

        # Parse mentions if it's a string
        mentions: list[str] = []
        if isinstance(mentions_raw, str):
            try:
                mentions = json.loads(mentions_raw)
            except json.JSONDecodeError:
                mentions = []
        elif isinstance(mentions_raw, list):
            mentions = mentions_raw

        result = await ctx.tools.send_message(content, mentions)

        return ParlantToolResult(
            data={"status": "success", "result": result},
        )
    except Exception as e:
        logger.error(f"send_message failed: {e}", exc_info=True)
        return ParlantToolResult(
            data={"status": "error", "message": str(e)},
        )


async def _send_event_tool(
    ctx: ParlantToolContext, args: dict[str, Any]
) -> ParlantToolResult:
    """Send an event to the chat room."""
    try:
        content = args.get("content", "")
        message_type: Literal["thought", "error", "task"] = args.get(
            "message_type", "thought"
        )

        result = await ctx.tools.send_event(content, message_type, None)

        return ParlantToolResult(
            data={"status": "success", "result": result},
        )
    except Exception as e:
        logger.error(f"send_event failed: {e}", exc_info=True)
        return ParlantToolResult(
            data={"status": "error", "message": str(e)},
        )


async def _add_participant_tool(
    ctx: ParlantToolContext, args: dict[str, Any]
) -> ParlantToolResult:
    """Add a participant to the chat room."""
    try:
        name = args.get("name", "")
        role = args.get("role", "member")

        result = await ctx.tools.add_participant(name, role)

        return ParlantToolResult(
            data={
                "status": "success",
                "message": f"Participant '{name}' added as {role}",
                **result,
            },
        )
    except Exception as e:
        logger.error(f"add_participant failed: {e}", exc_info=True)
        return ParlantToolResult(
            data={"status": "error", "message": str(e)},
        )


async def _remove_participant_tool(
    ctx: ParlantToolContext, args: dict[str, Any]
) -> ParlantToolResult:
    """Remove a participant from the chat room."""
    try:
        name = args.get("name", "")

        result = await ctx.tools.remove_participant(name)

        return ParlantToolResult(
            data={
                "status": "success",
                "message": f"Participant '{name}' removed",
                **result,
            },
        )
    except Exception as e:
        logger.error(f"remove_participant failed: {e}", exc_info=True)
        return ParlantToolResult(
            data={"status": "error", "message": str(e)},
        )


async def _get_participants_tool(
    ctx: ParlantToolContext, args: dict[str, Any]
) -> ParlantToolResult:
    """Get participants in the chat room."""
    try:
        participants = await ctx.tools.get_participants()

        return ParlantToolResult(
            data={
                "status": "success",
                "participants": participants,
                "count": len(participants),
            },
        )
    except Exception as e:
        logger.error(f"get_participants failed: {e}", exc_info=True)
        return ParlantToolResult(
            data={"status": "error", "message": str(e)},
        )


async def _lookup_peers_tool(
    ctx: ParlantToolContext, args: dict[str, Any]
) -> ParlantToolResult:
    """Look up available peers."""
    try:
        page = args.get("page", 1)
        page_size = args.get("page_size", 50)

        result = await ctx.tools.lookup_peers(page, page_size)

        return ParlantToolResult(
            data={"status": "success", **result},
        )
    except Exception as e:
        logger.error(f"lookup_peers failed: {e}", exc_info=True)
        return ParlantToolResult(
            data={"status": "error", "message": str(e)},
        )


async def _create_chatroom_tool(
    ctx: ParlantToolContext, args: dict[str, Any]
) -> ParlantToolResult:
    """Create a new chat room."""
    try:
        task_id = args.get("task_id") or None

        room_id = await ctx.tools.create_chatroom(task_id)

        return ParlantToolResult(
            data={
                "status": "success",
                "message": "Chat room created",
                "room_id": room_id,
            },
        )
    except Exception as e:
        logger.error(f"create_chatroom failed: {e}", exc_info=True)
        return ParlantToolResult(
            data={"status": "error", "message": str(e)},
        )


def create_parlant_tools() -> list[ParlantToolDefinition]:
    """
    Create Parlant-compatible tool definitions for Thenvoi platform tools.

    Returns:
        List of ParlantToolDefinition instances
    """
    return [
        ParlantToolDefinition(
            name="send_message",
            description=get_tool_description("send_message"),
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The message content to send",
                    },
                    "mentions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of participant names to mention",
                        "default": [],
                    },
                },
                "required": ["content"],
            },
            func=_send_message_tool,
        ),
        ParlantToolDefinition(
            name="send_event",
            description=get_tool_description("send_event"),
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The event content",
                    },
                    "message_type": {
                        "type": "string",
                        "enum": ["thought", "error", "task"],
                        "description": "Type of event",
                        "default": "thought",
                    },
                },
                "required": ["content"],
            },
            func=_send_event_tool,
        ),
        ParlantToolDefinition(
            name="add_participant",
            description=get_tool_description("add_participant"),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of participant to add",
                    },
                    "role": {
                        "type": "string",
                        "description": "Role (member/admin)",
                        "default": "member",
                    },
                },
                "required": ["name"],
            },
            func=_add_participant_tool,
        ),
        ParlantToolDefinition(
            name="remove_participant",
            description=get_tool_description("remove_participant"),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of participant to remove",
                    },
                },
                "required": ["name"],
            },
            func=_remove_participant_tool,
        ),
        ParlantToolDefinition(
            name="get_participants",
            description=get_tool_description("get_participants"),
            parameters={
                "type": "object",
                "properties": {},
            },
            func=_get_participants_tool,
        ),
        ParlantToolDefinition(
            name="lookup_peers",
            description=get_tool_description("lookup_peers"),
            parameters={
                "type": "object",
                "properties": {
                    "page": {
                        "type": "integer",
                        "description": "Page number",
                        "default": 1,
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "Items per page",
                        "default": 50,
                    },
                },
            },
            func=_lookup_peers_tool,
        ),
        ParlantToolDefinition(
            name="create_chatroom",
            description=get_tool_description("create_chatroom"),
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Optional task ID to associate with room",
                    },
                },
            },
            func=_create_chatroom_tool,
        ),
    ]


def get_tool_schemas_for_parlant() -> list[dict[str, Any]]:
    """
    Get OpenAI-compatible tool schemas for Parlant tool registration.

    Returns:
        List of tool schemas in OpenAI function format
    """
    tools = create_parlant_tools()
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]
