"""Standalone stdio MCP server exposing Thenvoi platform tools.

This module creates an MCP server that external ACP agents (codex-acp,
claude-code, gemini-cli, etc.) can spawn as a subprocess to access Thenvoi
platform tools. The server communicates via stdin/stdout using the MCP
protocol.

Environment variables (set by ACPClientAdapter automatically):
    THENVOI_API_KEY: API key for authentication.
    THENVOI_REST_URL: Base URL for Thenvoi REST API.
    THENVOI_ROOM_ID: Room ID for tool execution context.

Usage:
    # Spawned automatically by ACP agents via McpServerStdio config.
    # Can also be run directly for testing:
    python -m thenvoi.integrations.acp.mcp_server
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Create the MCP server instance
mcp = FastMCP("thenvoi", log_level="WARNING")


def _get_config() -> tuple[str, str, str]:
    """Read configuration from environment variables.

    Returns:
        Tuple of (api_key, rest_url, room_id).

    Raises:
        ValueError: If required environment variables are missing.
    """
    api_key = os.environ.get("THENVOI_API_KEY", "")
    rest_url = os.environ.get("THENVOI_REST_URL", "https://app.thenvoi.com")
    room_id = os.environ.get("THENVOI_ROOM_ID", "")

    if not api_key:
        raise ValueError("THENVOI_API_KEY environment variable is required")
    if not room_id:
        raise ValueError("THENVOI_ROOM_ID environment variable is required")

    return api_key, rest_url, room_id


def _get_tools(api_key: str, rest_url: str, room_id: str) -> Any:
    """Create AgentTools instance for the given room.

    Args:
        api_key: Thenvoi API key.
        rest_url: Base URL for REST API.
        room_id: Room ID for tool context.

    Returns:
        AgentTools instance bound to the room.
    """
    from thenvoi.client.rest import AsyncRestClient
    from thenvoi.runtime.tools import AgentTools

    rest = AsyncRestClient(base_url=rest_url, api_key=api_key)
    return AgentTools(room_id, rest)


def _make_result(data: Any) -> str:
    """Format tool result as JSON string."""
    return json.dumps(data, default=str)


def _make_error(error: str) -> str:
    """Format error result as JSON string."""
    return json.dumps({"status": "error", "message": error})


# ── Chat Tools ──


@mcp.tool()
async def thenvoi_send_message(content: str, mentions: str = "[]") -> str:
    """Send a message to the Thenvoi chat room.

    Use this to respond to users or other agents. Messages require at least
    one @mention in the mentions array.

    Args:
        content: The message content to send.
        mentions: JSON array of participant handles, e.g. '["@alice", "@bob/agent"]'.

    Returns:
        JSON string with status and message details.
    """
    try:
        api_key, rest_url, room_id = _get_config()
        tools = _get_tools(api_key, rest_url, room_id)

        mention_handles: list[str] = []
        if mentions:
            try:
                mention_handles = (
                    json.loads(mentions) if isinstance(mentions, str) else mentions
                )
            except json.JSONDecodeError:
                pass

        await tools.send_message(content, mention_handles)
        return _make_result({"status": "success", "message": "Message sent"})
    except Exception as e:
        logger.exception("thenvoi_send_message failed: %s", e)
        return _make_error(str(e))


@mcp.tool()
async def thenvoi_send_event(
    content: str, message_type: str = "thought", metadata: str = "{}"
) -> str:
    """Send an event (thought, tool_call, tool_result, error, task) to the chat room.

    Args:
        content: Human-readable event content.
        message_type: Type of event - "thought", "tool_call", "tool_result", "error", "task".
        metadata: Optional JSON object with structured data.

    Returns:
        JSON string with status.
    """
    try:
        api_key, rest_url, room_id = _get_config()
        tools = _get_tools(api_key, rest_url, room_id)

        meta = None
        if metadata and metadata != "{}":
            try:
                meta = json.loads(metadata) if isinstance(metadata, str) else metadata
            except json.JSONDecodeError:
                pass

        await tools.send_event(content, message_type, meta)
        return _make_result({"status": "success", "message": "Event sent"})
    except Exception as e:
        logger.exception("thenvoi_send_event failed: %s", e)
        return _make_error(str(e))


@mcp.tool()
async def thenvoi_add_participant(name: str, role: str = "member") -> str:
    """Add a participant (user or agent) to the chat room by name.

    Use thenvoi_lookup_peers first to see available participants.

    Args:
        name: Name of participant to add.
        role: Role for the participant - "member", "admin", or "owner".

    Returns:
        JSON string with participant details.
    """
    try:
        api_key, rest_url, room_id = _get_config()
        tools = _get_tools(api_key, rest_url, room_id)
        result = await tools.add_participant(name, role)
        return _make_result({"status": "success", **result})
    except Exception as e:
        logger.exception("thenvoi_add_participant failed: %s", e)
        return _make_error(str(e))


@mcp.tool()
async def thenvoi_remove_participant(name: str) -> str:
    """Remove a participant from the chat room by name.

    Args:
        name: Name of the participant to remove.

    Returns:
        JSON string with removal status.
    """
    try:
        api_key, rest_url, room_id = _get_config()
        tools = _get_tools(api_key, rest_url, room_id)
        result = await tools.remove_participant(name)
        return _make_result({"status": "success", **result})
    except Exception as e:
        logger.exception("thenvoi_remove_participant failed: %s", e)
        return _make_error(str(e))


@mcp.tool()
async def thenvoi_get_participants() -> str:
    """Get list of participants currently in the chat room.

    Returns:
        JSON string with participants list and count.
    """
    try:
        api_key, rest_url, room_id = _get_config()
        tools = _get_tools(api_key, rest_url, room_id)
        participants = await tools.get_participants()
        return _make_result(
            {
                "status": "success",
                "participants": participants,
                "count": len(participants),
            }
        )
    except Exception as e:
        logger.exception("thenvoi_get_participants failed: %s", e)
        return _make_error(str(e))


@mcp.tool()
async def thenvoi_lookup_peers(page: int = 1, page_size: int = 50) -> str:
    """Look up available users and agents that can be added to the chat room.

    Returns peers NOT already in the room.

    Args:
        page: Page number (default 1).
        page_size: Items per page (default 50, max 100).

    Returns:
        JSON string with peers list and metadata.
    """
    try:
        api_key, rest_url, room_id = _get_config()
        tools = _get_tools(api_key, rest_url, room_id)
        result = await tools.lookup_peers(page, page_size)
        return _make_result({"status": "success", **result})
    except Exception as e:
        logger.exception("thenvoi_lookup_peers failed: %s", e)
        return _make_error(str(e))


@mcp.tool()
async def thenvoi_create_chatroom(task_id: str = "") -> str:
    """Create a new chat room. Optionally link it to a task.

    Args:
        task_id: Optional task ID to associate with the room.

    Returns:
        JSON string with new room ID.
    """
    try:
        api_key, rest_url, room_id = _get_config()
        tools = _get_tools(api_key, rest_url, room_id)
        new_room_id = await tools.create_chatroom(task_id or None)
        return _make_result(
            {
                "status": "success",
                "message": "Chat room created",
                "room_id": new_room_id,
            }
        )
    except Exception as e:
        logger.exception("thenvoi_create_chatroom failed: %s", e)
        return _make_error(str(e))


def main() -> None:
    """Run the MCP server on stdio."""
    # Validate config early
    try:
        _get_config()
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
