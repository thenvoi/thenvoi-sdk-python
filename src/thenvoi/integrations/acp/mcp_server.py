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
from threading import Lock
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP

from thenvoi.runtime.tools import ALL_TOOL_NAMES

logger = logging.getLogger(__name__)

# Create the MCP server instance
mcp = FastMCP("thenvoi", log_level="WARNING")

# Module-level shared REST client and tools (initialized once on first use)
_tools_instance: Any = None
_tools_lock = Lock()


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
    if not rest_url.startswith(("http://", "https://")):
        raise ValueError("THENVOI_REST_URL must be a valid HTTP(S) URL")
    if not room_id:
        raise ValueError("THENVOI_ROOM_ID environment variable is required")

    return api_key, rest_url, room_id


def _get_tools() -> Any:
    """Get or create shared AgentTools instance.

    Reuses a single AsyncRestClient across all tool calls to avoid
    creating a new HTTP connection per invocation.

    Returns:
        AgentTools instance bound to the room.
    """
    global _tools_instance  # noqa: PLW0603  # Module-level singleton for connection reuse
    if _tools_instance is not None:
        return _tools_instance

    with _tools_lock:
        if _tools_instance is None:
            from thenvoi.client.rest import AsyncRestClient
            from thenvoi.runtime.tools import AgentTools

            api_key, rest_url, room_id = _get_config()
            rest = AsyncRestClient(base_url=rest_url, api_key=api_key)
            _tools_instance = AgentTools(room_id, rest)
    return _tools_instance


def _parse_json(value: str, fallback: Any = None) -> Any:
    """Parse a JSON string, returning fallback on decode error.

    Args:
        value: JSON string to parse.
        fallback: Value to return on parse failure.

    Returns:
        Parsed JSON value, or fallback if parsing fails.
    """
    if not value or value in ("{}", "[]"):
        return fallback
    try:
        return json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON input: %s", value[:100])
        return fallback


def _make_result(data: Any) -> str:
    """Format tool result as JSON string."""
    return json.dumps(data, default=str)


def _make_error(error: str) -> str:
    """Format error result as JSON string."""
    return json.dumps({"status": "error", "message": error})


async def _execute_tool(
    method_name: str,
    *,
    args: tuple[object, ...] = (),
    kwargs: dict[str, object],
    success_payload: Callable[[Any], dict[str, Any]] | None = None,
) -> str:
    """Invoke an AgentTools method and normalize the MCP response payload."""
    try:
        tools = _get_tools()
        result = await getattr(tools, method_name)(*args, **kwargs)
        if success_payload is None:
            if isinstance(result, dict):
                payload = {"status": "success", **result}
            else:
                raise TypeError(
                    f"Unexpected result for {method_name}: {type(result)!r}"
                )
        else:
            payload = success_payload(result)
        return _make_result(payload)
    except Exception as e:
        logger.exception("%s failed: %s", method_name, e)
        return _make_error(str(e))


def _message_payload(message: str) -> Callable[[Any], dict[str, Any]]:
    """Build a static success payload helper."""

    def payload(_: Any) -> dict[str, Any]:
        return {"status": "success", "message": message}

    return payload


def _room_payload(result: Any) -> dict[str, Any]:
    """Format create_chatroom output."""
    return {
        "status": "success",
        "message": "Chat room created",
        "room_id": result,
    }


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
    mention_handles: list[str] = _parse_json(mentions, [])
    return await _execute_tool(
        "send_message",
        args=(content, mention_handles),
        kwargs={},
        success_payload=_message_payload("Message sent"),
    )


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
    return await _execute_tool(
        "send_event",
        args=(content, message_type, _parse_json(metadata)),
        kwargs={},
        success_payload=_message_payload("Event sent"),
    )


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
    return await _execute_tool(
        "add_participant",
        kwargs={"name": name, "role": role},
    )


@mcp.tool()
async def thenvoi_remove_participant(name: str) -> str:
    """Remove a participant from the chat room by name.

    Args:
        name: Name of the participant to remove.

    Returns:
        JSON string with removal status.
    """
    return await _execute_tool(
        "remove_participant",
        kwargs={"name": name},
    )


@mcp.tool()
async def thenvoi_get_participants() -> str:
    """Get list of participants currently in the chat room.

    Returns:
        JSON string with participants list and count.
    """
    return await _execute_tool(
        "get_participants",
        kwargs={},
        success_payload=lambda participants: {
            "status": "success",
            "participants": participants,
            "count": len(participants),
        },
    )


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
    return await _execute_tool(
        "lookup_peers",
        kwargs={"page": page, "page_size": page_size},
    )


@mcp.tool()
async def thenvoi_create_chatroom(task_id: str = "") -> str:
    """Create a new chat room. Optionally link it to a task.

    Args:
        task_id: Optional task ID to associate with the room.

    Returns:
        JSON string with new room ID.
    """
    return await _execute_tool(
        "create_chatroom",
        kwargs={"task_id": task_id or None},
        success_payload=_room_payload,
    )


# ── Contact Tools ──


@mcp.tool()
async def thenvoi_list_contacts(page: int = 1, page_size: int = 50) -> str:
    """List the agent's contacts with pagination.

    Args:
        page: Page number (default 1).
        page_size: Items per page (default 50, max 100).

    Returns:
        JSON string with contacts list and metadata.
    """
    return await _execute_tool(
        "list_contacts",
        kwargs={"page": page, "page_size": page_size},
    )


@mcp.tool()
async def thenvoi_add_contact(handle: str, message: str = "") -> str:
    """Send a contact request to add someone as a contact.

    Args:
        handle: Handle of user/agent to add (e.g., '@john' or '@john/agent-name').
        message: Optional message with the request.

    Returns:
        JSON string with request id and status.
    """
    return await _execute_tool(
        "add_contact",
        kwargs={"handle": handle, "message": message or None},
    )


@mcp.tool()
async def thenvoi_remove_contact(handle: str = "", contact_id: str = "") -> str:
    """Remove an existing contact by handle or ID.

    Args:
        handle: Contact's handle (e.g., '@john').
        contact_id: Or contact record ID (UUID). Provide handle or contact_id.

    Returns:
        JSON string with removal status.
    """
    return await _execute_tool(
        "remove_contact",
        kwargs={"handle": handle or None, "contact_id": contact_id or None},
    )


@mcp.tool()
async def thenvoi_list_contact_requests(
    page: int = 1, page_size: int = 50, sent_status: str = "pending"
) -> str:
    """List received and sent contact requests.

    Args:
        page: Page number (default 1).
        page_size: Items per page per direction (default 50, max 100).
        sent_status: Filter sent requests by status (default 'pending').

    Returns:
        JSON string with received and sent request lists.
    """
    return await _execute_tool(
        "list_contact_requests",
        kwargs={
            "page": page,
            "page_size": page_size,
            "sent_status": sent_status,
        },
    )


@mcp.tool()
async def thenvoi_respond_contact_request(
    action: str, handle: str = "", request_id: str = ""
) -> str:
    """Respond to a contact request (approve, reject, or cancel).

    Args:
        action: Action to take - "approve", "reject", or "cancel".
        handle: Other party's handle.
        request_id: Or request ID (UUID). Provide handle or request_id.

    Returns:
        JSON string with updated request status.
    """
    if action not in ("approve", "reject", "cancel"):
        return _make_error(
            f"Invalid action '{action}'. Must be 'approve', 'reject', or 'cancel'."
        )
    return await _execute_tool(
        "respond_contact_request",
        kwargs={
            "action": action,
            "handle": handle or None,
            "request_id": request_id or None,
        },
    )


# ── Memory Tools ──


@mcp.tool()
async def thenvoi_list_memories(
    scope: str = "",
    system: str = "",
    type: str = "",
    segment: str = "",
    content_query: str = "",
    page_size: int = 50,
    status: str = "",
) -> str:
    """List memories accessible to the agent with optional filters.

    Args:
        scope: Filter by scope - "subject", "organization", or "all".
        system: Filter by memory system - "sensory", "working", or "long_term".
        type: Filter by memory type - "iconic", "echoic", "haptic", "episodic", "semantic", "procedural".
        segment: Filter by segment - "user", "agent", "tool", or "guideline".
        content_query: Full-text search query.
        page_size: Number of results per page (max 50).
        status: Filter by status - "active", "superseded", "archived", or "all".

    Returns:
        JSON string with memories list and metadata.
    """
    return await _execute_tool(
        "list_memories",
        kwargs={
            "scope": scope or None,
            "system": system or None,
            "type": type or None,
            "segment": segment or None,
            "content_query": content_query or None,
            "page_size": page_size,
            "status": status or None,
        },
    )


@mcp.tool()
async def thenvoi_store_memory(
    content: str,
    system: str,
    type: str,
    segment: str,
    thought: str,
    scope: str = "subject",
    subject_id: str = "",
    metadata: str = "{}",
) -> str:
    """Store a new memory entry.

    Args:
        content: The memory content.
        system: Memory system tier - "sensory", "working", or "long_term".
        type: Memory type - "iconic", "echoic", "haptic", "episodic", "semantic", "procedural".
        segment: Logical segment - "user", "agent", "tool", or "guideline".
        thought: Agent's reasoning for storing this memory.
        scope: Visibility scope - "subject" or "organization" (default "subject").
        subject_id: UUID of the subject (required for subject scope).
        metadata: Optional JSON object with tags and references.

    Returns:
        JSON string with created memory details.
    """
    kwargs: dict[str, Any] = {
        "content": content,
        "system": system,
        "type": type,
        "segment": segment,
        "thought": thought,
        "scope": scope,
    }
    meta = _parse_json(metadata)
    if subject_id:
        kwargs["subject_id"] = subject_id
    if meta:
        kwargs["metadata"] = meta
    return await _execute_tool("store_memory", kwargs=kwargs)


@mcp.tool()
async def thenvoi_get_memory(memory_id: str) -> str:
    """Retrieve a specific memory by ID.

    Args:
        memory_id: Memory ID (UUID).

    Returns:
        JSON string with memory details.
    """
    return await _execute_tool(
        "get_memory",
        kwargs={"memory_id": memory_id},
    )


@mcp.tool()
async def thenvoi_supersede_memory(memory_id: str) -> str:
    """Mark a memory as superseded (soft delete).

    Args:
        memory_id: Memory ID (UUID).

    Returns:
        JSON string with updated memory details.
    """
    return await _execute_tool(
        "supersede_memory",
        kwargs={"memory_id": memory_id},
    )


@mcp.tool()
async def thenvoi_archive_memory(memory_id: str) -> str:
    """Archive a memory (hide but preserve).

    Args:
        memory_id: Memory ID (UUID).

    Returns:
        JSON string with updated memory details.
    """
    return await _execute_tool(
        "archive_memory",
        kwargs={"memory_id": memory_id},
    )


_registered_tool_names = {name for name in ALL_TOOL_NAMES if name in globals()}
if _registered_tool_names != ALL_TOOL_NAMES:
    raise ValueError(
        f"ACP MCP server tool mismatch: missing={sorted(ALL_TOOL_NAMES - _registered_tool_names)}"
    )


def main() -> None:
    """Run the MCP server on stdio."""
    # Validate config early
    _get_config()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
