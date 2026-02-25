"""
Thenvoi platform tools exposed as MCP SDK server for Claude Agent SDK.

This module creates an in-process MCP SDK server that exposes Thenvoi
platform tools to Claude. Tools receive room_id from Claude (which knows
it from the system prompt) and execute real API calls.

Architecture:
    - Tools are defined inside create_thenvoi_mcp_server() to capture api_client
    - Each tool receives room_id as a parameter from Claude
    - Tools call the actual Thenvoi API and return real results
    - No stub pattern - Claude sees actual data
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

try:
    from claude_agent_sdk import tool, create_sdk_mcp_server
except ImportError as e:
    raise ImportError(
        "claude-agent-sdk is required for Claude SDK examples.\n"
        "Install with: pip install claude-agent-sdk\n"
        "Or: uv add claude-agent-sdk"
    ) from e

logger = logging.getLogger(__name__)


class ThenvoiApiClient(Protocol):
    """Protocol for Thenvoi API client operations needed by tools."""

    async def send_message(
        self, room_id: str, content: str, mentions: list[str] | None = None
    ) -> dict[str, Any]: ...

    async def send_event(
        self,
        room_id: str,
        content: str,
        message_type: str,
        metadata: dict | None = None,
    ) -> dict[str, Any]: ...

    async def add_participant(
        self, room_id: str, name: str, role: str = "member"
    ) -> dict[str, Any]: ...

    async def remove_participant(self, room_id: str, name: str) -> dict[str, Any]: ...

    async def get_participants(self, room_id: str) -> list[dict[str, Any]]: ...

    async def lookup_peers(
        self, room_id: str, page: int = 1, page_size: int = 50
    ) -> dict[str, Any]: ...

    async def create_chatroom(self, task_id: str | None = None) -> str: ...


def create_thenvoi_mcp_server(agent: Any):
    """
    Create MCP SDK server for Thenvoi platform tools.

    Creates an in-process MCP server that exposes Thenvoi platform tools.
    Tools receive room_id from Claude and execute real API calls via AgentTools.

    Args:
        agent: Agent instance with link.rest and runtime.executions

    Returns:
        MCP SDK server configuration

    Example:
        server = create_thenvoi_mcp_server(agent)

        options = ClaudeAgentOptions(
            mcp_servers={"thenvoi": server},
            allowed_tools=THENVOI_TOOLS
        )

    Note:
        Claude receives room_id in the system prompt and must pass it
        to each tool call.
    """
    from thenvoi.runtime.tools import AgentTools

    def _make_result(data: Any) -> dict[str, Any]:
        """Format tool result for MCP response."""
        return {"content": [{"type": "text", "text": json.dumps(data, default=str)}]}

    def _make_error(error: str) -> dict[str, Any]:
        """Format error result for MCP response."""
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"status": "error", "message": error}),
                }
            ],
            "is_error": True,
        }

    def _get_tools(room_id: str) -> AgentTools:
        """Get AgentTools for a room, with participants from execution context."""
        executions = agent.runtime.executions if agent.runtime else {}
        execution = executions.get(room_id)
        participants = execution.participants if execution else []
        return AgentTools(room_id, agent.link.rest, participants)

    def _get_participant_handles(room_id: str) -> list[str]:
        """Get list of participant handles in room."""
        executions = agent.runtime.executions if agent.runtime else {}
        execution = executions.get(room_id)
        participants = execution.participants if execution else []
        return [p.get("handle", "") for p in participants if p.get("handle")]

    @tool(
        "thenvoi_send_message",
        "Send a message to the Thenvoi chat room. Use this to respond to users. You MUST use this tool to communicate - text responses without this tool won't reach users.",
        {
            "room_id": str,
            "content": str,
            "mentions": str,  # JSON array of participant handles, e.g. '["@alice", "@bob/agent"]' or '[]'
        },
    )
    async def send_message(args: dict[str, Any]) -> dict[str, Any]:
        """Send message to chat room via API."""
        try:
            room_id = args.get("room_id", "")
            content = args.get("content", "")
            mentions_str = args.get("mentions", "[]")

            # Parse mentions JSON (handles like ["@alice", "@bob/agent"])
            mention_handles: list[str] = []
            if mentions_str:
                try:
                    mention_handles = (
                        json.loads(mentions_str)
                        if isinstance(mentions_str, str)
                        else mentions_str
                    )
                except json.JSONDecodeError:
                    pass

            logger.info("[%s] send_message: %s...", room_id, content[:100])

            # Get AgentTools for this room and send message
            tools = _get_tools(room_id)
            try:
                await tools.send_message(content, mention_handles)
            except ValueError as e:
                # Mention resolution failed
                available = _get_participant_handles(room_id)
                return _make_error(
                    f"{e}. Available handles: {available}. "
                    f"Use participant handles from the list."
                )

            return _make_result(
                {
                    "status": "success",
                    "message": "Message sent",
                }
            )

        except Exception as e:
            logger.error("send_message failed: %s", e, exc_info=True)
            return _make_error(str(e))

    @tool(
        "thenvoi_send_event",
        "Send an event (thought, tool_call, tool_result, error) to the chat room for transparency.",
        {
            "room_id": str,
            "content": str,
            "message_type": str,  # "thought", "tool_call", "tool_result", "error"
        },
    )
    async def send_event(args: dict[str, Any]) -> dict[str, Any]:
        """Send event to chat room via API."""
        try:
            room_id = args.get("room_id", "")
            content = args.get("content", "")
            message_type = args.get("message_type", "thought")

            logger.debug("[%s] send_event: type=%s", room_id, message_type)

            tools = _get_tools(room_id)
            await tools.send_event(content, message_type)

            return _make_result(
                {
                    "status": "success",
                    "message": "Event sent",
                }
            )

        except Exception as e:
            logger.error("send_event failed: %s", e, exc_info=True)
            return _make_error(str(e))

    @tool(
        "thenvoi_add_participant",
        "Add a participant (user or agent) to the chat room by name. Use thenvoi_lookup_peers first to see available participants.",
        {
            "room_id": str,
            "name": str,
            "role": str,  # "member", "admin", or "owner"
        },
    )
    async def add_participant(args: dict[str, Any]) -> dict[str, Any]:
        """Add participant to chat room via API."""
        try:
            room_id = args.get("room_id", "")
            name = args.get("name", "")
            role = args.get("role", "member")

            logger.info("[%s] add_participant: %s as %s", room_id, name, role)

            tools = _get_tools(room_id)
            result = await tools.add_participant(name, role)

            # NOTE: Race condition fix for participant mentions
            # WebSocket will eventually send participant_added event which updates
            # ExecutionContext.participants (see execution.py:701-702), but that happens
            # async. If Claude immediately tries to @mention the new participant,
            # mention resolution fails because WS event hasn't arrived yet.
            # We update the cache here to allow immediate mentions after add_participant.
            executions = agent.runtime.executions if agent.runtime else {}
            execution = executions.get(room_id)
            logger.debug(
                "[%s] add_participant: runtime=%s, executions=%s, execution=%s",
                room_id,
                agent.runtime,
                list(executions.keys()),
                execution,
            )
            if execution:
                new_participant = {
                    "id": result["id"],
                    "name": result["name"],
                    "type": "Agent",  # Default, could be User
                }
                execution.add_participant(new_participant)
                logger.info(
                    "[%s] Updated participants cache: added %s, total=%s",
                    room_id,
                    result["name"],
                    len(execution.participants),
                )

            return _make_result(
                {
                    "status": "success",
                    "message": f"Participant '{name}' added as {role}",
                    **result,
                }
            )

        except Exception as e:
            logger.error("add_participant failed: %s", e, exc_info=True)
            return _make_error(str(e))

    @tool(
        "thenvoi_remove_participant",
        "Remove a participant from the chat room by name.",
        {
            "room_id": str,
            "name": str,
        },
    )
    async def remove_participant(args: dict[str, Any]) -> dict[str, Any]:
        """Remove participant from chat room via API."""
        try:
            room_id = args.get("room_id", "")
            name = args.get("name", "")

            logger.info("[%s] remove_participant: %s", room_id, name)

            tools = _get_tools(room_id)
            result = await tools.remove_participant(name)

            # NOTE: Race condition fix - same as add_participant (see above)
            # Update cache immediately so removed participant can't be mentioned.
            executions = agent.runtime.executions if agent.runtime else {}
            execution = executions.get(room_id)
            if execution:
                execution.remove_participant(result["id"])

            return _make_result(
                {
                    "status": "success",
                    "message": f"Participant '{name}' removed",
                    **result,
                }
            )

        except Exception as e:
            logger.error("remove_participant failed: %s", e, exc_info=True)
            return _make_error(str(e))

    @tool(
        "thenvoi_get_participants",
        "Get list of participants currently in the chat room.",
        {
            "room_id": str,
        },
    )
    async def get_participants(args: dict[str, Any]) -> dict[str, Any]:
        """Get participants in chat room via API."""
        try:
            room_id = args.get("room_id", "")

            logger.debug("[%s] get_participants", room_id)

            tools = _get_tools(room_id)
            participants = await tools.get_participants()

            return _make_result(
                {
                    "status": "success",
                    "participants": participants,
                    "count": len(participants),
                }
            )

        except Exception as e:
            logger.error("get_participants failed: %s", e, exc_info=True)
            return _make_error(str(e))

    @tool(
        "thenvoi_lookup_peers",
        "Look up available users and agents that can be added to the chat room. Returns peers NOT already in the room.",
        {
            "room_id": str,
            "page": int,
            "page_size": int,
        },
    )
    async def lookup_peers(args: dict[str, Any]) -> dict[str, Any]:
        """Look up available peers via API."""
        try:
            room_id = args.get("room_id", "")
            page = args.get("page", 1)
            page_size = args.get("page_size", 50)

            logger.debug(
                "[%s] lookup_peers: page=%s, page_size=%s", room_id, page, page_size
            )

            tools = _get_tools(room_id)
            result = await tools.lookup_peers(page, page_size)

            return _make_result(
                {
                    "status": "success",
                    **result,
                }
            )

        except Exception as e:
            logger.error("lookup_peers failed: %s", e, exc_info=True)
            return _make_error(str(e))

    @tool(
        "thenvoi_create_chatroom",
        "Create a new chat room. Optionally link it to a task by providing a task_id.",
        {
            "room_id": str,
            "task_id": str,  # Optional task ID to associate with the room
        },
    )
    async def create_chatroom(args: dict[str, Any]) -> dict[str, Any]:
        """Create a new chat room via API."""
        task_id = args.get("task_id") or None
        try:
            room_id = args.get("room_id", "")
            logger.info("[%s] create_chatroom: task_id=%s", room_id, task_id)

            tools = _get_tools(room_id)
            new_room_id = await tools.create_chatroom(task_id)

            return _make_result(
                {
                    "status": "success",
                    "message": "Chat room created",
                    "room_id": new_room_id,
                }
            )

        except Exception as e:
            logger.error(
                "create_chatroom failed (task_id=%s): %s", task_id, e, exc_info=True
            )
            return _make_error(str(e))

    @tool(
        "thenvoi_list_contacts",
        "List agent's contacts with pagination. Returns contacts list and metadata.",
        {
            "room_id": str,
            "page": int,
            "page_size": int,
        },
    )
    async def list_contacts(args: dict[str, Any]) -> dict[str, Any]:
        """List agent's contacts via API."""
        try:
            room_id = args.get("room_id", "")
            page = args.get("page", 1)
            page_size = args.get("page_size", 50)

            logger.debug(
                "[%s] list_contacts: page=%s, page_size=%s", room_id, page, page_size
            )

            tools = _get_tools(room_id)
            result = await tools.list_contacts(page, page_size)

            return _make_result(
                {
                    "status": "success",
                    **result,
                }
            )

        except Exception as e:
            logger.error("list_contacts failed: %s", e, exc_info=True)
            return _make_error(str(e))

    @tool(
        "thenvoi_add_contact",
        "Send a contact request to add someone as a contact. Returns 'pending' when request is created, 'approved' when inverse request existed.",
        {
            "type": "object",
            "properties": {
                "room_id": {"type": "string"},
                "handle": {"type": "string"},
                "message": {
                    "type": "string",
                    "description": "Optional message to include with the contact request",
                },
            },
            "required": ["room_id", "handle"],
        },
    )
    async def add_contact(args: dict[str, Any]) -> dict[str, Any]:
        """Send contact request via API."""
        try:
            room_id = args.get("room_id", "")
            handle = args.get("handle", "")
            message = args.get("message") or None

            logger.info("[%s] add_contact: handle=%s", room_id, handle)

            tools = _get_tools(room_id)
            result = await tools.add_contact(handle, message)

            return _make_result(
                {
                    "status": "success",
                    **result,
                }
            )

        except Exception as e:
            logger.error("add_contact failed: %s", e, exc_info=True)
            return _make_error(str(e))

    @tool(
        "thenvoi_remove_contact",
        "Remove an existing contact by handle or contact ID. Provide either handle or contact_id.",
        {
            "room_id": str,
            "handle": str,
            "contact_id": str,
        },
    )
    async def remove_contact(args: dict[str, Any]) -> dict[str, Any]:
        """Remove contact via API."""
        try:
            room_id = args.get("room_id", "")
            handle = args.get("handle") or None
            contact_id = args.get("contact_id") or None

            if not handle and not contact_id:
                return _make_error("Either handle or contact_id must be provided")

            logger.info(
                "[%s] remove_contact: handle=%s, contact_id=%s",
                room_id,
                handle,
                contact_id,
            )

            tools = _get_tools(room_id)
            result = await tools.remove_contact(handle, contact_id)

            return _make_result(
                {
                    "status": "success",
                    **result,
                }
            )

        except Exception as e:
            logger.error("remove_contact failed: %s", e, exc_info=True)
            return _make_error(str(e))

    @tool(
        "thenvoi_list_contact_requests",
        "List both received and sent contact requests. Received are always pending. Sent can be filtered by status.",
        {
            "room_id": str,
            "page": int,
            "page_size": int,
            "sent_status": str,
        },
    )
    async def list_contact_requests(args: dict[str, Any]) -> dict[str, Any]:
        """List contact requests via API."""
        try:
            room_id = args.get("room_id", "")
            page = args.get("page", 1)
            page_size = args.get("page_size", 50)
            sent_status = args.get("sent_status", "pending")

            logger.debug(
                "[%s] list_contact_requests: page=%s, sent_status=%s",
                room_id,
                page,
                sent_status,
            )

            tools = _get_tools(room_id)
            result = await tools.list_contact_requests(page, page_size, sent_status)

            return _make_result(
                {
                    "status": "success",
                    **result,
                }
            )

        except Exception as e:
            logger.error("list_contact_requests failed: %s", e, exc_info=True)
            return _make_error(str(e))

    @tool(
        "thenvoi_respond_contact_request",
        "Respond to a contact request. Actions: 'approve'/'reject' for received requests, 'cancel' for sent requests. Provide either handle or request_id.",
        {
            "room_id": str,
            "action": str,
            "handle": str,
            "request_id": str,
        },
    )
    async def respond_contact_request(args: dict[str, Any]) -> dict[str, Any]:
        """Respond to contact request via API."""
        try:
            room_id = args.get("room_id", "")
            action = args.get("action", "")
            handle = args.get("handle") or None
            request_id = args.get("request_id") or None

            if not handle and not request_id:
                return _make_error("Either handle or request_id must be provided")

            if action not in ("approve", "reject", "cancel"):
                return _make_error(
                    f"Invalid action '{action}'. Use 'approve', 'reject', or 'cancel'"
                )

            logger.info(
                "[%s] respond_contact_request: action=%s, handle=%s, request_id=%s",
                room_id,
                action,
                handle,
                request_id,
            )

            tools = _get_tools(room_id)
            result = await tools.respond_contact_request(action, handle, request_id)

            return _make_result(
                {
                    "status": "success",
                    **result,
                }
            )

        except Exception as e:
            logger.error("respond_contact_request failed: %s", e, exc_info=True)
            return _make_error(str(e))

    # Create MCP SDK server with all tools
    server = create_sdk_mcp_server(
        name="thenvoi",
        version="1.0.0",
        tools=[
            send_message,
            send_event,
            add_participant,
            remove_participant,
            get_participants,
            lookup_peers,
            create_chatroom,
            list_contacts,
            add_contact,
            remove_contact,
            list_contact_requests,
            respond_contact_request,
        ],
    )

    logger.info("Thenvoi MCP SDK server created with 12 real tools")

    return server


# Tool names as constants (MCP naming convention: mcp__{server}__{tool})
THENVOI_TOOLS = [
    "mcp__thenvoi__thenvoi_send_message",
    "mcp__thenvoi__thenvoi_send_event",
    "mcp__thenvoi__thenvoi_add_participant",
    "mcp__thenvoi__thenvoi_remove_participant",
    "mcp__thenvoi__thenvoi_get_participants",
    "mcp__thenvoi__thenvoi_lookup_peers",
    "mcp__thenvoi__thenvoi_create_chatroom",
    "mcp__thenvoi__thenvoi_list_contacts",
    "mcp__thenvoi__thenvoi_add_contact",
    "mcp__thenvoi__thenvoi_remove_contact",
    "mcp__thenvoi__thenvoi_list_contact_requests",
    "mcp__thenvoi__thenvoi_respond_contact_request",
]
