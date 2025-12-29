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

    def _get_participant_names(room_id: str) -> list[str]:
        """Get list of participant names in room."""
        executions = agent.runtime.executions if agent.runtime else {}
        execution = executions.get(room_id)
        participants = execution.participants if execution else []
        return [p.get("name", "") for p in participants if p.get("name")]

    @tool(
        "send_message",
        "Send a message to the Thenvoi chat room. Use this to respond to users. You MUST use this tool to communicate - text responses without this tool won't reach users.",
        {
            "room_id": str,
            "content": str,
            "mentions": str,  # JSON array of participant names, e.g. '["Alice", "Bob"]' or '[]'
        },
    )
    async def send_message(args: dict[str, Any]) -> dict[str, Any]:
        """Send message to chat room via API."""
        try:
            room_id = args.get("room_id", "")
            content = args.get("content", "")
            mentions_str = args.get("mentions", "[]")

            # Parse mentions JSON (names like ["Alice", "Bob"])
            mention_names: list[str] = []
            if mentions_str:
                try:
                    mention_names = (
                        json.loads(mentions_str)
                        if isinstance(mentions_str, str)
                        else mentions_str
                    )
                except json.JSONDecodeError:
                    pass

            logger.info(f"[{room_id}] send_message: {content[:100]}...")

            # Get AgentTools for this room and send message
            tools = _get_tools(room_id)
            try:
                await tools.send_message(content, mention_names)
            except ValueError as e:
                # Mention resolution failed
                available = _get_participant_names(room_id)
                return _make_error(
                    f"{e}. Available participants: {available}. "
                    f"Use exact participant names from the list."
                )

            return _make_result(
                {
                    "status": "success",
                    "message": "Message sent",
                }
            )

        except Exception as e:
            logger.error(f"send_message failed: {e}", exc_info=True)
            return _make_error(str(e))

    @tool(
        "send_event",
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

            logger.debug(f"[{room_id}] send_event: type={message_type}")

            tools = _get_tools(room_id)
            await tools.send_event(content, message_type)

            return _make_result(
                {
                    "status": "success",
                    "message": "Event sent",
                }
            )

        except Exception as e:
            logger.error(f"send_event failed: {e}", exc_info=True)
            return _make_error(str(e))

    @tool(
        "add_participant",
        "Add a participant (user or agent) to the chat room by name. Use lookup_peers first to see available participants.",
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

            logger.info(f"[{room_id}] add_participant: {name} as {role}")

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
                f"[{room_id}] add_participant: runtime={agent.runtime}, executions={list(executions.keys())}, execution={execution}"
            )
            if execution:
                new_participant = {
                    "id": result["id"],
                    "name": result["name"],
                    "type": "Agent",  # Default, could be User
                }
                execution.add_participant(new_participant)
                logger.info(
                    f"[{room_id}] Updated participants cache: added {result['name']}, total={len(execution.participants)}"
                )

            return _make_result(
                {
                    "status": "success",
                    "message": f"Participant '{name}' added as {role}",
                    **result,
                }
            )

        except Exception as e:
            logger.error(f"add_participant failed: {e}", exc_info=True)
            return _make_error(str(e))

    @tool(
        "remove_participant",
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

            logger.info(f"[{room_id}] remove_participant: {name}")

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
            logger.error(f"remove_participant failed: {e}", exc_info=True)
            return _make_error(str(e))

    @tool(
        "get_participants",
        "Get list of participants currently in the chat room.",
        {
            "room_id": str,
        },
    )
    async def get_participants(args: dict[str, Any]) -> dict[str, Any]:
        """Get participants in chat room via API."""
        try:
            room_id = args.get("room_id", "")

            logger.debug(f"[{room_id}] get_participants")

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
            logger.error(f"get_participants failed: {e}", exc_info=True)
            return _make_error(str(e))

    @tool(
        "lookup_peers",
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
                f"[{room_id}] lookup_peers: page={page}, page_size={page_size}"
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
            logger.error(f"lookup_peers failed: {e}", exc_info=True)
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
        ],
    )

    logger.info("Thenvoi MCP SDK server created with 6 real tools")

    return server


# Tool names as constants (MCP naming convention: mcp__{server}__{tool})
THENVOI_TOOLS = [
    "mcp__thenvoi__send_message",
    "mcp__thenvoi__send_event",
    "mcp__thenvoi__add_participant",
    "mcp__thenvoi__remove_participant",
    "mcp__thenvoi__get_participants",
    "mcp__thenvoi__lookup_peers",
]
