"""
Parlant tool definitions that wrap Thenvoi AgentTools.

These tools are defined at server startup and use a session-keyed registry
to access the current room's tools during execution.

This module provides the same tools as LangGraph/Claude adapters:
- send_message: Send messages to the chat room
- send_event: Send events (thought, error, task)
- add_participant: Add agents/users to the room
- remove_participant: Remove participants
- lookup_peers: Find available agents
- get_participants: List current participants
- create_chatroom: Create new rooms

NOTE: We intentionally do NOT use `from __future__ import annotations` here
because Parlant's @p.tool decorator checks annotation types at runtime.
"""

import logging
import warnings
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Session-keyed registry to hold tools for each session
# This approach works across async contexts (unlike ContextVar)
_session_tools: dict[str, Any] = {}

# Track whether send_message was called for each session
# This helps the adapter know if it needs to forward Parlant's response
_session_message_sent: dict[str, bool] = {}


def set_session_tools(session_id: str, tools: Optional[Any]) -> None:
    """Set the tools for a specific Parlant session."""
    if tools is None:
        _session_tools.pop(session_id, None)
        _session_message_sent.pop(session_id, None)
    else:
        _session_tools[session_id] = tools
        _session_message_sent[session_id] = False
    logger.debug("Set tools for session %s: %s", session_id, tools is not None)


def get_session_tools(session_id: str) -> Optional[Any]:
    """Get the tools for a specific Parlant session."""
    tools = _session_tools.get(session_id)
    logger.debug(
        "Get tools for session_id=%s: found=%s, available_sessions=%s",
        session_id,
        tools is not None,
        list(_session_tools.keys()),
    )
    return tools


def mark_message_sent(session_id: str) -> None:
    """Mark that a message was sent via the send_message tool for this session."""
    _session_message_sent[session_id] = True
    logger.debug("Marked message sent for session %s", session_id)


def was_message_sent(session_id: str) -> bool:
    """Check if a message was sent via the send_message tool for this session."""
    return _session_message_sent.get(session_id, False)


# Keep old API for backwards compatibility (deprecated)
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
    return None  # Always returns None, tools now accessed via session_id


def create_parlant_tools() -> list[Any]:
    """
    Create Parlant tool definitions that wrap Thenvoi tools.

    These tools use context variables to access the current room's
    AgentToolsProtocol during execution.

    Returns:
        List of Parlant ToolEntry objects
    """
    try:
        import parlant.sdk as p
        from parlant.core.tools import ToolContext, ToolResult
    except ImportError:
        logger.warning("Parlant SDK not installed, skipping tool creation")
        return []

    @p.tool
    async def thenvoi_send_message(
        context: ToolContext,
        content: str,
        mentions: str,
    ) -> ToolResult:
        """
        Send a message to the chat room.

        Use this to respond to users or other agents. Messages require @mentions
        to reach users. You MUST use this tool to communicate.

        Args:
            context: Parlant tool context (automatically provided)
            content: The message content to send
            mentions: Comma-separated list of participant names to @mention (e.g., "Alice, Bob")

        Returns:
            Confirmation of message sent or error
        """
        logger.info(
            "[Parlant Tool] send_message called: session=%s, content=%s..., mentions=%s",
            context.session_id,
            content[:50],
            mentions,
        )
        tools = get_session_tools(context.session_id)
        if not tools:
            logger.error(
                "[Parlant Tool] send_message: No tools available for session %s",
                context.session_id,
            )
            return ToolResult(data="Error: No tools available in current context")

        try:
            # Parse mentions from comma-separated string
            mention_list = [m.strip() for m in mentions.split(",") if m.strip()]
            if not mention_list:
                logger.warning("[Parlant Tool] send_message: No mentions provided")
                return ToolResult(data="Error: At least one mention is required")

            logger.info("[Parlant Tool] Sending message to: %s", mention_list)
            await tools.send_message(content, mention_list)
            # Mark that we sent a message via the tool (so adapter doesn't duplicate)
            mark_message_sent(context.session_id)
            logger.info("[Parlant Tool] Message sent successfully via tool")
            return ToolResult(data=f"Message sent to {', '.join(mention_list)}")
        except Exception as e:
            logger.error("[Parlant Tool] Error sending message: %s", e, exc_info=True)
            return ToolResult(data=f"Error sending message: {e}")

    @p.tool
    async def thenvoi_send_event(
        context: ToolContext,
        content: str,
        message_type: str,
    ) -> ToolResult:
        """
        Send an event to the chat room. No mentions required.

        Use this to share your reasoning or report status.

        Args:
            context: Parlant tool context (automatically provided)
            content: Human-readable event content
            message_type: Type of event - 'thought' (share reasoning), 'error' (report problem), or 'task' (report progress)

        Returns:
            Confirmation of event sent or error
        """
        logger.info(
            "[Parlant Tool] send_event called: session=%s, type=%s",
            context.session_id,
            message_type,
        )
        tools = get_session_tools(context.session_id)
        if not tools:
            logger.error(
                "[Parlant Tool] send_event: No tools available for session %s",
                context.session_id,
            )
            return ToolResult(data="Error: No tools available in current context")

        if message_type not in ("thought", "error", "task"):
            return ToolResult(
                data=f"Error: Invalid message_type '{message_type}'. Use 'thought', 'error', or 'task'"
            )

        try:
            await tools.send_event(content, message_type, None)
            logger.info("[Parlant Tool] Event (%s) sent successfully", message_type)
            return ToolResult(data=f"Event ({message_type}) sent successfully")
        except Exception as e:
            logger.error("[Parlant Tool] Error sending event: %s", e, exc_info=True)
            return ToolResult(data=f"Error sending event: {e}")

    @p.tool
    async def thenvoi_add_participant(
        context: ToolContext,
        name: str,
    ) -> ToolResult:
        """
        Invite an agent or user to join this chat room.

        Args:
            context: Parlant tool context (automatically provided)
            name: REQUIRED - The name of the agent to add. Must match exactly from lookup_peers (e.g. "Pirate Captain", "Research Agent", "Weather Assistant")

        Returns:
            Success message or error description

        Example calls:
            add_participant(name="Pirate Captain")
            add_participant(name="Research Agent")
        """
        logger.info(
            "[Parlant Tool] add_participant called: session=%s, name=%s",
            context.session_id,
            name,
        )
        tools = get_session_tools(context.session_id)
        if not tools:
            logger.error(
                "[Parlant Tool] add_participant: No tools available for session %s",
                context.session_id,
            )
            return ToolResult(data="Error: No tools available in current context")

        try:
            result = await tools.add_participant(name, "member")
            status = result.get("status", "added")
            if status == "already_in_room":
                logger.info("[Parlant Tool] '%s' is already in the room", name)
                return ToolResult(
                    data=f"'{name}' is already in the room - no action needed"
                )
            logger.info("[Parlant Tool] Successfully added '%s' to the room", name)
            return ToolResult(data=f"Successfully added '{name}' to the room")
        except Exception as e:
            logger.error(
                "[Parlant Tool] Error adding participant '%s': %s",
                name,
                e,
                exc_info=True,
            )
            return ToolResult(data=f"Error adding participant '{name}': {e}")

    @p.tool
    async def thenvoi_remove_participant(
        context: ToolContext,
        name: str,
    ) -> ToolResult:
        """
        Remove a participant from this chat room.

        Args:
            context: Parlant tool context (automatically provided)
            name: REQUIRED - The name of the participant to remove. Must match exactly from get_participants (e.g. "Pirate Captain", "Research Agent")

        Returns:
            Success message or error description

        Example calls:
            remove_participant(name="Pirate Captain")
            remove_participant(name="Research Agent")
        """
        logger.info(
            "[Parlant Tool] remove_participant called: session=%s, name=%s",
            context.session_id,
            name,
        )
        tools = get_session_tools(context.session_id)
        if not tools:
            logger.error(
                "[Parlant Tool] remove_participant: No tools available for session %s",
                context.session_id,
            )
            return ToolResult(data="Error: No tools available in current context")

        try:
            await tools.remove_participant(name)
            logger.info("[Parlant Tool] Successfully removed '%s' from the room", name)
            return ToolResult(data=f"Successfully removed '{name}' from the room")
        except Exception as e:
            logger.error(
                "[Parlant Tool] Error removing participant '%s': %s",
                name,
                e,
                exc_info=True,
            )
            return ToolResult(data=f"Error removing participant '{name}': {e}")

    @p.tool
    async def thenvoi_lookup_peers(
        context: ToolContext,
    ) -> ToolResult:
        """
        List available peers (agents and users) that can be added to this room.

        Automatically excludes peers already in the room. Use this to find
        specialized agents when you cannot answer a question directly.

        Args:
            context: Parlant tool context (automatically provided)

        Returns:
            List of available agents with their names and descriptions
        """
        logger.info(
            "[Parlant Tool] lookup_peers called: session=%s", context.session_id
        )
        tools = get_session_tools(context.session_id)
        if not tools:
            logger.error(
                "[Parlant Tool] lookup_peers: No tools available for session %s",
                context.session_id,
            )
            return ToolResult(data="Error: No tools available in current context")

        try:
            # Use defaults - pagination rarely needed for agent lookups
            result = await tools.lookup_peers(page=1, page_size=50)
            logger.info("[Parlant Tool] lookup_peers result: %s", result)
            if isinstance(result, dict):
                peers = result.get("peers", [])
                metadata = result.get("metadata", {})
                if not peers:
                    return ToolResult(data="No available agents found")

                lines = [
                    f"Available agents (page {metadata.get('page', 1)} of {metadata.get('total_pages', 1)}):"
                ]
                for peer in peers:
                    name = peer.get("name", "Unknown")
                    desc = peer.get("description", "No description")
                    peer_type = peer.get("type", "Agent")
                    lines.append(f"- {name} ({peer_type}): {desc}")
                return ToolResult(data="\n".join(lines))
            return ToolResult(data=str(result))
        except Exception as e:
            logger.error("[Parlant Tool] Error looking up peers: %s", e, exc_info=True)
            return ToolResult(data=f"Error looking up peers: {e}")

    @p.tool
    async def thenvoi_get_participants(
        context: ToolContext,
    ) -> ToolResult:
        """
        Get the list of all participants currently in the chat room.

        Args:
            context: Parlant tool context (automatically provided)

        Returns:
            List of current participants with their names and types
        """
        logger.info(
            "[Parlant Tool] get_participants called: session=%s", context.session_id
        )
        tools = get_session_tools(context.session_id)
        if not tools:
            logger.error(
                "[Parlant Tool] get_participants: No tools available for session %s",
                context.session_id,
            )
            return ToolResult(data="Error: No tools available in current context")

        try:
            result = await tools.get_participants()
            logger.info("[Parlant Tool] get_participants result: %s", result)
            if isinstance(result, list):
                if not result:
                    return ToolResult(data="No participants in the room")
                lines = ["Current participants:"]
                for participant in result:
                    name = participant.get("name", "Unknown")
                    p_type = participant.get("type", "Unknown")
                    lines.append(f"- {name} ({p_type})")
                return ToolResult(data="\n".join(lines))
            return ToolResult(data=str(result))
        except Exception as e:
            logger.error(
                "[Parlant Tool] Error getting participants: %s", e, exc_info=True
            )
            return ToolResult(data=f"Error getting participants: {e}")

    @p.tool
    async def thenvoi_create_chatroom(
        context: ToolContext,
        task_id: str = "",
    ) -> ToolResult:
        """
        Create a new chat room for a specific task or conversation.

        Args:
            context: Parlant tool context (automatically provided)
            task_id: Optional task ID to associate with the room

        Returns:
            The ID of the newly created room
        """
        logger.info(
            "[Parlant Tool] create_chatroom called: session=%s, task_id=%s",
            context.session_id,
            task_id,
        )
        tools = get_session_tools(context.session_id)
        if not tools:
            logger.error(
                "[Parlant Tool] create_chatroom: No tools available for session %s",
                context.session_id,
            )
            return ToolResult(data="Error: No tools available in current context")

        try:
            result = await tools.create_chatroom(task_id if task_id else None)
            logger.info("[Parlant Tool] Created chatroom: %s", result)
            return ToolResult(data=f"Created new chat room: {result}")
        except Exception as e:
            logger.error("[Parlant Tool] Error creating chatroom: %s", e, exc_info=True)
            return ToolResult(data=f"Error creating chatroom: {e}")

    return [
        thenvoi_send_message,
        thenvoi_send_event,
        thenvoi_add_participant,
        thenvoi_remove_participant,
        thenvoi_lookup_peers,
        thenvoi_get_participants,
        thenvoi_create_chatroom,
    ]
