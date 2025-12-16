"""
Convert AgentTools to LangChain StructuredTool format.

This module provides the bridge between the SDK's AgentTools and LangChain's
StructuredTool format for use with LangGraph.
"""

from typing import Any, Literal

from langchain_core.tools import StructuredTool

from thenvoi.runtime.tools import AgentTools


def agent_tools_to_langchain(tools: AgentTools) -> list[Any]:
    """
    Convert AgentTools to LangChain StructuredTool instances.

    Args:
        tools: AgentTools instance bound to a room

    Returns:
        List of LangChain StructuredTool instances
    """

    # Create wrapper functions that capture the tools instance
    # All wrappers catch exceptions and return error strings so LLM can see failures
    async def send_message_wrapper(
        content: str, mentions: list[str]
    ) -> dict[str, Any] | str:
        """Send a message to the chat room. Provide participant names in mentions."""
        try:
            return await tools.send_message(content, mentions)
        except Exception as e:
            return f"Error sending message: {e}"

    async def add_participant_wrapper(
        name: str, role: str = "member"
    ) -> dict[str, Any] | str:
        """Add a participant (agent or user) to the chat room by name. Use lookup_peers first to find available agents."""
        try:
            return await tools.add_participant(name, role)
        except Exception as e:
            return f"Error adding participant '{name}': {e}"

    async def remove_participant_wrapper(name: str) -> dict[str, Any] | str:
        """Remove a participant from the chat room by name."""
        try:
            return await tools.remove_participant(name)
        except Exception as e:
            return f"Error removing participant '{name}': {e}"

    async def lookup_peers_wrapper(
        page: int = 1, page_size: int = 50
    ) -> dict[str, Any] | str:
        """List available peers (agents and users) on the platform. Returns paginated results with metadata."""
        try:
            return await tools.lookup_peers(page, page_size)
        except Exception as e:
            return f"Error looking up peers: {e}"

    async def get_participants_wrapper() -> list[dict[str, Any]] | str:
        """Get participants in the chat room."""
        try:
            return await tools.get_participants()
        except Exception as e:
            return f"Error getting participants: {e}"

    async def create_chatroom_wrapper(name: str) -> str:
        """Create a new chat room."""
        try:
            return await tools.create_chatroom(name)
        except Exception as e:
            return f"Error creating chatroom '{name}': {e}"

    async def send_event_wrapper(
        content: str,
        message_type: Literal["thought", "error", "task"],
    ) -> dict[str, Any] | str:
        """Send an event to the chat room. No mentions required.

        message_type options:
        - 'thought': Share your reasoning or plan BEFORE taking actions. Explain what you're about to do and why.
        - 'error': Report an error or problem that occurred.
        - 'task': Report task progress or completion status.

        Always send a thought before complex actions to keep users informed.
        """
        try:
            return await tools.send_event(content, message_type, None)
        except Exception as e:
            return f"Error sending event: {e}"

    return [
        StructuredTool.from_function(
            coroutine=send_message_wrapper,
            name="send_message",
            description="Send a message to the chat room. Provide participant names in mentions array.",
        ),
        StructuredTool.from_function(
            coroutine=add_participant_wrapper,
            name="add_participant",
            description="Add a participant (agent or user) to the chat room by name. Use lookup_peers first to find available agents. Provide name and optionally role (owner/admin/member, default: member).",
        ),
        StructuredTool.from_function(
            coroutine=remove_participant_wrapper,
            name="remove_participant",
            description="Remove a participant from the chat room by name.",
        ),
        StructuredTool.from_function(
            coroutine=lookup_peers_wrapper,
            name="lookup_peers",
            description="List available peers (agents and users) that can be added to this room. Automatically excludes peers already in the room. Supports pagination with page and page_size parameters. Returns dict with 'peers' list and 'metadata' (page, page_size, total_count, total_pages).",
        ),
        StructuredTool.from_function(
            coroutine=get_participants_wrapper,
            name="get_participants",
            description="Get a list of all participants in the current chat room.",
        ),
        StructuredTool.from_function(
            coroutine=create_chatroom_wrapper,
            name="create_chatroom",
            description="Create a new chat room for a specific task or conversation.",
        ),
        StructuredTool.from_function(
            coroutine=send_event_wrapper,
            name="send_event",
            description="Send an event. Use 'thought' to share reasoning BEFORE actions, 'error' for problems, 'task' for progress updates. Always send a thought before complex actions.",
        ),
    ]
