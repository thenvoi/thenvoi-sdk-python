"""
Convert AgentTools to LangChain StructuredTool format.

This module provides the bridge between the SDK's AgentTools and LangChain's
StructuredTool format for use with LangGraph.
"""

from typing import Any, Literal

from langchain_core.tools import StructuredTool

from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.runtime.tools import get_tool_description


def agent_tools_to_langchain(tools: AgentToolsProtocol) -> list[Any]:
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

    async def create_chatroom_wrapper(task_id: str | None = None) -> str:
        """Create a new chat room."""
        try:
            return await tools.create_chatroom(task_id)
        except Exception as e:
            return f"Error creating chatroom (task_id={task_id}): {e}"

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
            description=get_tool_description("send_message"),
        ),
        StructuredTool.from_function(
            coroutine=add_participant_wrapper,
            name="add_participant",
            description=get_tool_description("add_participant"),
        ),
        StructuredTool.from_function(
            coroutine=remove_participant_wrapper,
            name="remove_participant",
            description=get_tool_description("remove_participant"),
        ),
        StructuredTool.from_function(
            coroutine=lookup_peers_wrapper,
            name="lookup_peers",
            description=get_tool_description("lookup_peers"),
        ),
        StructuredTool.from_function(
            coroutine=get_participants_wrapper,
            name="get_participants",
            description=get_tool_description("get_participants"),
        ),
        StructuredTool.from_function(
            coroutine=create_chatroom_wrapper,
            name="create_chatroom",
            description=get_tool_description("create_chatroom"),
        ),
        StructuredTool.from_function(
            coroutine=send_event_wrapper,
            name="send_event",
            description=get_tool_description("send_event"),
        ),
    ]
