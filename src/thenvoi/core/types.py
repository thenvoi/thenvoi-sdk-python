"""
Core types for Thenvoi agent SDK.

This module defines the data structures and type aliases used across
the SDK for agent communication and platform interaction.

KEY DESIGN PRINCIPLE:
    SDK does NOT send messages directly.
    All communication is via AgentTools used by the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal, overload

from pydantic import BaseModel

from .tool_definitions import TOOL_MODELS

if TYPE_CHECKING:
    from anthropic.types import ToolParam

    from .agent import ThenvoiAgent


@dataclass
class AgentConfig:
    """Configuration for ThenvoiAgent coordinator."""

    auto_subscribe_existing_rooms: bool = True


@dataclass
class SessionConfig:
    """Configuration for AgentSession."""

    enable_context_cache: bool = True
    context_cache_ttl_seconds: int = 300
    max_context_messages: int = 100
    max_message_retries: int = 1  # Max attempts per message before permanently failing
    enable_context_hydration: bool = True  # Whether to fetch history from platform API


@dataclass
class PlatformMessage:
    """
    Message from platform (normalized for adapters).

    This is the message format passed to MessageHandlers.
    """

    id: str
    room_id: str
    content: str
    sender_id: str
    sender_type: str  # "User", "Agent", "System"
    sender_name: str | None
    message_type: str
    metadata: dict[str, Any]
    created_at: datetime

    def format_for_llm(self) -> str:
        """
        Format message with sender prefix for LLM consumption.

        Returns string in format: [SENDER_NAME]: message content
        """
        sender = self.sender_name or self.sender_type
        return f"[{sender}]: {self.content}"


@dataclass
class ConversationContext:
    """
    Hydrated context for a room.

    Contains conversation history and participant information
    for context-aware processing.
    """

    room_id: str
    messages: list[dict[str, Any]]
    participants: list[dict[str, Any]]
    hydrated_at: datetime


class AgentTools:
    """
    Tools available to the LLM for platform interaction.

    Bound to a specific room_id. Passed to MessageHandler.
    LLM uses these tools to communicate (SDK never sends directly).

    This class provides:
    - Tool methods (send_message, add_participant, etc.)
    - Schema converters for different LLM frameworks

    Example:
        async def my_handler(msg: PlatformMessage, tools: AgentTools):
            # LLM uses tools to respond
            await tools.send_message("Hello!")

            # Or get tool schemas for your LLM
            anthropic_tools = tools.to_anthropic_tools()
    """

    def __init__(self, room_id: str, coordinator: "ThenvoiAgent"):
        """
        Initialize AgentTools for a specific room.

        Args:
            room_id: The room this tools instance is bound to
            coordinator: The ThenvoiAgent coordinator (for internal API calls)
        """
        self.room_id = room_id
        self._coordinator = coordinator

    # --- Tools for LLM ---

    async def send_message(
        self, content: str, mentions: list[str] | list[dict[str, str]] | None = None
    ) -> dict[str, Any]:
        """
        Send a message to the current room.

        Args:
            content: Message content to send
            mentions: List of participant names (strings). SDK resolves names to IDs.

        Returns:
            Full API response dict with message details (id, content, sender, etc.)

        Raises:
            ValueError: If a mentioned name is not found in participants
        """
        resolved_mentions = self._resolve_mentions(mentions or [])
        return await self._coordinator._send_message_internal(
            self.room_id, content, resolved_mentions
        )

    def _resolve_mentions(
        self, mentions: list[str] | list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Resolve mention names to {id, name} dicts using session participants."""
        session = self._coordinator.active_sessions.get(self.room_id)
        participants = session.participants if session else []

        # Build name -> id lookup
        name_to_id = {p.get("name"): p.get("id") for p in participants}

        resolved = []
        for mention in mentions:
            if isinstance(mention, str):
                name = mention
            else:
                name = mention.get("name", "")
                if mention.get("id"):
                    resolved.append({"id": mention["id"], "name": name})
                    continue

            participant_id = name_to_id.get(name)
            if not participant_id:
                available = list(name_to_id.keys())
                raise ValueError(
                    f"Unknown participant '{name}'. Available: {available}"
                )

            resolved.append({"id": participant_id, "name": name})

        return resolved

    async def send_event(
        self,
        content: str,
        message_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Send an event to the current room.

        Events don't require mentions - use for tool_call, tool_result, error, thought, task.

        Args:
            content: Human-readable event content
            message_type: One of: tool_call, tool_result, thought, error, task
            metadata: Optional structured data for the event

        Returns:
            Full API response dict with event details
        """
        return await self._coordinator._send_event_internal(
            self.room_id, content, message_type, metadata
        )

    async def create_chatroom(self, name: str) -> str:
        """
        Create a new chat room.

        Args:
            name: Name for the new chat room

        Returns:
            Room ID of the created room
        """
        return await self._coordinator._create_chatroom_internal(name)

    async def add_participant(self, name: str, role: str = "member") -> dict[str, Any]:
        """
        Add a participant to the current room by name.

        Args:
            name: Name of the participant (agent or user) to add
            role: Role in room - "owner", "admin", or "member" (default)

        Returns:
            Dict with added participant info (id, name, role, status)
        """
        return await self._coordinator._add_participant_internal(
            self.room_id, name, role
        )

    async def remove_participant(self, name: str) -> dict[str, Any]:
        """
        Remove a participant from the current room by name.

        Args:
            name: Name of the participant to remove

        Returns:
            Dict with removed participant info (id, name, status)
        """
        return await self._coordinator._remove_participant_internal(self.room_id, name)

    async def lookup_peers(self, page: int = 1, page_size: int = 50) -> dict[str, Any]:
        """
        Find available peers (agents and users) on the platform.

        Automatically filters to peers NOT already in the current room.

        Args:
            page: Page number (default 1)
            page_size: Items per page (default 50, max 100)

        Returns:
            Dict with 'peers' list and 'metadata' (page, page_size, total_count, total_pages)
        """
        return await self._coordinator._lookup_peers_internal(
            page=page, page_size=page_size, not_in_chat=self.room_id
        )

    async def get_participants(self) -> list[dict[str, Any]]:
        """
        Get participants in the current room.

        Returns:
            List of participant information dictionaries
        """
        return await self._coordinator._get_participants_internal(self.room_id)

    async def get_context(self) -> ConversationContext:
        """
        Get conversation context for the current room.

        Returns:
            ConversationContext with messages and participants
        """
        return await self._coordinator.get_context(self.room_id)

    # --- Schema converters ---

    @property
    def tool_models(self) -> dict[str, type[BaseModel]]:
        """Get Pydantic models for all tools."""
        return TOOL_MODELS

    @overload
    def get_tool_schemas(self, format: Literal["anthropic"]) -> list[ToolParam]: ...
    @overload
    def get_tool_schemas(self, format: Literal["openai"]) -> list[dict[str, Any]]: ...
    def get_tool_schemas(
        self, format: Literal["openai", "anthropic"]
    ) -> list[dict[str, Any]] | list[ToolParam]:
        """
        Get tool schemas in provider-specific format.

        Args:
            format: Target format - "openai" or "anthropic"

        Returns:
            List of tool definitions in the requested format
        """
        tools: list[Any] = []
        for name, model in TOOL_MODELS.items():
            schema = model.model_json_schema()
            # Remove Pydantic-specific keys
            schema.pop("title", None)

            if format == "openai":
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": model.__doc__ or "",
                            "parameters": schema,
                        },
                    }
                )
            elif format == "anthropic":
                tools.append(
                    {
                        "name": name,
                        "description": model.__doc__ or "",
                        "input_schema": schema,
                    }
                )
        return tools

    async def execute_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Execute a tool call by name with validated arguments.

        This is a convenience method for frameworks that need to
        dispatch tool calls programmatically. Errors are caught and
        returned as strings so the LLM can see them and potentially retry.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool (validated against Pydantic model)

        Returns:
            Tool execution result, or error string if execution failed
        """
        # Validate arguments against Pydantic model
        try:
            if tool_name in TOOL_MODELS:
                model = TOOL_MODELS[tool_name]
                validated = model.model_validate(arguments)
                arguments = validated.model_dump(exclude_none=True)
        except Exception as e:
            return f"Error validating {tool_name} arguments: {e}"

        # Dispatch to tool method
        dispatch = {
            "send_message": lambda: self.send_message(
                arguments["content"], arguments.get("mentions")
            ),
            "send_event": lambda: self.send_event(
                arguments["content"],
                arguments["message_type"],
                arguments.get("metadata"),
            ),
            "add_participant": lambda: self.add_participant(
                arguments["name"], arguments.get("role", "member")
            ),
            "remove_participant": lambda: self.remove_participant(arguments["name"]),
            "lookup_peers": lambda: self.lookup_peers(
                arguments.get("page", 1), arguments.get("page_size", 50)
            ),
            "get_participants": lambda: self.get_participants(),
            "create_chatroom": lambda: self.create_chatroom(arguments["name"]),
        }

        if tool_name not in dispatch:
            return f"Unknown tool: {tool_name}"

        try:
            return await dispatch[tool_name]()
        except Exception as e:
            return f"Error executing {tool_name}: {e}"


# Callback type - receives AgentTools, NOT ThenvoiAgent
MessageHandler = Callable[[PlatformMessage, AgentTools], Awaitable[None]]
