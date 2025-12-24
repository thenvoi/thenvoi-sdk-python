"""
AgentTools - Tools for LLM platform interaction.

Bound to a room_id. Uses AsyncRestClient directly for API calls.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, Field

from thenvoi.core.protocols import AgentToolsProtocol

if TYPE_CHECKING:
    from anthropic.types import ToolParam

    from thenvoi.client.rest import AsyncRestClient

    from .execution import ExecutionContext

logger = logging.getLogger(__name__)


# --- Tool input models (single source of truth for schemas) ---


class SendMessageInput(BaseModel):
    """Send a message to the chat room. Use this to respond to users or other agents."""

    content: str = Field(..., description="The message content to send")
    mentions: list[str] = Field(
        ...,
        min_length=1,
        description="List of participant names to @mention. At least one required.",
    )


class SendEventInput(BaseModel):
    """Send an event to the chat room. Use for thoughts, errors, or task updates."""

    content: str = Field(..., description="Human-readable event content")
    message_type: Literal["thought", "error", "task"] = Field(
        ..., description="Type of event"
    )
    metadata: dict[str, Any] | None = Field(
        None, description="Optional structured data for the event"
    )


class AddParticipantInput(BaseModel):
    """Add a participant (agent or user) to the chat room."""

    name: str = Field(
        ...,
        description="Name of participant to add (must match a name from lookup_peers)",
    )
    role: Literal["owner", "admin", "member"] = Field(
        "member", description="Role for the participant in this room"
    )


class RemoveParticipantInput(BaseModel):
    """Remove a participant from the chat room by name."""

    name: str = Field(..., description="Name of the participant to remove")


class LookupPeersInput(BaseModel):
    """List available peers (agents and users) that can be added to this room."""

    page: int = Field(1, description="Page number")
    page_size: int = Field(50, le=100, description="Items per page (max 100)")


class GetParticipantsInput(BaseModel):
    """Get a list of all participants in the current chat room."""

    pass  # No parameters required


class CreateChatroomInput(BaseModel):
    """Create a new chat room for a specific task or conversation."""

    name: str = Field(..., description="Name for the new chat room")


# Registry mapping tool names to their input models
TOOL_MODELS: dict[str, type[BaseModel]] = {
    "send_message": SendMessageInput,
    "send_event": SendEventInput,
    "add_participant": AddParticipantInput,
    "remove_participant": RemoveParticipantInput,
    "lookup_peers": LookupPeersInput,
    "get_participants": GetParticipantsInput,
    "create_chatroom": CreateChatroomInput,
}


class AgentTools(AgentToolsProtocol):
    """
    Tools for LLM platform interaction.

    Uses AsyncRestClient directly for API calls.
    Bound to a specific room_id. Passed to execution handlers.

    This class provides:
    - Tool methods (send_message, add_participant, etc.)
    - Schema converters for different LLM frameworks
    - execute_tool_call() for programmatic dispatch

    Example (from ExecutionContext):
        tools = AgentTools.from_context(ctx)
        await tools.send_message("Hello!", mentions=["User"])

    Example (manual construction):
        tools = AgentTools(room_id, rest_client, participants=[...])
        schemas = tools.get_tool_schemas("anthropic")
    """

    def __init__(
        self,
        room_id: str,
        rest: "AsyncRestClient",
        participants: list[dict[str, Any]] | None = None,
    ):
        """
        Initialize AgentTools for a specific room.

        Args:
            room_id: The room this tools instance is bound to
            rest: AsyncRestClient for API calls
            participants: Optional list of participants for mention resolution
        """
        self.room_id = room_id
        self.rest = rest
        self._participants = participants or []

    @classmethod
    def from_context(cls, ctx: "ExecutionContext") -> "AgentTools":
        """
        Create AgentTools from an ExecutionContext.

        Convenience method for SDK-heavy users.

        Args:
            ctx: ExecutionContext to create tools from

        Returns:
            AgentTools instance bound to the context's room
        """
        return cls(ctx.room_id, ctx.link.rest, ctx.participants)

    # --- Tool methods ---

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
        from thenvoi.client.rest import (
            ChatMessageRequest,
            ChatMessageRequestMentionsItem,
        )

        resolved_mentions = self._resolve_mentions(mentions or [])
        logger.debug(f"Sending message to room {self.room_id}")

        # Convert to API format
        mention_items = [
            ChatMessageRequestMentionsItem(id=m["id"], name=m["name"])
            for m in resolved_mentions
        ]

        response = await self.rest.agent_api.create_agent_chat_message(
            chat_id=self.room_id,
            message=ChatMessageRequest(content=content, mentions=mention_items),
        )
        if not response.data:
            raise RuntimeError("Failed to send message - no response data")
        return response.data.model_dump()

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
        from thenvoi.client.rest import ChatEventRequest

        logger.debug(f"Sending {message_type} event to room {self.room_id}")

        response = await self.rest.agent_api.create_agent_chat_event(
            chat_id=self.room_id,
            event=ChatEventRequest(
                content=content,
                message_type=message_type,
                metadata=metadata,
            ),
        )
        if not response.data:
            raise RuntimeError("Failed to send event - no response data")
        return response.data.model_dump()

    async def create_chatroom(self, name: str) -> str:
        """
        Create a new chat room.

        Args:
            name: Name for the new chat room

        Returns:
            Room ID of the created room
        """
        logger.debug(f"Creating chatroom: {name}")
        # Note: This would need the actual API method
        raise NotImplementedError("create_chatroom not yet implemented in REST client")

    async def add_participant(self, name: str, role: str = "member") -> dict[str, Any]:
        """
        Add a participant to the current room by name.

        Args:
            name: Name of the participant (agent or user) to add
            role: Role in room - "owner", "admin", or "member" (default)

        Returns:
            Dict with added participant info (id, name, role, status)

        Raises:
            ValueError: If participant not found by name
        """
        from thenvoi.client.rest import ParticipantRequest

        logger.debug(
            f"Adding participant '{name}' with role '{role}' to room {self.room_id}"
        )

        # Look up participant ID by name (paginates through all peers)
        participant = await self._lookup_peer_by_name(name)
        if not participant:
            raise ValueError(
                f"Participant '{name}' not found. Use lookup_peers to find available peers."
            )

        participant_id = participant["id"]
        logger.debug(f"Resolved '{name}' to ID: {participant_id}")

        await self.rest.agent_api.add_agent_chat_participant(
            chat_id=self.room_id,
            participant=ParticipantRequest(participant_id=participant_id, role=role),
        )

        # Update internal participant cache for immediate mention resolution
        # NOTE: WebSocket will eventually deliver participant_added event, but this
        # allows @mentions to work immediately after add_participant returns.
        new_participant = {
            "id": participant_id,
            "name": name,
            "type": participant.get("type", "Agent"),
        }
        self._participants.append(new_participant)
        logger.debug(
            f"Updated participant cache: added {name}, total={len(self._participants)}"
        )

        return {
            "id": participant_id,
            "name": name,
            "role": role,
            "status": "added",
        }

    async def remove_participant(self, name: str) -> dict[str, Any]:
        """
        Remove a participant from the current room by name.

        Args:
            name: Name of the participant to remove

        Returns:
            Dict with removed participant info (id, name, status)

        Raises:
            ValueError: If participant not found in room
        """
        logger.debug(f"Removing participant '{name}' from room {self.room_id}")

        # Look up participant ID by name from current room participants
        participants = await self.get_participants()
        participant = None
        for p in participants:
            if p.get("name", "").lower() == name.lower():
                participant = p
                break

        if not participant:
            raise ValueError(f"Participant '{name}' not found in this room.")

        participant_id = participant["id"]
        logger.debug(f"Resolved '{name}' to ID: {participant_id}")

        await self.rest.agent_api.remove_agent_chat_participant(
            self.room_id,
            participant_id,
        )

        # Update internal participant cache
        # NOTE: WebSocket will eventually deliver participant_removed event, but this
        # prevents @mentions to the removed participant immediately after removal.
        self._participants = [
            p for p in self._participants if p.get("id") != participant_id
        ]
        logger.debug(
            f"Updated participant cache: removed {name}, total={len(self._participants)}"
        )

        return {
            "id": participant_id,
            "name": name,
            "status": "removed",
        }

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
        logger.debug(f"Looking up peers: page={page}, page_size={page_size}")
        response = await self.rest.agent_api.list_agent_peers(
            page=page,
            page_size=page_size,
            not_in_chat=self.room_id,
        )

        peers = []
        if response.data:
            peers = [
                {
                    "id": peer.id,
                    "name": peer.name,
                    "type": getattr(peer, "type", "Agent"),
                    "description": peer.description,
                }
                for peer in response.data
            ]

        metadata = {
            "page": response.metadata.page if response.metadata else page,
            "page_size": response.metadata.page_size
            if response.metadata
            else page_size,
            "total_count": response.metadata.total_count
            if response.metadata
            else len(peers),
            "total_pages": response.metadata.total_pages if response.metadata else 1,
        }

        return {"peers": peers, "metadata": metadata}

    async def get_participants(self) -> list[dict[str, Any]]:
        """
        Get participants in the current room.

        Returns:
            List of participant information dictionaries
        """
        logger.debug(f"Getting participants for room {self.room_id}")
        response = await self.rest.agent_api.list_agent_chat_participants(
            chat_id=self.room_id,
        )
        if not response.data:
            return []

        return [
            {
                "id": p.id,
                "name": p.name,
                "type": p.type,
            }
            for p in response.data
        ]

    # --- Mention resolution ---

    def _resolve_mentions(
        self, mentions: list[str] | list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """
        Resolve mention names to {id, name} dicts using cached participants.

        Args:
            mentions: List of names (strings) or already-resolved dicts

        Returns:
            List of {id, name} dicts

        Raises:
            ValueError: If a name is not found in participants
        """
        # Build name -> id lookup from cached participants
        name_to_id = {p.get("name"): p.get("id") for p in self._participants}

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

    async def _lookup_peer_by_name(self, name: str) -> dict[str, Any] | None:
        """
        Find a peer by name, paginating through all results.

        Args:
            name: Name to search for (case-insensitive)

        Returns:
            Peer dict if found, None otherwise
        """
        page = 1
        while True:
            result = await self.lookup_peers(page=page, page_size=100)
            for peer in result["peers"]:
                if peer.get("name", "").lower() == name.lower():
                    return peer

            # Check if more pages
            metadata = result["metadata"]
            if page >= metadata.get("total_pages", 1):
                break
            page += 1

        return None

    # --- Schema converters ---

    @property
    def tool_models(self) -> dict[str, type[BaseModel]]:
        """Get Pydantic models for all tools."""
        return TOOL_MODELS

    def get_tool_schemas(self, format: str) -> list[dict[str, Any]] | list["ToolParam"]:
        """
        Get tool schemas in provider-specific format.

        Args:
            format: Target format - "openai" or "anthropic"

        Returns:
            List of tool definitions in the requested format

        Raises:
            ValueError: If format is not "openai" or "anthropic"
        """
        if format not in ("openai", "anthropic"):
            raise ValueError(
                f"Invalid format: {format}. Must be 'openai' or 'anthropic'"
            )

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

    def get_anthropic_tool_schemas(self) -> list["ToolParam"]:
        """Get tool schemas in Anthropic format (strongly typed)."""
        return cast(list["ToolParam"], self.get_tool_schemas("anthropic"))

    def get_openai_tool_schemas(self) -> list[dict[str, Any]]:
        """Get tool schemas in OpenAI format (strongly typed)."""
        return cast(list[dict[str, Any]], self.get_tool_schemas("openai"))

    async def execute_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Execute a tool call by name with validated arguments.

        Convenience method for frameworks that need to dispatch tool calls
        programmatically. Errors are caught and returned as strings so the
        LLM can see them and potentially retry.

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
