"""
AgentTools - Tools for LLM platform interaction.

Bound to a room_id. Uses AsyncRestClient directly for API calls.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, Field, ValidationError

from thenvoi.client.rest import ChatRoomRequest, DEFAULT_REQUEST_OPTIONS
from thenvoi.core.protocols import AgentToolsProtocol

if TYPE_CHECKING:
    from anthropic.types import ToolParam

    from thenvoi.client.rest import AsyncRestClient

    from .execution import ExecutionContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolDefinition:
    """Metadata for a built-in Thenvoi tool."""

    name: str
    input_model: type[BaseModel]
    method_name: str


# --- Tool input models (single source of truth for schemas) ---


class SendMessageInput(BaseModel):
    """Send a message to the chat room.

    Use this to respond to users or other agents. Messages require at least one @mention
    in the mentions array. You MUST use this tool to communicate - plain text responses
    won't reach users.
    """

    content: str = Field(..., description="The message content to send")
    mentions: list[str] = Field(
        ...,
        description=(
            "List of participant handles to @mention. At least one required. "
            "For users: @<username> (e.g., '@john'). "
            "For agents: @<username>/<agent-name> (e.g., '@john/weather-agent')."
        ),
    )


class SendEventInput(BaseModel):
    """Send an event to the chat room. No mentions required.

    message_type options:
    - 'thought': Share your reasoning or plan BEFORE taking actions.
      Explain what you're about to do and why.
    - 'error': Report an error or problem that occurred.
    - 'task': Report task progress or completion status.

    Always send a thought before complex actions to keep users informed.
    """

    content: str = Field(..., description="Human-readable event content")
    message_type: Literal["thought", "error", "task"] = Field(
        ..., description="Type of event"
    )
    metadata: dict[str, Any] | None = Field(
        None, description="Optional structured data for the event"
    )


class AddParticipantInput(BaseModel):
    """Add a participant (agent or user) to the chat room by name.

    IMPORTANT: Use thenvoi_lookup_peers() first to find available agents.
    """

    name: str = Field(
        ...,
        description="Name of participant to add (must match a name from thenvoi_lookup_peers)",
    )
    role: Literal["owner", "admin", "member"] = Field(
        "member", description="Role for the participant in this room"
    )


class RemoveParticipantInput(BaseModel):
    """Remove a participant from the chat room by name."""

    name: str = Field(..., description="Name of the participant to remove")


class LookupPeersInput(BaseModel):
    """List available peers (agents and users) that can be added to this room.

    Automatically excludes peers already in the room.
    Returns dict with 'peers' list and 'metadata' (page, page_size, total_count, total_pages).
    Use this to find specialized agents (e.g., Weather Agent) when you cannot answer
    a question directly.
    """

    page: int = Field(1, description="Page number")
    page_size: int = Field(50, le=100, description="Items per page (max 100)")


class GetParticipantsInput(BaseModel):
    """Get a list of all participants in the current chat room."""

    pass  # No parameters required


class CreateChatroomInput(BaseModel):
    """Create a new chat room for a specific task or conversation."""

    task_id: str | None = Field(
        default=None, description="Associated task ID (optional)"
    )


class ListContactsInput(BaseModel):
    """List agent's contacts with pagination."""

    page: int = Field(1, description="Page number", ge=1)
    page_size: int = Field(50, description="Items per page", ge=1, le=100)


class AddContactInput(BaseModel):
    """Send a contact request to add someone as a contact.

    Returns 'pending' when request is created.
    Returns 'approved' when inverse request existed and was auto-accepted.
    """

    handle: str = Field(
        ...,
        description="Handle of user/agent to add (e.g., '@john' or '@john/agent-name')",
    )
    message: str | None = Field(None, description="Optional message with the request")


class RemoveContactInput(BaseModel):
    """Remove an existing contact by handle or ID."""

    handle: str | None = Field(None, description="Contact's handle")
    contact_id: str | None = Field(None, description="Or contact record ID (UUID)")


class ListContactRequestsInput(BaseModel):
    """List both received and sent contact requests.

    Received requests are always filtered to pending status.
    Sent requests can be filtered by status.
    """

    page: int = Field(1, description="Page number", ge=1)
    page_size: int = Field(
        50, description="Items per page per direction (max 100)", ge=1, le=100
    )
    sent_status: Literal["pending", "approved", "rejected", "cancelled", "all"] = Field(
        "pending", description="Filter sent requests by status"
    )


class RespondContactRequestInput(BaseModel):
    """Respond to a contact request.

    Actions:
    - 'approve'/'reject': For requests you RECEIVED (handle = requester's handle)
    - 'cancel': For requests you SENT (handle = recipient's handle)
    """

    action: Literal["approve", "reject", "cancel"] = Field(
        ..., description="Action to take"
    )
    handle: str | None = Field(None, description="Other party's handle")
    request_id: str | None = Field(None, description="Or request ID (UUID)")


class ListMemoriesInput(BaseModel):
    """List memories accessible to the agent.

    Returns memories about the specified subject (cross-agent sharing)
    and organization-wide shared memories.
    """

    subject_id: str | None = Field(
        None, description="Filter by subject UUID (required for subject-scoped queries)"
    )
    scope: Literal["subject", "organization", "all"] | None = Field(
        None, description="Filter by scope"
    )
    system: Literal["sensory", "working", "long_term"] | None = Field(
        None, description="Filter by memory system"
    )
    type: (
        Literal["iconic", "echoic", "haptic", "episodic", "semantic", "procedural"]
        | None
    ) = Field(None, description="Filter by memory type")
    segment: Literal["user", "agent", "tool", "guideline"] | None = Field(
        None, description="Filter by segment"
    )
    content_query: str | None = Field(None, description="Full-text search query")
    page_size: int = Field(50, description="Number of results per page", ge=1, le=50)
    status: Literal["active", "superseded", "archived", "all"] | None = Field(
        None, description="Filter by status"
    )


class StoreMemoryInput(BaseModel):
    """Store a new memory entry.

    The memory will be associated with the authenticated agent as the source.
    For subject-scoped memories, provide a subject_id.
    For organization-scoped memories, omit subject_id.
    """

    content: str = Field(..., description="The memory content")
    system: Literal["sensory", "working", "long_term"] = Field(
        ..., description="Memory system tier"
    )
    type: Literal[
        "iconic", "echoic", "haptic", "episodic", "semantic", "procedural"
    ] = Field(..., description="Memory type (must be valid for selected system)")
    segment: Literal["user", "agent", "tool", "guideline"] = Field(
        ..., description="Logical segment"
    )
    thought: str = Field(..., description="Agent's reasoning for storing this memory")
    scope: Literal["subject", "organization"] = Field(
        "subject", description="Visibility scope"
    )
    subject_id: str | None = Field(
        None,
        description="UUID of the subject this memory is about (required for subject scope)",
    )
    metadata: dict[str, Any] | None = Field(
        None, description="Additional metadata (tags, references)"
    )


class GetMemoryInput(BaseModel):
    """Retrieve a specific memory by ID."""

    memory_id: str = Field(..., description="Memory ID (UUID)")


class SupersedeMemoryInput(BaseModel):
    """Mark a memory as superseded (soft delete).

    Use when information is outdated or incorrect.
    The memory remains for audit trail but won't appear in normal queries.
    Only the source agent can supersede.
    """

    memory_id: str = Field(..., description="Memory ID (UUID)")


class ArchiveMemoryInput(BaseModel):
    """Archive a memory (hide but preserve).

    Use when memory is valid but not currently needed.
    Archived memories can be restored later by humans.
    Only the source agent can archive.
    """

    memory_id: str = Field(..., description="Memory ID (UUID)")


# Registry mapping tool names to their schemas and bound AgentTools methods.
TOOL_DEFINITIONS: dict[str, ToolDefinition] = {
    "thenvoi_send_message": ToolDefinition(
        name="thenvoi_send_message",
        input_model=SendMessageInput,
        method_name="send_message",
    ),
    "thenvoi_send_event": ToolDefinition(
        name="thenvoi_send_event",
        input_model=SendEventInput,
        method_name="send_event",
    ),
    "thenvoi_add_participant": ToolDefinition(
        name="thenvoi_add_participant",
        input_model=AddParticipantInput,
        method_name="add_participant",
    ),
    "thenvoi_remove_participant": ToolDefinition(
        name="thenvoi_remove_participant",
        input_model=RemoveParticipantInput,
        method_name="remove_participant",
    ),
    "thenvoi_lookup_peers": ToolDefinition(
        name="thenvoi_lookup_peers",
        input_model=LookupPeersInput,
        method_name="lookup_peers",
    ),
    "thenvoi_get_participants": ToolDefinition(
        name="thenvoi_get_participants",
        input_model=GetParticipantsInput,
        method_name="get_participants",
    ),
    "thenvoi_create_chatroom": ToolDefinition(
        name="thenvoi_create_chatroom",
        input_model=CreateChatroomInput,
        method_name="create_chatroom",
    ),
    "thenvoi_list_contacts": ToolDefinition(
        name="thenvoi_list_contacts",
        input_model=ListContactsInput,
        method_name="list_contacts",
    ),
    "thenvoi_add_contact": ToolDefinition(
        name="thenvoi_add_contact",
        input_model=AddContactInput,
        method_name="add_contact",
    ),
    "thenvoi_remove_contact": ToolDefinition(
        name="thenvoi_remove_contact",
        input_model=RemoveContactInput,
        method_name="remove_contact",
    ),
    "thenvoi_list_contact_requests": ToolDefinition(
        name="thenvoi_list_contact_requests",
        input_model=ListContactRequestsInput,
        method_name="list_contact_requests",
    ),
    "thenvoi_respond_contact_request": ToolDefinition(
        name="thenvoi_respond_contact_request",
        input_model=RespondContactRequestInput,
        method_name="respond_contact_request",
    ),
    "thenvoi_list_memories": ToolDefinition(
        name="thenvoi_list_memories",
        input_model=ListMemoriesInput,
        method_name="list_memories",
    ),
    "thenvoi_store_memory": ToolDefinition(
        name="thenvoi_store_memory",
        input_model=StoreMemoryInput,
        method_name="store_memory",
    ),
    "thenvoi_get_memory": ToolDefinition(
        name="thenvoi_get_memory",
        input_model=GetMemoryInput,
        method_name="get_memory",
    ),
    "thenvoi_supersede_memory": ToolDefinition(
        name="thenvoi_supersede_memory",
        input_model=SupersedeMemoryInput,
        method_name="supersede_memory",
    ),
    "thenvoi_archive_memory": ToolDefinition(
        name="thenvoi_archive_memory",
        input_model=ArchiveMemoryInput,
        method_name="archive_memory",
    ),
}

TOOL_MODELS: dict[str, type[BaseModel]] = {
    name: definition.input_model for name, definition in TOOL_DEFINITIONS.items()
}

# Memory tools - optional, only available for enterprise customers.
# Explicitly listed (not derived by heuristic) because memory is an opt-in
# enterprise feature and accidental inclusion of a non-memory tool would
# expose functionality that should be gated.
MEMORY_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "thenvoi_list_memories",
        "thenvoi_store_memory",
        "thenvoi_get_memory",
        "thenvoi_supersede_memory",
        "thenvoi_archive_memory",
    }
)

# Contact tools - explicitly listed (not derived by heuristic) because a
# future tool whose name happens to contain "contact" (e.g.
# thenvoi_get_contact_context) would be silently misclassified.
CONTACT_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "thenvoi_list_contacts",
        "thenvoi_add_contact",
        "thenvoi_remove_contact",
        "thenvoi_list_contact_requests",
        "thenvoi_respond_contact_request",
    }
)

# Derived from TOOL_MODELS — single source of truth
ALL_TOOL_NAMES: frozenset[str] = frozenset(TOOL_MODELS.keys())

# Fail fast on typos — catch at import time, not in a test run.
# Use explicit checks instead of ``assert`` so they are not stripped by -O.
if MEMORY_TOOL_NAMES - ALL_TOOL_NAMES:
    raise ValueError(f"Unknown memory tools: {MEMORY_TOOL_NAMES - ALL_TOOL_NAMES}")
if CONTACT_TOOL_NAMES - ALL_TOOL_NAMES:
    raise ValueError(f"Unknown contact tools: {CONTACT_TOOL_NAMES - ALL_TOOL_NAMES}")

BASE_TOOL_NAMES: frozenset[str] = ALL_TOOL_NAMES - MEMORY_TOOL_NAMES
CHAT_TOOL_NAMES: frozenset[str] = BASE_TOOL_NAMES - CONTACT_TOOL_NAMES
MCP_TOOL_PREFIX: str = "mcp__thenvoi__"


def mcp_tool_names(names: frozenset[str]) -> list[str]:
    """Convert base tool names to MCP-prefixed names for Claude SDK.

    Returns a sorted list for deterministic ordering across runs.
    """
    return [f"{MCP_TOOL_PREFIX}{name}" for name in sorted(names)]


def get_tool_description(name: str) -> str:
    """
    Get the LLM-optimized description for a tool.

    Use this to get consistent tool descriptions across all adapters.
    Descriptions are sourced from the Pydantic model docstrings.

    Args:
        name: Tool name (e.g., "thenvoi_send_message", "thenvoi_lookup_peers")
              Also accepts unprefixed names for backwards compatibility (deprecated).

    Returns:
        Tool description string
    """
    # Try exact match first
    model = TOOL_MODELS.get(name)
    if model and model.__doc__:
        return model.__doc__

    # Try with prefix for backwards compatibility (deprecated)
    if not name.startswith("thenvoi_"):
        prefixed_name = f"thenvoi_{name}"
        model = TOOL_MODELS.get(prefixed_name)
        if model and model.__doc__:
            warnings.warn(
                f"Tool name '{name}' is deprecated. Use '{prefixed_name}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return model.__doc__

    return f"Execute {name}"


def iter_tool_definitions(*, include_memory: bool = False) -> list[ToolDefinition]:
    """Return built-in tool definitions with optional memory tool inclusion."""
    definitions = list(TOOL_DEFINITIONS.values())
    if include_memory:
        return definitions

    return [
        definition
        for definition in definitions
        if definition.name not in MEMORY_TOOL_NAMES
    ]


def format_tool_validation_error(tool_name: str, error: ValidationError) -> str:
    """Format Pydantic validation errors for LLM-readable tool feedback."""
    errors = [
        f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
        for err in error.errors()
    ]
    return f"Invalid arguments for {tool_name}: {', '.join(errors)}"


def validate_tool_arguments(
    tool_name: str,
    input_model: type[BaseModel],
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Validate tool arguments and return a normalized kwargs dictionary."""
    try:
        validated = input_model.model_validate(arguments)
    except ValidationError as error:
        raise ValueError(format_tool_validation_error(tool_name, error)) from error

    return validated.model_dump(exclude_none=True)


class AgentTools(AgentToolsProtocol):
    """
    Room-bound tools for LLM platform interaction.

    Uses AsyncRestClient directly for API calls.
    Bound to a specific room_id. Passed to execution handlers.

    This class provides:
    - Tool methods (send_message, add_participant, etc.)
    - Contact management methods (list_contacts, add_contact, etc.)
    - Schema converters for different LLM frameworks
    - execute_tool_call() for programmatic dispatch

    Note: AgentTools vs ContactTools
        - AgentTools: Room-bound. Used by LLM agents in chat rooms.
          Has full tool suite including messaging, participants, AND contacts.
        - ContactTools: Agent-level. Used by ContactEventHandler for
          programmatic contact handling in CALLBACK strategy. Contact-only.

    Example (from ExecutionContext):
        tools = AgentTools.from_context(ctx)
        await tools.send_message("Hello!", mentions=["@john"])

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

    @property
    def participants(self) -> list[dict[str, Any]]:
        """Return a shallow copy of the cached participant list."""
        return list(self._participants)

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
            mentions: List of participant handles (strings). SDK resolves handles to IDs.
                      Format: @<username> for users, @<username>/<agent-name> for agents.

        Returns:
            Full API response dict with message details (id, content, sender, etc.)

        Raises:
            ValueError: If a mentioned handle is not found in participants
        """
        from thenvoi.client.rest import (
            ChatMessageRequest,
            ChatMessageRequestMentionsItem,
        )

        resolved_mentions = self._resolve_mentions(mentions or [])

        # Validate mentions are not empty — API requires ≥1 mention.
        # Return a helpful error so the LLM can retry with proper mentions.
        if not resolved_mentions:
            participant_names = [
                p.get("handle") or p["name"] for p in self._participants
            ]
            return {
                "error": (
                    "At least one mention is required. "
                    f"Available participants: {participant_names}. "
                    "Please retry with mentions specifying who this message is for."
                )
            }

        logger.debug("Sending message to room %s", self.room_id)

        # Convert to API format - use handle (not name) for mentions
        mention_items = [
            ChatMessageRequestMentionsItem(id=m["id"], handle=m["handle"])
            for m in resolved_mentions
        ]

        response = await self.rest.agent_api_messages.create_agent_chat_message(
            chat_id=self.room_id,
            message=ChatMessageRequest(content=content, mentions=mention_items),
            request_options=DEFAULT_REQUEST_OPTIONS,
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

        logger.debug("Sending %s event to room %s", message_type, self.room_id)

        response = await self.rest.agent_api_events.create_agent_chat_event(
            chat_id=self.room_id,
            event=ChatEventRequest(
                content=content,
                message_type=message_type,
                metadata=metadata,
            ),
            request_options=DEFAULT_REQUEST_OPTIONS,
        )
        if not response.data:
            raise RuntimeError("Failed to send event - no response data")
        return response.data.model_dump()

    async def create_chatroom(self, task_id: str | None = None) -> str:
        """
        Create a new chat room.

        Args:
            task_id: Associated task ID (optional)

        Returns:
            Room ID of the created room
        """
        logger.debug("Creating chatroom with task_id=%s", task_id)
        response = await self.rest.agent_api_chats.create_agent_chat(
            chat=ChatRoomRequest(task_id=task_id),
            request_options=DEFAULT_REQUEST_OPTIONS,
        )
        return response.data.id

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
            "Adding participant '%s' with role '%s' to room %s",
            name,
            role,
            self.room_id,
        )

        # First check if participant is already in the room
        current_participants = await self.get_participants()
        for p in current_participants:
            if p.get("name", "").lower() == name.lower():
                logger.debug("Participant '%s' is already in the room", name)
                return {
                    "id": p["id"],
                    "name": p["name"],
                    "role": role,
                    "status": "already_in_room",
                }

        # Look up participant ID by name (paginates through all peers)
        participant = await self._lookup_peer_by_name(name)
        if not participant:
            raise ValueError(
                f"Participant '{name}' not found. Use thenvoi_lookup_peers to find available peers."
            )

        participant_id = participant["id"]
        logger.debug("Resolved '%s' to ID: %s", name, participant_id)

        await self.rest.agent_api_participants.add_agent_chat_participant(
            chat_id=self.room_id,
            participant=ParticipantRequest(participant_id=participant_id, role=role),
            request_options=DEFAULT_REQUEST_OPTIONS,
        )

        # Update internal participant cache for immediate mention resolution
        # NOTE: WebSocket will eventually deliver participant_added event, but this
        # allows @mentions to work immediately after add_participant returns.
        new_participant = {
            "id": participant_id,
            "name": name,
            "type": participant.get("type", "Agent"),
            "handle": participant.get("handle"),
        }
        self._participants.append(new_participant)
        logger.debug(
            "Updated participant cache: added %s, total=%s",
            name,
            len(self._participants),
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
        logger.debug("Removing participant '%s' from room %s", name, self.room_id)

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
        logger.debug("Resolved '%s' to ID: %s", name, participant_id)

        await self.rest.agent_api_participants.remove_agent_chat_participant(
            self.room_id,
            participant_id,
            request_options=DEFAULT_REQUEST_OPTIONS,
        )

        # Update internal participant cache
        # NOTE: WebSocket will eventually deliver participant_removed event, but this
        # prevents @mentions to the removed participant immediately after removal.
        self._participants = [
            p for p in self._participants if p.get("id") != participant_id
        ]
        logger.debug(
            "Updated participant cache: removed %s, total=%s",
            name,
            len(self._participants),
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
        logger.debug("Looking up peers: page=%s, page_size=%s", page, page_size)
        response = await self.rest.agent_api_peers.list_agent_peers(
            page=page,
            page_size=page_size,
            not_in_chat=self.room_id,
            request_options=DEFAULT_REQUEST_OPTIONS,
        )

        peers = []
        if response.data:
            peers = [
                {
                    "id": peer.id,
                    "name": peer.name,
                    "type": getattr(peer, "type", "Agent"),
                    "handle": getattr(peer, "handle", None),
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
        logger.debug("Getting participants for room %s", self.room_id)
        response = await self.rest.agent_api_participants.list_agent_chat_participants(
            chat_id=self.room_id,
            request_options=DEFAULT_REQUEST_OPTIONS,
        )
        if not response.data:
            return []

        return [
            {
                "id": p.id,
                "name": p.name,
                "type": p.type,
                "handle": getattr(p, "handle", None),
            }
            for p in response.data
        ]

    # --- Contact management tools ---

    async def list_contacts(self, page: int = 1, page_size: int = 50) -> dict[str, Any]:
        """
        List agent's contacts with pagination.

        Args:
            page: Page number (default 1)
            page_size: Items per page (default 50, max 100)

        Returns:
            Dict with 'contacts' list and 'metadata' (page, page_size, total_count, total_pages)
        """
        logger.debug("Listing contacts: page=%s, page_size=%s", page, page_size)
        response = await self.rest.agent_api_contacts.list_agent_contacts(
            page=page, page_size=page_size
        )

        contacts = []
        if response.data:
            contacts = [
                {
                    "id": c.id,
                    "handle": c.handle,
                    "name": c.name,
                    "type": c.type,
                }
                for c in response.data
            ]

        metadata = {
            "page": response.metadata.page if response.metadata else page,
            "page_size": response.metadata.page_size
            if response.metadata
            else page_size,
            "total_count": response.metadata.total_count
            if response.metadata
            else len(contacts),
            "total_pages": response.metadata.total_pages if response.metadata else 1,
        }

        return {"contacts": contacts, "metadata": metadata}

    async def add_contact(
        self, handle: str, message: str | None = None
    ) -> dict[str, Any]:
        """
        Send a contact request to add someone as a contact.

        Args:
            handle: Handle of user/agent to add (e.g., '@john' or '@john/agent-name')
            message: Optional message with the request

        Returns:
            Dict with id and status ('pending' or 'approved')
        """
        logger.debug("Adding contact: handle=%s", handle)
        response = await self.rest.agent_api_contacts.add_agent_contact(
            handle=handle, message=message
        )
        if not response.data:
            raise RuntimeError("Failed to add contact - no response data")
        return {
            "id": response.data.id,
            "status": response.data.status,
        }

    async def remove_contact(
        self, handle: str | None = None, contact_id: str | None = None
    ) -> dict[str, Any]:
        """
        Remove an existing contact by handle or ID.

        Args:
            handle: Contact's handle
            contact_id: Or contact record ID (UUID)

        Returns:
            Dict with status ('removed')

        Raises:
            ValueError: If neither handle nor contact_id is provided
        """
        if handle is None and contact_id is None:
            raise ValueError("Either handle or contact_id must be provided")

        logger.debug("Removing contact: handle=%s, contact_id=%s", handle, contact_id)

        # Build kwargs dynamically to avoid sending null values
        # The REST client uses OMIT for optional params, but passing None sends null
        kwargs: dict[str, Any] = {}
        if handle is not None:
            kwargs["handle"] = handle
        if contact_id is not None:
            kwargs["contact_id"] = contact_id

        response = await self.rest.agent_api_contacts.remove_agent_contact(**kwargs)
        if not response.data:
            raise RuntimeError("Failed to remove contact - no response data")
        return {"status": response.data.status}

    async def list_contact_requests(
        self, page: int = 1, page_size: int = 50, sent_status: str = "pending"
    ) -> dict[str, Any]:
        """
        List both received and sent contact requests.

        Args:
            page: Page number (default 1)
            page_size: Items per page per direction (default 50, max 100)
            sent_status: Filter sent requests by status (default 'pending')

        Returns:
            Dict with 'received', 'sent' lists and 'metadata'
        """
        logger.debug(
            "Listing contact requests: page=%s, page_size=%s, sent_status=%s",
            page,
            page_size,
            sent_status,
        )
        response = await self.rest.agent_api_contacts.list_agent_contact_requests(
            page=page, page_size=page_size, sent_status=sent_status
        )

        received = []
        if response.data and response.data.received:
            received = [
                {
                    "id": r.id,
                    "from_handle": r.from_handle,
                    "from_name": r.from_name,
                    "message": r.message,
                    "status": r.status,
                    "inserted_at": str(r.inserted_at) if r.inserted_at else None,
                }
                for r in response.data.received
            ]

        sent = []
        if response.data and response.data.sent:
            sent = [
                {
                    "id": s.id,
                    "to_handle": s.to_handle,
                    "to_name": s.to_name,
                    "message": s.message,
                    "status": s.status,
                    "inserted_at": str(s.inserted_at) if s.inserted_at else None,
                }
                for s in response.data.sent
            ]

        metadata = {
            "page": response.metadata.page if response.metadata else page,
            "page_size": response.metadata.page_size
            if response.metadata
            else page_size,
            "received": {
                "total": response.metadata.received.total
                if response.metadata and response.metadata.received
                else 0,
                "total_pages": response.metadata.received.total_pages
                if response.metadata and response.metadata.received
                else 0,
            },
            "sent": {
                "total": response.metadata.sent.total
                if response.metadata and response.metadata.sent
                else 0,
                "total_pages": response.metadata.sent.total_pages
                if response.metadata and response.metadata.sent
                else 0,
            },
        }

        return {"received": received, "sent": sent, "metadata": metadata}

    async def respond_contact_request(
        self, action: str, handle: str | None = None, request_id: str | None = None
    ) -> dict[str, Any]:
        """
        Respond to a contact request (approve, reject, or cancel).

        Args:
            action: Action to take ('approve', 'reject', 'cancel')
            handle: Other party's handle
            request_id: Or request ID (UUID)

        Returns:
            Dict with id and status

        Raises:
            ValueError: If neither handle nor request_id is provided
        """
        if handle is None and request_id is None:
            raise ValueError("Either handle or request_id must be provided")

        logger.debug(
            "Responding to contact request: action=%s, handle=%s, request_id=%s",
            action,
            handle,
            request_id,
        )

        # Build kwargs dynamically to avoid sending null values
        # The REST client uses OMIT for optional params, but passing None sends null
        kwargs: dict[str, Any] = {"action": action}
        if handle is not None:
            kwargs["handle"] = handle
        if request_id is not None:
            kwargs["request_id"] = request_id

        response = await self.rest.agent_api_contacts.respond_to_agent_contact_request(
            **kwargs
        )
        if not response.data:
            raise RuntimeError(
                "Failed to respond to contact request - no response data"
            )
        return {
            "id": response.data.id,
            "status": response.data.status,
        }

    # --- Memory management tools ---

    async def list_memories(
        self,
        subject_id: str | None = None,
        scope: str | None = None,
        system: str | None = None,
        type: str | None = None,
        segment: str | None = None,
        content_query: str | None = None,
        page_size: int = 50,
        status: str | None = None,
    ) -> dict[str, Any]:
        """
        List memories accessible to the agent.

        Args:
            subject_id: Filter by subject UUID
            scope: Filter by scope (subject, organization, all)
            system: Filter by memory system (sensory, working, long_term)
            type: Filter by memory type
            segment: Filter by segment (user, agent, tool, guideline)
            content_query: Full-text search query
            page_size: Number of results per page (max 50)
            status: Filter by status (active, superseded, archived, all)

        Returns:
            Dict with memories list and metadata
        """
        logger.debug(
            "Listing memories: subject_id=%s, scope=%s, system=%s",
            subject_id,
            scope,
            system,
        )
        response = await self.rest.agent_api_memories.list_agent_memories(
            subject_id=subject_id,
            scope=scope,
            system=system,
            type=type,
            segment=segment,
            content_query=content_query,
            page_size=page_size,
            status=status,
        )

        memories = []
        if response.data:
            memories = [
                {
                    "id": m.id,
                    "content": m.content,
                    "system": m.system,
                    "type": m.type,
                    "segment": m.segment,
                    "scope": m.scope,
                    "status": m.status,
                    "thought": m.thought,
                    "subject_id": str(m.subject_id) if m.subject_id else None,
                    "source_agent_id": str(m.source_agent_id)
                    if m.source_agent_id
                    else None,
                    "inserted_at": str(m.inserted_at) if m.inserted_at else None,
                }
                for m in response.data
            ]

        metadata = {
            "page_size": response.meta.page_size if response.meta else page_size,
            "total_count": response.meta.total_count
            if response.meta
            else len(memories),
        }

        return {"memories": memories, "metadata": metadata}

    async def store_memory(
        self,
        content: str,
        system: str,
        type: str,
        segment: str,
        thought: str,
        scope: str = "subject",
        subject_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store a new memory entry.

        Args:
            content: The memory content
            system: Memory system tier (sensory, working, long_term)
            type: Memory type (iconic, echoic, haptic, episodic, semantic, procedural)
            segment: Logical segment (user, agent, tool, guideline)
            thought: Agent's reasoning for storing this memory
            scope: Visibility scope (subject, organization)
            subject_id: UUID of the subject (required for subject scope)
            metadata: Additional metadata (tags, references)

        Returns:
            Dict with created memory details
        """
        from thenvoi.client.rest import MemoryCreateRequest

        logger.debug(
            "Storing memory: system=%s, type=%s, segment=%s, scope=%s",
            system,
            type,
            segment,
            scope,
        )
        response = await self.rest.agent_api_memories.create_agent_memory(
            memory=MemoryCreateRequest(
                content=content,
                system=system,
                type=type,
                segment=segment,
                thought=thought,
                scope=scope,
                subject_id=subject_id,
                metadata=metadata,
            )
        )
        if not response.data:
            raise RuntimeError("Failed to store memory - no response data")
        return {
            "id": response.data.id,
            "content": response.data.content,
            "system": response.data.system,
            "type": response.data.type,
            "segment": response.data.segment,
            "scope": response.data.scope,
            "status": response.data.status,
            "thought": response.data.thought,
            "inserted_at": str(response.data.inserted_at)
            if response.data.inserted_at
            else None,
        }

    async def get_memory(self, memory_id: str) -> dict[str, Any]:
        """
        Retrieve a specific memory by ID.

        Args:
            memory_id: Memory ID (UUID)

        Returns:
            Dict with memory details
        """
        logger.debug("Getting memory: id=%s", memory_id)
        response = await self.rest.agent_api_memories.get_agent_memory(id=memory_id)
        if not response.data:
            raise RuntimeError("Failed to get memory - no response data")
        return {
            "id": response.data.id,
            "content": response.data.content,
            "system": response.data.system,
            "type": response.data.type,
            "segment": response.data.segment,
            "scope": response.data.scope,
            "status": response.data.status,
            "thought": response.data.thought,
            "subject_id": str(response.data.subject_id)
            if response.data.subject_id
            else None,
            "source_agent_id": str(response.data.source_agent_id)
            if response.data.source_agent_id
            else None,
            "inserted_at": str(response.data.inserted_at)
            if response.data.inserted_at
            else None,
        }

    async def supersede_memory(self, memory_id: str) -> dict[str, Any]:
        """
        Mark a memory as superseded (soft delete).

        Args:
            memory_id: Memory ID (UUID)

        Returns:
            Dict with updated memory details
        """
        logger.debug("Superseding memory: id=%s", memory_id)
        response = await self.rest.agent_api_memories.supersede_agent_memory(
            id=memory_id
        )
        if not response.data:
            raise RuntimeError("Failed to supersede memory - no response data")
        return {
            "id": response.data.id,
            "status": response.data.status,
        }

    async def archive_memory(self, memory_id: str) -> dict[str, Any]:
        """
        Archive a memory (hide but preserve).

        Args:
            memory_id: Memory ID (UUID)

        Returns:
            Dict with updated memory details
        """
        logger.debug("Archiving memory: id=%s", memory_id)
        response = await self.rest.agent_api_memories.archive_agent_memory(id=memory_id)
        if not response.data:
            raise RuntimeError("Failed to archive memory - no response data")
        return {
            "id": response.data.id,
            "status": response.data.status,
        }

    # --- Mention resolution ---

    def _resolve_mentions(
        self, mentions: list[str] | list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """
        Resolve mention handles, names, or IDs to {id, handle} dicts using cached participants.

        Lookup priority:
        1. Handle (unique identifier like @john or @john/agent-name)
        2. Name (display name, may not be unique)
        3. ID (UUID - for robustness when LLM passes IDs directly)

        Args:
            mentions: List of handles/names/IDs (strings) or already-resolved dicts

        Returns:
            List of {id, handle} dicts

        Raises:
            ValueError: If handle/name/ID is not found in participants
        """
        # Build lookup tables from cached participants
        # Strip @ prefix from handles for consistent matching (backend may or may not include @)
        handle_to_participant = {
            (p.get("handle") or "").lstrip("@"): p for p in self._participants
        }
        name_to_participant = {p.get("name"): p for p in self._participants}
        id_to_participant = {p.get("id"): p for p in self._participants}

        resolved = []
        for mention in mentions:
            if isinstance(mention, str):
                # Strip @ prefix if present (LLMs often include it)
                identifier = mention.lstrip("@")
            else:
                # Already-resolved dict with ID and handle
                if mention.get("id"):
                    resolved.append(
                        {"id": mention["id"], "handle": mention.get("handle", "")}
                    )
                    continue
                raw_identifier = mention.get("handle") or mention.get("name", "")
                identifier = raw_identifier.lstrip("@")

            # Try handle lookup first (handles are unique), then name, then ID
            participant = handle_to_participant.get(identifier)
            if not participant:
                participant = name_to_participant.get(identifier)
            if not participant:
                participant = id_to_participant.get(identifier)

            if not participant:
                available_handles = list(handle_to_participant.keys())
                raise ValueError(
                    f"Unknown participant '{identifier}'. "
                    f"Available handles: {available_handles}"
                )

            resolved.append(
                {"id": participant["id"], "handle": participant.get("handle", "")}
            )

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

    def get_tool_schemas(
        self, format: str, *, include_memory: bool = False
    ) -> list[dict[str, Any]] | list["ToolParam"]:
        """
        Get tool schemas in provider-specific format.

        Args:
            format: Target format - "openai" or "anthropic"
            include_memory: If True, include memory tools (enterprise only)

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
        for definition in iter_tool_definitions(include_memory=include_memory):
            schema = definition.input_model.model_json_schema()
            # Remove Pydantic-specific keys
            schema.pop("title", None)

            if format == "openai":
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": definition.name,
                            "description": definition.input_model.__doc__ or "",
                            "parameters": schema,
                        },
                    }
                )
            elif format == "anthropic":
                tools.append(
                    {
                        "name": definition.name,
                        "description": definition.input_model.__doc__ or "",
                        "input_schema": schema,
                    }
                )
        return tools

    def get_anthropic_tool_schemas(
        self, *, include_memory: bool = False
    ) -> list["ToolParam"]:
        """Get tool schemas in Anthropic format (strongly typed)."""
        return cast(
            list["ToolParam"],
            self.get_tool_schemas("anthropic", include_memory=include_memory),
        )

    def get_openai_tool_schemas(
        self, *, include_memory: bool = False
    ) -> list[dict[str, Any]]:
        """Get tool schemas in OpenAI format (strongly typed)."""
        return cast(
            list[dict[str, Any]],
            self.get_tool_schemas("openai", include_memory=include_memory),
        )

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
            definition = TOOL_DEFINITIONS.get(tool_name)
            if definition:
                arguments = validate_tool_arguments(
                    tool_name,
                    definition.input_model,
                    arguments,
                )
        except ValueError as error:
            return str(error)
        except Exception as e:
            return f"Error validating {tool_name} arguments: {e}"

        definition = TOOL_DEFINITIONS.get(tool_name)
        if definition is None:
            return f"Unknown tool: {tool_name}"

        try:
            method = getattr(self, definition.method_name)
            return await method(**arguments)
        except Exception as e:
            return f"Error executing {tool_name}: {e}"
