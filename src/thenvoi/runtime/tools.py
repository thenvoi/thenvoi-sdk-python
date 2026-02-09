"""
AgentTools - Tools for LLM platform interaction.

Bound to a room_id. Uses AsyncRestClient directly for API calls.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, Field, ValidationError

from thenvoi.client.rest import ChatRoomRequest, DEFAULT_REQUEST_OPTIONS
from thenvoi.core.protocols import AgentToolsProtocol

if TYPE_CHECKING:
    from anthropic.types import ToolParam

    from thenvoi.client.rest import AsyncRestClient

    from .execution import ExecutionContext

logger = logging.getLogger(__name__)


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
        min_length=1,
        description="List of participant names to @mention. At least one required.",
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


# Registry mapping tool names to their input models
TOOL_MODELS: dict[str, type[BaseModel]] = {
    "thenvoi_send_message": SendMessageInput,
    "thenvoi_send_event": SendEventInput,
    "thenvoi_add_participant": AddParticipantInput,
    "thenvoi_remove_participant": RemoveParticipantInput,
    "thenvoi_lookup_peers": LookupPeersInput,
    "thenvoi_get_participants": GetParticipantsInput,
    "thenvoi_create_chatroom": CreateChatroomInput,
    "thenvoi_list_contacts": ListContactsInput,
    "thenvoi_add_contact": AddContactInput,
    "thenvoi_remove_contact": RemoveContactInput,
    "thenvoi_list_contact_requests": ListContactRequestsInput,
    "thenvoi_respond_contact_request": RespondContactRequestInput,
    "thenvoi_list_memories": ListMemoriesInput,
    "thenvoi_store_memory": StoreMemoryInput,
    "thenvoi_get_memory": GetMemoryInput,
    "thenvoi_supersede_memory": SupersedeMemoryInput,
    "thenvoi_archive_memory": ArchiveMemoryInput,
}


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
        logger.debug("Sending message to room %s", self.room_id)

        # Convert to API format
        mention_items = [
            ChatMessageRequestMentionsItem(id=m["id"], name=m["name"])
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
            f"Adding participant '{name}' with role '{role}' to room {self.room_id}"
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
        """
        logger.debug("Removing contact: handle=%s, contact_id=%s", handle, contact_id)
        response = await self.rest.agent_api_contacts.remove_agent_contact(
            handle=handle, contact_id=contact_id
        )
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
        """
        logger.debug(
            "Responding to contact request: action=%s, handle=%s, request_id=%s",
            action,
            handle,
            request_id,
        )
        response = await self.rest.agent_api_contacts.respond_to_agent_contact_request(
            action=action, handle=handle, request_id=request_id
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
        except ValidationError as e:
            # Format validation errors for better LLM readability
            errors = [
                f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
                for err in e.errors()
            ]
            return f"Invalid arguments for {tool_name}: {', '.join(errors)}"
        except Exception as e:
            return f"Error validating {tool_name} arguments: {e}"

        # Dispatch to tool method
        dispatch = {
            "thenvoi_send_message": lambda: self.send_message(
                arguments["content"], arguments.get("mentions")
            ),
            "thenvoi_send_event": lambda: self.send_event(
                arguments["content"],
                arguments["message_type"],
                arguments.get("metadata"),
            ),
            "thenvoi_add_participant": lambda: self.add_participant(
                arguments["name"], arguments.get("role", "member")
            ),
            "thenvoi_remove_participant": lambda: self.remove_participant(
                arguments["name"]
            ),
            "thenvoi_lookup_peers": lambda: self.lookup_peers(
                arguments.get("page", 1), arguments.get("page_size", 50)
            ),
            "thenvoi_get_participants": lambda: self.get_participants(),
            "thenvoi_create_chatroom": lambda: self.create_chatroom(
                arguments.get("task_id")
            ),
            "thenvoi_list_contacts": lambda: self.list_contacts(
                arguments.get("page", 1), arguments.get("page_size", 50)
            ),
            "thenvoi_add_contact": lambda: self.add_contact(
                arguments["handle"], arguments.get("message")
            ),
            "thenvoi_remove_contact": lambda: self.remove_contact(
                arguments.get("handle"), arguments.get("contact_id")
            ),
            "thenvoi_list_contact_requests": lambda: self.list_contact_requests(
                arguments.get("page", 1),
                arguments.get("page_size", 50),
                arguments.get("sent_status", "pending"),
            ),
            "thenvoi_respond_contact_request": lambda: self.respond_contact_request(
                arguments["action"],
                arguments.get("handle"),
                arguments.get("request_id"),
            ),
            "thenvoi_list_memories": lambda: self.list_memories(
                subject_id=arguments.get("subject_id"),
                scope=arguments.get("scope"),
                system=arguments.get("system"),
                type=arguments.get("type"),
                segment=arguments.get("segment"),
                content_query=arguments.get("content_query"),
                page_size=arguments.get("page_size", 50),
                status=arguments.get("status"),
            ),
            "thenvoi_store_memory": lambda: self.store_memory(
                content=arguments["content"],
                system=arguments["system"],
                type=arguments["type"],
                segment=arguments["segment"],
                thought=arguments["thought"],
                scope=arguments.get("scope", "subject"),
                subject_id=arguments.get("subject_id"),
                metadata=arguments.get("metadata"),
            ),
            "thenvoi_get_memory": lambda: self.get_memory(
                memory_id=arguments["memory_id"],
            ),
            "thenvoi_supersede_memory": lambda: self.supersede_memory(
                memory_id=arguments["memory_id"],
            ),
            "thenvoi_archive_memory": lambda: self.archive_memory(
                memory_id=arguments["memory_id"],
            ),
        }

        if tool_name not in dispatch:
            return f"Unknown tool: {tool_name}"

        try:
            return await dispatch[tool_name]()
        except Exception as e:
            return f"Error executing {tool_name}: {e}"
