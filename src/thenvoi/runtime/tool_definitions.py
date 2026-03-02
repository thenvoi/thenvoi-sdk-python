"""Canonical tool input models and registry constants."""

from __future__ import annotations

import warnings
from typing import Any, Literal

from pydantic import BaseModel, Field


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

    pass


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

MEMORY_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "thenvoi_list_memories",
        "thenvoi_store_memory",
        "thenvoi_get_memory",
        "thenvoi_supersede_memory",
        "thenvoi_archive_memory",
    }
)

CONTACT_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "thenvoi_list_contacts",
        "thenvoi_add_contact",
        "thenvoi_remove_contact",
        "thenvoi_list_contact_requests",
        "thenvoi_respond_contact_request",
    }
)

ALL_TOOL_NAMES: frozenset[str] = frozenset(TOOL_MODELS.keys())

if MEMORY_TOOL_NAMES - ALL_TOOL_NAMES:
    raise ValueError(f"Unknown memory tools: {MEMORY_TOOL_NAMES - ALL_TOOL_NAMES}")
if CONTACT_TOOL_NAMES - ALL_TOOL_NAMES:
    raise ValueError(f"Unknown contact tools: {CONTACT_TOOL_NAMES - ALL_TOOL_NAMES}")

BASE_TOOL_NAMES: frozenset[str] = ALL_TOOL_NAMES - MEMORY_TOOL_NAMES
CHAT_TOOL_NAMES: frozenset[str] = BASE_TOOL_NAMES - CONTACT_TOOL_NAMES
MCP_TOOL_PREFIX: str = "mcp__thenvoi__"


def mcp_tool_names(names: frozenset[str]) -> list[str]:
    """Convert base tool names to MCP-prefixed names for Claude SDK."""
    return [f"{MCP_TOOL_PREFIX}{name}" for name in sorted(names)]


def get_tool_description(name: str) -> str:
    """Get the LLM-optimized description for a tool."""
    model = TOOL_MODELS.get(name)
    if model and model.__doc__:
        return model.__doc__

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


__all__ = [
    "TOOL_MODELS",
    "MEMORY_TOOL_NAMES",
    "CONTACT_TOOL_NAMES",
    "ALL_TOOL_NAMES",
    "BASE_TOOL_NAMES",
    "CHAT_TOOL_NAMES",
    "MCP_TOOL_PREFIX",
    "mcp_tool_names",
    "get_tool_description",
    "SendMessageInput",
    "SendEventInput",
    "AddParticipantInput",
    "RemoveParticipantInput",
    "LookupPeersInput",
    "GetParticipantsInput",
    "CreateChatroomInput",
    "ListContactsInput",
    "AddContactInput",
    "RemoveContactInput",
    "ListContactRequestsInput",
    "RespondContactRequestInput",
    "ListMemoriesInput",
    "StoreMemoryInput",
    "GetMemoryInput",
    "SupersedeMemoryInput",
    "ArchiveMemoryInput",
]
