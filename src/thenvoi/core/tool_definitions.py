"""
Pydantic models for platform tools - single source of truth.

These models define the schema for all tools available to LLMs.
Descriptions come from docstrings and Field(..., description=).
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


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
