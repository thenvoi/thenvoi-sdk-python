"""
Runtime types for Thenvoi agent SDK.

Extracted from core/types.py - data structures used across the runtime layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from thenvoi.platform.event import ContactEvent

    from .contact_tools import ContactTools
    from .tools import AgentTools


@dataclass
class AgentConfig:
    """Configuration for agent runtime."""

    auto_subscribe_existing_rooms: bool = True


@dataclass
class SessionConfig:
    """Configuration for execution context."""

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


# Callback type - receives AgentTools, NOT ThenvoiAgent
MessageHandler = Callable[["PlatformMessage", "AgentTools"], Awaitable[None]]


# --- Contact Event Configuration ---


class ContactEventStrategy(Enum):
    """How to handle contact WebSocket events.

    - DISABLED: Ignore contact events (default). Use manual "check contacts" workflow.
    - CALLBACK: Programmatic handling via on_event callback. No LLM involvement.
    - HUB_ROOM: LLM reasoning in a dedicated hub room.
    """

    DISABLED = "disabled"
    CALLBACK = "callback"
    HUB_ROOM = "hub_room"


# Type alias for contact event callback
ContactEventCallback = Callable[["ContactEvent", "ContactTools"], Awaitable[None]]


@dataclass
class ContactEventConfig:
    """Configuration for contact event handling.

    Composable modes:
    - CALLBACK + broadcast_changes=True: Auto-handle + awareness everywhere
    - HUB_ROOM + broadcast_changes=True: LLM decides + awareness everywhere
    - DISABLED + broadcast_changes=True: Just awareness, manual handling

    Example (auto-approve all requests):
        async def auto_approve(event: ContactEvent, tools: ContactTools) -> None:
            if isinstance(event, ContactRequestReceivedEvent):
                await tools.respond_contact_request("approve", request_id=event.payload.id)

        config = ContactEventConfig(
            strategy=ContactEventStrategy.CALLBACK,
            on_event=auto_approve,
            broadcast_changes=True,
        )

    Example (LLM decides in hub room):
        config = ContactEventConfig(
            strategy=ContactEventStrategy.HUB_ROOM,
            broadcast_changes=True,
        )
    """

    strategy: ContactEventStrategy = ContactEventStrategy.DISABLED
    """Strategy for handling contact events."""

    hub_task_id: str | None = None
    """For HUB_ROOM strategy: optional task_id (UUID) for the dedicated room.
    If None, creates a room without an associated task."""

    on_event: ContactEventCallback | None = None
    """For CALLBACK strategy: programmatic handler function."""

    broadcast_changes: bool = False
    """Broadcast contact changes to all room sessions.

    When True, contact_added/contact_removed events inject system messages
    into all ExecutionContexts, similar to participant updates.
    Works with any strategy (DISABLED, CALLBACK, HUB_ROOM).
    """

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.strategy == ContactEventStrategy.CALLBACK and self.on_event is None:
            raise ValueError("CALLBACK strategy requires on_event callback")
