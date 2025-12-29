"""
Runtime types for Thenvoi agent SDK.

Extracted from core/types.py - data structures used across the runtime layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
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
