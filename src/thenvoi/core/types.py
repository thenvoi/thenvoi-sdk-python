"""Core types for composition-based agent architecture."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from thenvoi.core.protocols import AgentToolsProtocol, HistoryConverter

T = TypeVar("T")


class Capability(str, Enum):
    """Platform tool categories an adapter can expose to the LLM.

    These control tool-schema inclusion only -- they do NOT affect
    runtime event routing (WebSocket subscriptions, contact-event
    strategies, hub-room creation).  Those remain under
    ContactEventConfig / ContactEventStrategy in runtime/types.py.
    """

    MEMORY = "memory"
    CONTACTS = "contacts"


class Emit(str, Enum):
    """Event types an adapter can emit to the platform."""

    EXECUTION = "execution"
    THOUGHTS = "thoughts"
    TASK_EVENTS = "task_events"


@dataclass(frozen=True)
class AdapterFeatures:
    """Shared adapter feature settings. Framework-agnostic knobs only.

    Custom tools are NOT included -- they are adapter-local because each
    framework has its own tool type.

    Accepts list/set inputs for convenience; normalizes to frozen types
    internally.
    """

    capabilities: frozenset[Capability] = frozenset()
    emit: frozenset[Emit] = frozenset()
    include_tools: tuple[str, ...] | None = None
    exclude_tools: tuple[str, ...] | None = None
    include_categories: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "capabilities", frozenset(self.capabilities or ()))
        object.__setattr__(self, "emit", frozenset(self.emit or ()))
        if self.include_tools is not None:
            object.__setattr__(self, "include_tools", tuple(self.include_tools))
        if self.exclude_tools is not None:
            object.__setattr__(self, "exclude_tools", tuple(self.exclude_tools))
        if self.include_categories is not None:
            object.__setattr__(
                self, "include_categories", tuple(self.include_categories)
            )


@dataclass(frozen=True)
class PlatformMessage:
    """Message from the platform."""

    id: str
    room_id: str
    content: str
    sender_id: str
    sender_type: str
    sender_name: str | None
    message_type: str
    metadata: Any  # Flexible - decoupled from transport layer schemas
    created_at: datetime

    def format_for_llm(self) -> str:
        """Format message for LLM consumption."""
        name = self.sender_name or self.sender_type or "Unknown"
        return f"[{name}]: {self.content}"


@dataclass(frozen=True)
class HistoryProvider:
    """
    Provides platform history with lazy conversion.

    Stores raw history, converts on-demand via converter.
    This avoids coupling to any specific framework.
    """

    raw: list[dict[str, Any]]

    def convert(self, converter: "HistoryConverter[T]") -> T:
        """
        Convert history using provided converter.

        Args:
            converter: Framework-specific converter

        Returns:
            History in framework-specific format
        """
        return converter.convert(self.raw)

    def __len__(self) -> int:
        return len(self.raw)

    def __bool__(self) -> bool:
        return bool(self.raw)


@dataclass(frozen=True)
class AgentInput:
    """
    Input to framework adapter.

    Contains everything an adapter needs to process a message.
    History is provided via HistoryProvider for lazy conversion.
    """

    msg: PlatformMessage
    tools: "AgentToolsProtocol"  # Protocol for testability (FakeAgentTools)
    history: HistoryProvider
    participants_msg: str | None
    contacts_msg: str | None  # Contact changes broadcast message
    is_session_bootstrap: bool
    room_id: str
