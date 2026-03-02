"""Core types for composition-based agent architecture."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from thenvoi.core.protocols import AgentToolsProtocol, HistoryConverter

T = TypeVar("T")
H = TypeVar("H")
ToolsT = TypeVar("ToolsT")


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


def resolve_platform_message(
    turn: object | None,
    *,
    msg: PlatformMessage | None = None,
    expected_context_name: str,
) -> PlatformMessage:
    """Resolve legacy arg-bag message inputs into a typed PlatformMessage."""
    candidate = msg if msg is not None else turn
    if isinstance(candidate, PlatformMessage):
        return candidate
    raise TypeError(
        f"on_message() requires {expected_context_name} or msg PlatformMessage"
    )


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


@dataclass(frozen=True)
class ChatMessageTurnContext(Generic[H, ToolsT]):
    """Typed message-turn contract for room-bound chat adapters."""

    msg: PlatformMessage
    tools: ToolsT
    history: H
    participants_msg: str | None
    contacts_msg: str | None
    is_session_bootstrap: bool
    room_id: str

    @classmethod
    def from_message(
        cls,
        *,
        msg: PlatformMessage,
        tools: ToolsT,
        history: H,
        participants_msg: str | None = None,
        contacts_msg: str | None = None,
        is_session_bootstrap: bool = False,
        room_id: str | None = None,
    ) -> ChatMessageTurnContext[H, ToolsT]:
        """Create a chat turn context from canonical message + room data."""
        return cls(
            msg=msg,
            tools=tools,
            history=history,
            participants_msg=participants_msg,
            contacts_msg=contacts_msg,
            is_session_bootstrap=is_session_bootstrap,
            room_id=room_id or msg.room_id,
        )

    @classmethod
    def from_agent_input(
        cls,
        *,
        inp: AgentInput,
        history: H,
        tools: ToolsT,
    ) -> ChatMessageTurnContext[H, ToolsT]:
        """Create a chat turn context from AgentInput and converted history."""
        return cls(
            msg=inp.msg,
            tools=tools,
            history=history,
            participants_msg=inp.participants_msg,
            contacts_msg=inp.contacts_msg,
            is_session_bootstrap=inp.is_session_bootstrap,
            room_id=inp.room_id,
        )


@dataclass(frozen=True)
class ControlMessageTurnContext(Generic[H]):
    """Typed message-turn contract for control-plane adapters."""

    msg: PlatformMessage
    history: H
    participants_msg: str | None
    contacts_msg: str | None
    is_session_bootstrap: bool
    room_id: str

    @classmethod
    def from_message(
        cls,
        *,
        msg: PlatformMessage,
        history: H,
        participants_msg: str | None = None,
        contacts_msg: str | None = None,
        is_session_bootstrap: bool = False,
        room_id: str | None = None,
    ) -> ControlMessageTurnContext[H]:
        """Create a control-plane turn context from canonical message + room data."""
        return cls(
            msg=msg,
            history=history,
            participants_msg=participants_msg,
            contacts_msg=contacts_msg,
            is_session_bootstrap=is_session_bootstrap,
            room_id=room_id or msg.room_id,
        )

    @classmethod
    def from_agent_input(
        cls,
        *,
        inp: AgentInput,
        history: H,
    ) -> ControlMessageTurnContext[H]:
        """Create a control-plane turn context from AgentInput and history."""
        return cls(
            msg=inp.msg,
            history=history,
            participants_msg=inp.participants_msg,
            contacts_msg=inp.contacts_msg,
            is_session_bootstrap=inp.is_session_bootstrap,
            room_id=inp.room_id,
        )
