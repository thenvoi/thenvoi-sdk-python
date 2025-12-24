"""Simple adapter base class for easy user DX."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, cast

from thenvoi.core.protocols import AgentToolsProtocol, HistoryConverter
from thenvoi.core.types import AgentInput, PlatformMessage

# Type variable for history type - bound by converter
H = TypeVar("H")


class SimpleAdapter(Generic[H], ABC):
    """
    Simple base class for framework adapters.

    Generic over H (history type) for full type safety.
    Users extend this and override on_message().

    Example:
        class MyAdapter(SimpleAdapter[list[ChatMessage]]):
            def __init__(self):
                super().__init__(history_converter=MyHistoryConverter())

            async def on_message(
                self,
                msg: PlatformMessage,
                tools: AgentToolsProtocol,
                history: list[ChatMessage],  # Fully typed!
                participants_msg: str | None,
                *,
                is_session_bootstrap: bool,
                room_id: str,
            ) -> None:
                ...
    """

    def __init__(
        self,
        *,
        history_converter: HistoryConverter[H] | None = None,
    ):
        """
        Initialize adapter.

        Args:
            history_converter: Optional converter for automatic history conversion.
                              Pass via __init__ to avoid shared state issues.
        """
        self.history_converter = history_converter
        self.agent_name: str = ""
        self.agent_description: str = ""

    @abstractmethod
    async def on_message(
        self,
        msg: PlatformMessage,
        tools: AgentToolsProtocol,
        history: H,
        participants_msg: str | None,
        *,
        is_session_bootstrap: bool,
        room_id: str,
    ) -> None:
        """
        Handle incoming message.

        Args:
            msg: Platform message
            tools: Agent tools (send_message, send_event, etc.)
            history: Converted history as type H
            participants_msg: Participants update message, or None
            is_session_bootstrap: True if adapter session is starting (first message from this room)
            room_id: The room identifier
        """
        ...

    async def on_cleanup(self, room_id: str) -> None:
        """Override for session cleanup."""
        pass

    async def on_started(self, agent_name: str, agent_description: str) -> None:
        """Override for post-start setup."""
        self.agent_name = agent_name
        self.agent_description = agent_description

    # --- FrameworkAdapter protocol implementation ---

    async def on_event(self, inp: AgentInput) -> None:
        """Implements FrameworkAdapter.on_event()."""
        # Convert history if converter is set
        if self.history_converter:
            converted_history: Any = inp.history.convert(self.history_converter)
        else:
            # No converter: pass raw HistoryProvider as H
            # Adapters without converters should type as SimpleAdapter[HistoryProvider]
            converted_history = inp.history

        await self.on_message(
            msg=inp.msg,
            tools=inp.tools,
            history=cast("H", converted_history),
            participants_msg=inp.participants_msg,
            is_session_bootstrap=inp.is_session_bootstrap,
            room_id=inp.room_id,
        )
