"""Shared lifecycle/conversion base for framework adapter families."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, cast

from thenvoi.core.protocols import HistoryConverter
from thenvoi.core.types import AgentInput

H = TypeVar("H")


class AdapterLifecycleBase(Generic[H], ABC):
    """Own adapter startup/cleanup and history conversion semantics once."""

    def __init__(
        self,
        *,
        history_converter: HistoryConverter[H] | None = None,
    ) -> None:
        self.history_converter = history_converter
        self.agent_name: str = ""
        self.agent_description: str = ""

    async def on_cleanup(self, room_id: str) -> None:
        """Override for session cleanup."""
        pass

    async def on_started(self, agent_name: str, agent_description: str) -> None:
        """Override for post-start setup."""
        self.agent_name = agent_name
        self.agent_description = agent_description

        if self.history_converter and hasattr(self.history_converter, "set_agent_name"):
            self.history_converter.set_agent_name(agent_name)

    async def on_event(self, inp: AgentInput) -> None:
        """Implements FrameworkAdapter.on_event()."""
        if self.history_converter:
            converted_history: Any = inp.history.convert(self.history_converter)
        else:
            converted_history = inp.history

        await self._dispatch_on_event(inp=inp, history=cast("H", converted_history))

    @abstractmethod
    async def _dispatch_on_event(self, *, inp: AgentInput, history: H) -> None:
        """Dispatch converted event payload to adapter-specific message handler."""
        ...
