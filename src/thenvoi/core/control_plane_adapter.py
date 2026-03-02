"""Control-plane adapter base for REST/HTTP-orchestrated integrations."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, Generic, TypeVar, cast

from thenvoi.core.adapter_base import AdapterLifecycleBase
from thenvoi.core.types import (
    AgentInput,
    ControlMessageTurnContext,
    PlatformMessage,
    resolve_platform_message,
)

H = TypeVar("H")

LegacyControlTurnHandler = Callable[
    ["ControlPlaneAdapter[Any]", ControlMessageTurnContext[Any]],
    Awaitable[None],
]


def legacy_control_turn_compat(
    handler: LegacyControlTurnHandler,
) -> LegacyControlTurnHandler:
    """Accept legacy control-plane arg-bag calls and normalize into turn context."""

    @wraps(handler)
    async def _wrapped(
        self: ControlPlaneAdapter[Any],
        turn: ControlMessageTurnContext[Any] | PlatformMessage | None = None,
        history: Any | None = None,
        participants_msg: str | None = None,
        contacts_msg: str | None = None,
        *,
        is_session_bootstrap: bool = False,
        room_id: str | None = None,
        msg: PlatformMessage | None = None,
    ) -> None:
        normalized_turn = self.normalize_control_turn(
            turn,
            history,
            participants_msg,
            contacts_msg,
            is_session_bootstrap=is_session_bootstrap,
            room_id=room_id,
            msg=msg,
        )
        await handler(self, normalized_turn)

    return cast("LegacyControlTurnHandler", _wrapped)


class ControlPlaneAdapter(AdapterLifecycleBase[H], Generic[H]):
    """Base adapter for integrations that do not use room-bound tool calls."""

    @abstractmethod
    async def on_message(
        self,
        turn: ControlMessageTurnContext[H],
    ) -> None:
        """Handle incoming message for a control-plane integration."""
        ...

    def normalize_control_turn(
        self,
        turn: ControlMessageTurnContext[H] | PlatformMessage | None = None,
        history: H | None = None,
        participants_msg: str | None = None,
        contacts_msg: str | None = None,
        *,
        is_session_bootstrap: bool = False,
        room_id: str | None = None,
        msg: PlatformMessage | None = None,
    ) -> ControlMessageTurnContext[H]:
        """Normalize context-style and legacy argument-bag on_message calls."""
        if isinstance(turn, ControlMessageTurnContext):
            return turn

        legacy_msg = resolve_platform_message(
            turn,
            msg=msg,
            expected_context_name="ControlMessageTurnContext",
        )
        if history is None:
            raise TypeError("on_message() missing required argument: 'history'")

        return ControlMessageTurnContext.from_message(
            msg=legacy_msg,
            history=history,
            participants_msg=participants_msg,
            contacts_msg=contacts_msg,
            is_session_bootstrap=is_session_bootstrap,
            room_id=room_id,
        )

    async def _dispatch_on_event(self, *, inp: AgentInput, history: H) -> None:
        """Dispatch converted event payload into control-plane on_message()."""
        await self.on_message(
            ControlMessageTurnContext.from_agent_input(inp=inp, history=history)
        )
