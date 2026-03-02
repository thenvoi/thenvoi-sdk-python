"""Simple adapter base class for easy user DX."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable, Callable, MutableMapping, MutableSequence, MutableSet
from functools import wraps
import json
import logging
from typing import Any, Generic, cast
from typing_extensions import TypeVar

from thenvoi.core.adapter_base import AdapterLifecycleBase
from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.core.types import (
    AgentInput,
    ChatMessageTurnContext,
    PlatformMessage,
    resolve_platform_message,
)

# Type variable for history type - bound by converter
H = TypeVar("H")
ToolsT = TypeVar("ToolsT", default=AgentToolsProtocol)
S = TypeVar("S")
HistoryEntryT = TypeVar("HistoryEntryT")

logger = logging.getLogger(__name__)

LegacyChatTurnHandler = Callable[
    ["SimpleAdapter[Any, Any]", ChatMessageTurnContext[Any, Any]],
    Awaitable[None],
]


def legacy_chat_turn_compat(handler: LegacyChatTurnHandler) -> LegacyChatTurnHandler:
    """Accept legacy on_message arg-bag calls and normalize into typed turn context."""

    @wraps(handler)
    async def _wrapped(
        self: SimpleAdapter[Any, Any],
        turn: ChatMessageTurnContext[Any, Any] | PlatformMessage | None = None,
        tools: Any | None = None,
        history: Any | None = None,
        participants_msg: str | None = None,
        contacts_msg: str | None = None,
        *,
        is_session_bootstrap: bool = False,
        room_id: str | None = None,
        msg: PlatformMessage | None = None,
    ) -> None:
        normalized_turn = self.normalize_chat_turn(
            turn,
            tools,
            history,
            participants_msg,
            contacts_msg,
            is_session_bootstrap=is_session_bootstrap,
            room_id=room_id,
            msg=msg,
        )
        await handler(self, normalized_turn)

    return cast("LegacyChatTurnHandler", _wrapped)


def build_metadata_updates(
    *,
    participants_msg: str | None,
    contacts_msg: str | None,
) -> list[str]:
    """Build canonical metadata update messages in deterministic order."""
    updates: list[str] = []
    if participants_msg:
        updates.append(f"[System]: {participants_msg}")
    if contacts_msg:
        updates.append(f"[System]: {contacts_msg}")
    return updates


def prepend_metadata_updates_to_message(
    user_message: str,
    *,
    participants_msg: str | None,
    contacts_msg: str | None,
) -> str:
    """Prepend canonical metadata updates to a provider user message."""
    updates = build_metadata_updates(
        participants_msg=participants_msg,
        contacts_msg=contacts_msg,
    )
    if not updates:
        return user_message
    return "\n\n".join([*updates, user_message])


def append_metadata_updates(
    target: MutableSequence[HistoryEntryT],
    *,
    participants_msg: str | None,
    contacts_msg: str | None,
    make_entry: Callable[[str], HistoryEntryT],
) -> int:
    """Append canonical metadata updates to a target sequence via entry factory."""
    updates = build_metadata_updates(
        participants_msg=participants_msg,
        contacts_msg=contacts_msg,
    )
    for update in updates:
        target.append(make_entry(update))
    return len(updates)


class SimpleAdapter(AdapterLifecycleBase[H], Generic[H, ToolsT]):
    """
    Simple base class for framework adapters.

    Generic over H (history type) and ToolsT (tool capability protocol).
    Users extend this and override on_message().

    Example:
        class MyAdapter(SimpleAdapter[list[ChatMessage], AgentToolsProtocol]):
            def __init__(self):
                super().__init__(history_converter=MyHistoryConverter())

            async def on_message(
                self,
                turn: ChatMessageTurnContext[list[ChatMessage], AgentToolsProtocol],
            ) -> None:
                ...
    """

    @abstractmethod
    async def on_message(
        self,
        turn: ChatMessageTurnContext[H, ToolsT],
    ) -> None:
        """
        Handle incoming message.

        Args:
            turn: Typed message-turn context containing message, tools,
                converted history, metadata updates, and room/session fields.
        """
        ...

    def normalize_chat_turn(
        self,
        turn: ChatMessageTurnContext[H, ToolsT] | PlatformMessage | None = None,
        tools: ToolsT | None = None,
        history: H | None = None,
        participants_msg: str | None = None,
        contacts_msg: str | None = None,
        *,
        is_session_bootstrap: bool = False,
        room_id: str | None = None,
        msg: PlatformMessage | None = None,
    ) -> ChatMessageTurnContext[H, ToolsT]:
        """Normalize context-style and legacy argument-bag on_message calls."""
        if isinstance(turn, ChatMessageTurnContext):
            return turn

        legacy_msg = resolve_platform_message(
            turn,
            msg=msg,
            expected_context_name="ChatMessageTurnContext",
        )
        if tools is None:
            raise TypeError("on_message() missing required argument: 'tools'")
        if history is None:
            raise TypeError("on_message() missing required argument: 'history'")

        return ChatMessageTurnContext.from_message(
            msg=legacy_msg,
            tools=tools,
            history=history,
            participants_msg=participants_msg,
            contacts_msg=contacts_msg,
            is_session_bootstrap=is_session_bootstrap,
            room_id=room_id,
        )

    def stage_room_state(
        self,
        state_by_room: MutableMapping[str, S],
        *,
        room_id: str,
        is_session_bootstrap: bool,
        hydrated_state: S | None,
        default_factory: Callable[[], S],
        ensure_on_non_bootstrap: bool = True,
    ) -> S | None:
        """Apply consistent bootstrap/fallback lifecycle for room-scoped state."""
        if is_session_bootstrap:
            state_by_room[room_id] = (
                hydrated_state if hydrated_state is not None else default_factory()
            )
            return state_by_room[room_id]

        if ensure_on_non_bootstrap and room_id not in state_by_room:
            state_by_room[room_id] = default_factory()
            return state_by_room[room_id]

        return state_by_room.get(room_id)

    def stage_bootstrap_payload(
        self,
        payload_by_room: MutableMapping[str, S],
        *,
        room_id: str,
        is_session_bootstrap: bool,
        payload: S | None,
    ) -> None:
        """Store bootstrap-only payload for room rehydration flows."""
        if not is_session_bootstrap:
            return
        if payload is None:
            payload_by_room.pop(room_id, None)
            return
        payload_by_room[room_id] = payload

    def cleanup_room_state(
        self,
        state_by_room: MutableMapping[str, Any],
        *,
        room_id: str,
    ) -> bool:
        """Drop room-scoped state and return whether state existed."""
        if room_id not in state_by_room:
            return False
        del state_by_room[room_id]
        return True

    def mark_bootstrap_room(
        self,
        bootstrapped_rooms: MutableSet[str],
        *,
        room_id: str,
        is_session_bootstrap: bool,
    ) -> bool:
        """Record a room bootstrap exactly once per active room session."""
        if not is_session_bootstrap or room_id in bootstrapped_rooms:
            return False
        bootstrapped_rooms.add(room_id)
        return True

    def stage_room_history(
        self,
        history_by_room: MutableMapping[str, list[HistoryEntryT]],
        *,
        room_id: str,
        is_session_bootstrap: bool,
        hydrated_history: list[HistoryEntryT] | None = None,
    ) -> list[HistoryEntryT]:
        """Normalize per-room history bootstrap/fallback wiring in one place."""
        room_history = self.stage_room_state(
            history_by_room,
            room_id=room_id,
            is_session_bootstrap=is_session_bootstrap,
            hydrated_state=hydrated_history,
            default_factory=list,
            ensure_on_non_bootstrap=True,
        )
        if room_history is None:
            room_history = []
            history_by_room[room_id] = room_history
        return room_history

    def build_metadata_updates(
        self,
        *,
        participants_msg: str | None,
        contacts_msg: str | None,
    ) -> list[str]:
        """Build canonical metadata update messages in deterministic order."""
        return build_metadata_updates(
            participants_msg=participants_msg,
            contacts_msg=contacts_msg,
        )

    def apply_metadata_updates(
        self,
        target: list[HistoryEntryT],
        *,
        participants_msg: str | None,
        contacts_msg: str | None,
        make_entry: Callable[[str], HistoryEntryT],
    ) -> int:
        """Apply canonical metadata updates to any room-history-like sequence."""
        return append_metadata_updates(
            target,
            participants_msg=participants_msg,
            contacts_msg=contacts_msg,
            make_entry=make_entry,
        )

    def stage_room_history_with_updates(
        self,
        history_by_room: MutableMapping[str, list[HistoryEntryT]],
        *,
        room_id: str,
        is_session_bootstrap: bool,
        hydrated_history: list[HistoryEntryT] | None,
        participants_msg: str | None,
        contacts_msg: str | None,
        make_update_entry: Callable[[str], HistoryEntryT],
    ) -> tuple[list[HistoryEntryT], int]:
        """Stage room history and inject metadata updates using shared policy."""
        room_history = self.stage_room_history(
            history_by_room,
            room_id=room_id,
            is_session_bootstrap=is_session_bootstrap,
            hydrated_history=hydrated_history,
        )
        system_update_count = self.apply_metadata_updates(
            room_history,
            participants_msg=participants_msg,
            contacts_msg=contacts_msg,
            make_entry=make_update_entry,
        )
        return room_history, system_update_count

    def _record_lifecycle_nonfatal(
        self,
        operation: str,
        error: Exception,
        **context: Any,
    ) -> None:
        recorder = getattr(self, "_record_nonfatal_error", None)
        if callable(recorder):
            recorder(operation, error, **context)
            return
        logger.warning(
            "Non-fatal %s error (context=%s): %s",
            operation,
            context,
            error,
            exc_info=True,
        )

    async def send_lifecycle_event(
        self,
        tools: AgentToolsProtocol,
        *,
        content: str,
        message_type: str,
        operation: str,
        metadata: dict[str, Any] | None = None,
        **context: Any,
    ) -> bool:
        """Best-effort event reporting that never interrupts adapter execution."""
        try:
            send_kwargs: dict[str, Any] = {
                "content": content,
                "message_type": message_type,
            }
            if metadata is not None:
                send_kwargs["metadata"] = metadata
            await tools.send_event(**send_kwargs)
            return True
        except Exception as error:
            self._record_lifecycle_nonfatal(operation, error, **context)
            return False

    async def send_tool_call_event(
        self,
        tools: AgentToolsProtocol,
        *,
        payload: dict[str, Any],
        operation: str = "tool_call_event",
        **context: Any,
    ) -> bool:
        """Report tool_call payload using shared lifecycle policy."""
        return await self.send_lifecycle_event(
            tools,
            content=json.dumps(payload, default=str),
            message_type="tool_call",
            operation=operation,
            **context,
        )

    async def send_tool_result_event(
        self,
        tools: AgentToolsProtocol,
        *,
        payload: dict[str, Any],
        operation: str = "tool_result_event",
        **context: Any,
    ) -> bool:
        """Report tool_result payload using shared lifecycle policy."""
        return await self.send_lifecycle_event(
            tools,
            content=json.dumps(payload, default=str),
            message_type="tool_result",
            operation=operation,
            **context,
        )

    async def send_error_event(
        self,
        tools: AgentToolsProtocol,
        *,
        error: str,
        operation: str = "report_error_event",
        **context: Any,
    ) -> bool:
        """Report adapter errors via platform events using shared policy."""
        return await self.send_lifecycle_event(
            tools,
            content=f"Error: {error}",
            message_type="error",
            operation=operation,
            reported_error=error,
            **context,
        )

    async def report_adapter_error(
        self,
        tools: AgentToolsProtocol,
        *,
        error: Exception | str,
        operation: str = "report_error_event",
        **context: Any,
    ) -> bool:
        """Report adapter processing errors with one shared, typed contract."""
        return await self.send_error_event(
            tools,
            error=str(error),
            operation=operation,
            **context,
        )

    async def _dispatch_on_event(self, *, inp: AgentInput, history: H) -> None:
        """Dispatch converted event payload into tool-capable on_message()."""
        await self.on_message(
            ChatMessageTurnContext.from_agent_input(
                inp=inp,
                history=history,
                tools=cast("ToolsT", inp.tools),
            )
        )
