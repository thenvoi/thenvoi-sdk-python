"""Shared ThenvoiLink/WebSocket test doubles."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from unittest.mock import AsyncMock, MagicMock


def _response(data: Any) -> MagicMock:
    response = MagicMock()
    response.data = data
    return response


def _configure_async_iterator(link: MagicMock, events: Iterable[object] | None) -> None:
    async def event_stream() -> Any:
        for event in events or ():
            yield event

    link.__aiter__ = lambda *_args: event_stream()


def make_thenvoi_link_mock(
    *,
    agent_id: str = "agent-123",
    is_connected: bool = False,
    rooms: list[object] | None = None,
    participants: list[object] | None = None,
    context_messages: list[object] | None = None,
    next_message: object | None = None,
    stale_processing_messages: list[object] | None = None,
    stream_events: Iterable[object] | None = None,
) -> MagicMock:
    """Build a canonical ThenvoiLink test double with common SDK contracts."""
    link = MagicMock()
    link.agent_id = agent_id
    link.is_connected = is_connected

    # Core lifecycle and subscription methods.
    link.connect = AsyncMock()
    link.disconnect = AsyncMock()
    link.run_forever = AsyncMock()
    link.subscribe_agent_rooms = AsyncMock()
    link.subscribe_room = AsyncMock()
    link.unsubscribe_room = AsyncMock()
    link.subscribe_agent_contacts = AsyncMock()
    link.unsubscribe_agent_contacts = AsyncMock()

    # Message processing lifecycle.
    link.get_next_message = AsyncMock(return_value=next_message)
    link.get_stale_processing_messages = AsyncMock(
        return_value=list(stale_processing_messages or [])
    )
    link.mark_processing = AsyncMock()
    link.mark_processed = AsyncMock()
    link.mark_failed = AsyncMock()

    # REST clients frequently touched in runtime and platform tests.
    link.rest = MagicMock()
    link.rest.agent_api = MagicMock()
    link.rest.agent_api_identity = MagicMock()
    link.rest.agent_api_identity.get_agent_me = AsyncMock(return_value=_response(None))

    link.rest.agent_api_chats = MagicMock()
    link.rest.agent_api_chats.list_agent_chats = AsyncMock(
        return_value=_response(list(rooms or []))
    )
    chat_data = MagicMock()
    chat_data.id = "chat-123"
    link.rest.agent_api_chats.create_agent_chat = AsyncMock(
        return_value=_response(chat_data)
    )

    link.rest.agent_api_participants = MagicMock()
    link.rest.agent_api_participants.list_agent_chat_participants = AsyncMock(
        return_value=_response(list(participants or []))
    )

    link.rest.agent_api_context = MagicMock()
    link.rest.agent_api_context.get_agent_chat_context = AsyncMock(
        return_value=_response(list(context_messages or []))
    )

    link.rest.agent_api_events = MagicMock()
    link.rest.agent_api_events.create_agent_chat_event = AsyncMock()

    _configure_async_iterator(link, stream_events)
    return link


def make_websocket_client_mock() -> AsyncMock:
    """Build a canonical WebSocketClient test double with room/contact channels."""
    ws = AsyncMock()
    ws.__aenter__ = AsyncMock(return_value=ws)
    ws.__aexit__ = AsyncMock(return_value=None)
    ws.run_forever = AsyncMock()

    ws.join_chat_room_channel = AsyncMock()
    ws.leave_chat_room_channel = AsyncMock()
    ws.join_agent_rooms_channel = AsyncMock()
    ws.join_room_participants_channel = AsyncMock()
    ws.leave_room_participants_channel = AsyncMock()

    ws.join_agent_contacts_channel = AsyncMock()
    ws.leave_agent_contacts_channel = AsyncMock()
    return ws


__all__ = [
    "make_thenvoi_link_mock",
    "make_websocket_client_mock",
]
