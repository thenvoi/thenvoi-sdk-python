"""Tests for room-scoped tool operation mixin."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from thenvoi.runtime.tool_room_operations import RoomToolOperationsMixin


class _RoomTools(RoomToolOperationsMixin):
    def __init__(
        self,
        *,
        rest: Any,
        participants: list[dict[str, Any]] | None = None,
    ) -> None:
        self.room_id = "room-1"
        self.rest = rest
        self._participants = participants or []


def _response_with_dump(payload: dict[str, Any]) -> SimpleNamespace:
    data = MagicMock()
    data.model_dump.return_value = payload
    return SimpleNamespace(data=data)


@pytest.mark.asyncio
async def test_send_message_resolves_mentions_and_returns_dump() -> None:
    rest = MagicMock()
    rest.agent_api_messages.create_agent_chat_message = AsyncMock(
        return_value=_response_with_dump({"id": "msg-1"})
    )
    tools = _RoomTools(
        rest=rest,
        participants=[{"id": "u-1", "name": "Alice", "handle": "@alice"}],
    )

    result = await tools.send_message("hello", mentions=["Alice"])

    assert result == {"id": "msg-1"}
    kwargs = rest.agent_api_messages.create_agent_chat_message.await_args.kwargs
    assert kwargs["chat_id"] == "room-1"
    assert kwargs["message"].content == "hello"
    assert [mention.id for mention in kwargs["message"].mentions] == ["u-1"]


@pytest.mark.asyncio
async def test_send_message_raises_when_response_missing_data() -> None:
    rest = MagicMock()
    rest.agent_api_messages.create_agent_chat_message = AsyncMock(
        return_value=SimpleNamespace(data=None)
    )
    tools = _RoomTools(rest=rest)

    with pytest.raises(RuntimeError, match="Failed to send message"):
        await tools.send_message("hello", mentions=[])


@pytest.mark.asyncio
async def test_send_event_and_create_chatroom() -> None:
    rest = MagicMock()
    rest.agent_api_events.create_agent_chat_event = AsyncMock(
        return_value=_response_with_dump({"id": "evt-1"})
    )
    rest.agent_api_chats.create_agent_chat = AsyncMock(
        return_value=SimpleNamespace(data=SimpleNamespace(id="room-2"))
    )
    tools = _RoomTools(rest=rest)

    event_result = await tools.send_event("thinking", "thought", {"trace": "1"})
    room_id = await tools.create_chatroom(task_id="task-1")

    assert event_result == {"id": "evt-1"}
    assert room_id == "room-2"


@pytest.mark.asyncio
async def test_add_participant_returns_existing_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rest = MagicMock()
    tools = _RoomTools(rest=rest)
    monkeypatch.setattr(
        tools,
        "get_participants",
        AsyncMock(return_value=[{"id": "u-1", "name": "Alice"}]),
    )

    result = await tools.add_participant("Alice")

    assert result == {
        "id": "u-1",
        "name": "Alice",
        "role": "member",
        "status": "already_in_room",
    }


@pytest.mark.asyncio
async def test_add_participant_looks_up_peer_and_updates_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rest = MagicMock()
    rest.agent_api_participants.add_agent_chat_participant = AsyncMock()
    tools = _RoomTools(rest=rest)
    monkeypatch.setattr(tools, "get_participants", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        tools,
        "_lookup_peer_by_name",
        AsyncMock(
            return_value={
                "id": "agent-2",
                "name": "Bob",
                "type": "Agent",
                "handle": "@bob",
            }
        ),
    )

    result = await tools.add_participant("Bob", role="observer")

    assert result == {
        "id": "agent-2",
        "name": "Bob",
        "role": "observer",
        "status": "added",
    }
    assert tools._participants == [
        {
            "id": "agent-2",
            "name": "Bob",
            "type": "Agent",
            "handle": "@bob",
        }
    ]


@pytest.mark.asyncio
async def test_remove_participant_updates_cache_and_calls_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rest = MagicMock()
    rest.agent_api_participants.remove_agent_chat_participant = AsyncMock()
    tools = _RoomTools(
        rest=rest,
        participants=[
            {"id": "agent-2", "name": "Bob", "type": "Agent", "handle": "@bob"}
        ],
    )
    monkeypatch.setattr(
        tools,
        "get_participants",
        AsyncMock(return_value=[{"id": "agent-2", "name": "Bob"}]),
    )

    result = await tools.remove_participant("Bob")

    assert result == {"id": "agent-2", "name": "Bob", "status": "removed"}
    assert tools._participants == []
    rest.agent_api_participants.remove_agent_chat_participant.assert_awaited_once()


@pytest.mark.asyncio
async def test_remove_participant_raises_for_unknown_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = _RoomTools(rest=MagicMock())
    monkeypatch.setattr(tools, "get_participants", AsyncMock(return_value=[]))

    with pytest.raises(ValueError, match="not found in this room"):
        await tools.remove_participant("missing")


@pytest.mark.asyncio
async def test_lookup_peers_formats_payload_and_metadata() -> None:
    peer = SimpleNamespace(
        id="agent-2",
        name="Bob",
        type="Agent",
        handle="@bob",
        description="Research helper",
    )
    response = SimpleNamespace(
        data=[peer],
        metadata=SimpleNamespace(
            page=2,
            page_size=5,
            total_count=7,
            total_pages=2,
        ),
    )
    rest = MagicMock()
    rest.agent_api_peers.list_agent_peers = AsyncMock(return_value=response)
    tools = _RoomTools(rest=rest)

    result = await tools.lookup_peers(page=2, page_size=5)

    assert result == {
        "peers": [
            {
                "id": "agent-2",
                "name": "Bob",
                "type": "Agent",
                "handle": "@bob",
                "description": "Research helper",
            }
        ],
        "metadata": {
            "page": 2,
            "page_size": 5,
            "total_count": 7,
            "total_pages": 2,
        },
    }


@pytest.mark.asyncio
async def test_get_participants_returns_empty_list_when_api_returns_none() -> None:
    rest = MagicMock()
    rest.agent_api_participants.list_agent_chat_participants = AsyncMock(
        return_value=SimpleNamespace(data=None)
    )
    tools = _RoomTools(rest=rest)

    result = await tools.get_participants()

    assert result == []


def test_resolve_mentions_supports_string_and_dict_inputs() -> None:
    tools = _RoomTools(
        rest=MagicMock(),
        participants=[
            {"id": "u-1", "name": "Alice", "handle": "@alice"},
            {"id": "u-2", "name": "Bob", "handle": "@bob"},
        ],
    )

    resolved = tools._resolve_mentions(
        ["@alice", "Bob", {"id": "custom", "handle": "@custom"}]
    )

    assert resolved == [
        {"id": "u-1", "handle": "@alice"},
        {"id": "u-2", "handle": "@bob"},
        {"id": "custom", "handle": "@custom"},
    ]


def test_resolve_mentions_raises_for_unknown_participant() -> None:
    tools = _RoomTools(rest=MagicMock(), participants=[])

    with pytest.raises(ValueError, match="Unknown participant"):
        tools._resolve_mentions(["@missing"])


@pytest.mark.asyncio
async def test_lookup_peer_by_name_paginates_until_match() -> None:
    tools = _RoomTools(rest=MagicMock())
    tools.lookup_peers = AsyncMock(
        side_effect=[
            {
                "peers": [{"id": "a-1", "name": "Alice"}],
                "metadata": {"total_pages": 2},
            },
            {
                "peers": [{"id": "b-1", "name": "Bob"}],
                "metadata": {"total_pages": 2},
            },
        ]
    )

    result = await tools._lookup_peer_by_name("bob")

    assert result == {"id": "b-1", "name": "Bob"}
