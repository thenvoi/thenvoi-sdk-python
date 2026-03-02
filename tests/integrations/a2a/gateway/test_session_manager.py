"""Tests for A2A gateway session manager."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from thenvoi.integrations.a2a.gateway.session_manager import GatewaySessionManager
from thenvoi.integrations.a2a.gateway.types import GatewaySessionState


def _rest_client() -> MagicMock:
    rest = MagicMock()
    rest.agent_api_chats.create_agent_chat = AsyncMock()
    rest.agent_api_participants.add_agent_chat_participant = AsyncMock()
    rest.agent_api_events.create_agent_chat_event = AsyncMock()
    return rest


@pytest.mark.asyncio
async def test_get_or_create_room_creates_new_room_and_participant() -> None:
    rest = _rest_client()
    rest.agent_api_chats.create_agent_chat.return_value = SimpleNamespace(
        data=SimpleNamespace(id="room-1")
    )
    manager = GatewaySessionManager(rest)

    room_id, context_id = await manager.get_or_create_room("ctx-1", "peer-1")

    assert room_id == "room-1"
    assert context_id == "ctx-1"
    assert manager.context_to_room == {"ctx-1": "room-1"}
    assert manager.room_participants == {"room-1": {"peer-1"}}
    rest.agent_api_chats.create_agent_chat.assert_awaited_once()
    rest.agent_api_participants.add_agent_chat_participant.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_or_create_room_reuses_existing_context_without_rejoin() -> None:
    rest = _rest_client()
    manager = GatewaySessionManager(rest)
    manager.context_to_room["ctx-1"] = "room-1"
    manager.room_participants["room-1"] = {"peer-1"}

    room_id, context_id = await manager.get_or_create_room("ctx-1", "peer-1")

    assert room_id == "room-1"
    assert context_id == "ctx-1"
    rest.agent_api_chats.create_agent_chat.assert_not_awaited()
    rest.agent_api_participants.add_agent_chat_participant.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_or_create_room_adds_missing_peer_to_existing_room() -> None:
    rest = _rest_client()
    manager = GatewaySessionManager(rest)
    manager.context_to_room["ctx-1"] = "room-1"
    manager.room_participants["room-1"] = {"peer-1"}

    room_id, context_id = await manager.get_or_create_room("ctx-1", "peer-2")

    assert room_id == "room-1"
    assert context_id == "ctx-1"
    assert manager.room_participants["room-1"] == {"peer-1", "peer-2"}
    rest.agent_api_participants.add_agent_chat_participant.assert_awaited_once()


def test_rehydrate_merges_contexts_and_participants() -> None:
    manager = GatewaySessionManager(_rest_client())
    manager.context_to_room = {"ctx-existing": "room-existing"}
    manager.room_participants = {"room-existing": {"peer-a"}}
    history = GatewaySessionState(
        context_to_room={"ctx-existing": "room-existing", "ctx-new": "room-new"},
        room_participants={
            "room-existing": {"peer-b"},
            "room-new": {"peer-c"},
        },
    )

    manager.rehydrate(history)

    assert manager.context_to_room == {
        "ctx-existing": "room-existing",
        "ctx-new": "room-new",
    }
    assert manager.room_participants == {
        "room-existing": {"peer-a", "peer-b"},
        "room-new": {"peer-c"},
    }


@pytest.mark.asyncio
async def test_emit_context_event_persists_task_event_metadata() -> None:
    rest = _rest_client()
    manager = GatewaySessionManager(rest)

    await manager.emit_context_event("room-1", "ctx-1")

    kwargs = rest.agent_api_events.create_agent_chat_event.await_args.kwargs
    assert kwargs["chat_id"] == "room-1"
    assert kwargs["event"].message_type == "task"
    assert kwargs["event"].metadata == {
        "gateway_context_id": "ctx-1",
        "gateway_room_id": "room-1",
    }
