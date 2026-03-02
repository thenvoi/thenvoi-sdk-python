"""Direct tests for A2A bridge support modules."""

from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from thenvoi.client.streaming import (
    MessageCreatedPayload,
    ParticipantAddedPayload,
    ParticipantRemovedPayload,
)
from thenvoi.integrations.a2a_bridge.agent_mapping import parse_agent_mapping
from thenvoi.integrations.a2a_bridge.bridge_event_dispatcher import (
    BridgeEventDispatcher,
)
from thenvoi.integrations.a2a_bridge.handler import HandlerResult
from thenvoi.integrations.a2a_bridge.health import HealthServer
from thenvoi.integrations.a2a_bridge.message_dedup import MessageDeduplicator
from thenvoi.integrations.a2a_bridge.participant_directory import ParticipantDirectory
from thenvoi.integrations.a2a_bridge.reconnect_supervisor import ReconnectSupervisor
from thenvoi.integrations.a2a_bridge.route_dispatch import (
    _normalize_handler_result,
    build_dispatch_targets,
    execute_dispatch_targets,
    summarize_dispatch_failures,
)
from thenvoi.integrations.a2a_bridge.session import InMemorySessionStore
from thenvoi.platform.event import (
    MessageEvent,
    ParticipantAddedEvent,
    ParticipantRemovedEvent,
    RoomAddedEvent,
    RoomRemovedEvent,
)
from thenvoi.runtime.types import PlatformMessage


def test_parse_agent_mapping_accepts_valid_pairs_and_rejects_duplicates() -> None:
    assert parse_agent_mapping("alice:handler_a,bob:handler_b") == {
        "alice": "handler_a",
        "bob": "handler_b",
    }
    with pytest.raises(ValueError, match="Duplicate username"):
        parse_agent_mapping("alice:handler_a,alice:handler_b")


def test_message_deduplicator_tracks_seen_ids_with_eviction() -> None:
    dedup = MessageDeduplicator(OrderedDict(), max_size=1)

    assert dedup.seen("msg-1") is False
    assert dedup.seen("msg-1") is True
    assert dedup.seen("msg-2") is False
    assert dedup.seen("msg-1") is False


@pytest.mark.asyncio
async def test_participant_directory_fetches_and_updates_cache() -> None:
    response = SimpleNamespace(
        data=[
            SimpleNamespace(id="u-1", name="Alice", type="User"),
            SimpleNamespace(id="u-2", name="Bob", type="Agent"),
        ]
    )
    link = MagicMock()
    link.rest.agent_api_participants.list_agent_chat_participants = AsyncMock(
        return_value=response
    )
    directory = ParticipantDirectory(link, {})

    participants = await directory.get_for_room("room-1")
    assert participants[0]["name"] == "Alice"
    assert ParticipantDirectory.resolve_sender_name(participants, "u-2") == "Bob"

    directory.on_participant_added(
        "room-1",
        ParticipantAddedPayload(id="u-3", name="Carol", type="Agent"),
    )
    directory.on_participant_removed(
        "room-1",
        ParticipantRemovedPayload(id="u-1"),
    )
    assert [participant["id"] for participant in directory._cache["room-1"]] == [  # noqa: SLF001
        "u-2",
        "u-3",
    ]


@pytest.mark.asyncio
async def test_bridge_event_dispatcher_routes_supported_events() -> None:
    link = MagicMock()
    link.subscribe_room = AsyncMock()
    link.unsubscribe_room = AsyncMock()
    participant_directory = MagicMock()
    participant_directory.preload_room = AsyncMock()
    session_store = MagicMock()
    session_store.remove = AsyncMock()
    on_message = AsyncMock()
    dispatcher = BridgeEventDispatcher(
        link=link,
        participant_directory=participant_directory,
        session_store=session_store,
        on_message=on_message,
    )
    message_event = MessageEvent(
        room_id="room-1",
        payload=MessageCreatedPayload(
            id="msg-1",
            content="hello",
            message_type="text",
            sender_id="u-1",
            sender_type="User",
            sender_name="Alice",
            chat_room_id="room-1",
            inserted_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        ),
    )

    await dispatcher.dispatch(message_event)
    await dispatcher.dispatch(RoomAddedEvent(room_id="room-1"))
    await dispatcher.dispatch(RoomRemovedEvent(room_id="room-1"))
    await dispatcher.dispatch(
        ParticipantAddedEvent(
            room_id="room-1",
            payload=ParticipantAddedPayload(id="u-2", name="Bob", type="Agent"),
        )
    )
    await dispatcher.dispatch(
        ParticipantRemovedEvent(
            room_id="room-1",
            payload=ParticipantRemovedPayload(id="u-2"),
        )
    )

    on_message.assert_awaited_once()
    link.subscribe_room.assert_awaited_once_with("room-1")
    link.unsubscribe_room.assert_awaited_once_with("room-1")
    participant_directory.preload_room.assert_awaited_once_with("room-1")
    session_store.remove.assert_awaited_once_with("room-1")
    participant_directory.on_participant_added.assert_called_once()
    participant_directory.on_participant_removed.assert_called_once()


@pytest.mark.asyncio
async def test_reconnect_supervisor_retries_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        initial_delay=0.01,
        max_delay=0.05,
        multiplier=2.0,
        jitter=0.0,
        max_retries=3,
    )
    supervisor = ReconnectSupervisor(config)
    connected_event = asyncio.Event()
    shutdown_event = asyncio.Event()
    attempts = {"count": 0}

    async def connect_once() -> None:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("temporary failure")
        connected_event.set()

    disconnect = AsyncMock()
    sleep = AsyncMock()
    monkeypatch.setattr(
        "thenvoi.integrations.a2a_bridge.reconnect_supervisor.asyncio.sleep", sleep
    )

    await supervisor.run(
        connect_once=connect_once,
        disconnect=disconnect,
        connected_event=connected_event,
        shutdown_event=shutdown_event,
    )

    assert attempts["count"] == 2
    disconnect.assert_awaited_once()
    sleep.assert_awaited_once()


@pytest.mark.asyncio
async def test_route_dispatch_helpers_cover_success_timeout_and_failure() -> None:
    class _Handled:
        async def handle(
            self,
            message: PlatformMessage,
            mentioned_agent: str,
            tools: object,
        ) -> HandlerResult:
            return HandlerResult.handled(detail=f"ok:{mentioned_agent}:{message.id}")

    class _Slow:
        async def handle(
            self,
            message: PlatformMessage,
            mentioned_agent: str,
            tools: object,
        ) -> HandlerResult:
            await asyncio.sleep(0.05)
            return HandlerResult.handled()

    logger = MagicMock()
    mentions = [
        SimpleNamespace(username="alice"),
        SimpleNamespace(username="alice"),
        SimpleNamespace(username="bob"),
    ]
    targets = build_dispatch_targets(
        mentions,
        agent_mapping={"alice": "handler_a", "bob": "handler_b"},
        handlers={"handler_a": _Handled(), "handler_b": _Slow()},
        logger=logger,
    )
    assert [target.username for target in targets] == ["alice", "bob"]
    assert _normalize_handler_result(HandlerResult.handled()).status == "handled"
    with pytest.raises(TypeError, match="must return HandlerResult"):
        _normalize_handler_result(None)

    failures = await execute_dispatch_targets(
        targets,
        platform_message=PlatformMessage(
            id="msg-1",
            room_id="room-1",
            content="hello",
            sender_id="u-1",
            sender_type="User",
            sender_name="Alice",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        ),
        tools=object(),
        room_id="room-1",
        handler_timeout=0.001,
        logger=logger,
    )

    assert len(failures) == 1
    assert failures[0].username == "bob"
    all_failed, internal, user = summarize_dispatch_failures(
        failures,
        total_targets=2,
        max_error_len=40,
    )
    assert all_failed is False
    assert "handler_b" in internal
    assert "@bob" in user


@pytest.mark.asyncio
async def test_health_server_handler_and_session_store_basics() -> None:
    session_store = InMemorySessionStore()
    await session_store.get_or_create("room-1")
    assert await session_store.count() == 1

    link = MagicMock(is_connected=True)
    server = HealthServer(
        link=link,
        port=0,
        session_store=session_store,
        handler_count=2,
    )
    response = await server._health_handler(MagicMock())  # noqa: SLF001
    body = json.loads(response.text)

    assert response.status == 200
    assert body["status"] == "healthy"
    assert body["active_sessions"] == 1
    await server.stop()
