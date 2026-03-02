"""Tests for hub room coordination helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from thenvoi.runtime.contacts.hub_room import HubRoomCoordinator


@pytest.mark.asyncio
async def test_get_or_create_room_id_creates_once_and_reuses() -> None:
    create_room = AsyncMock(
        return_value=SimpleNamespace(data=SimpleNamespace(id="hub-room-1"))
    )
    coordinator = HubRoomCoordinator(create_room=create_room, task_id="task-1")

    room_id_first = await coordinator.get_or_create_room_id()
    room_id_second = await coordinator.get_or_create_room_id()

    assert room_id_first == "hub-room-1"
    assert room_id_second == "hub-room-1"
    assert coordinator.room_id == "hub-room-1"
    create_room.assert_awaited_once()


@pytest.mark.asyncio
async def test_wait_ready_returns_true_after_mark_ready() -> None:
    coordinator = HubRoomCoordinator(
        create_room=AsyncMock(),
        task_id=None,
    )
    coordinator.mark_ready()

    ready = await coordinator.wait_ready(timeout_seconds=0.1)

    assert ready is True


@pytest.mark.asyncio
async def test_wait_ready_times_out_when_not_marked() -> None:
    coordinator = HubRoomCoordinator(
        create_room=AsyncMock(),
        task_id=None,
    )

    ready = await coordinator.wait_ready(timeout_seconds=0.01)

    assert ready is False


def test_should_initialize_prompt_returns_true_once() -> None:
    coordinator = HubRoomCoordinator(create_room=AsyncMock(), task_id=None)

    assert coordinator.should_initialize_prompt() is True
    assert coordinator.should_initialize_prompt() is False
