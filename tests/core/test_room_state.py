"""Tests for shared room-scoped state containers."""

from __future__ import annotations

from thenvoi.core.room_state import RoomFlagStore, RoomStateStore


def test_room_state_store_behaves_like_mapping() -> None:
    store = RoomStateStore[int]()

    store["room-1"] = 1
    store["room-2"] = 2

    assert store["room-1"] == 1
    assert store.get("room-2") == 2
    assert len(store) == 2

    store.discard("room-1")
    assert "room-1" not in store
    assert len(store) == 1


def test_room_state_store_ensure_returns_existing_state() -> None:
    store = RoomStateStore[list[str]]()

    first = store.ensure("room-1", list)
    first.append("value")
    second = store.ensure("room-1", list)

    assert first is second
    assert second == ["value"]


def test_room_flag_store_tracks_marked_rooms() -> None:
    flags = RoomFlagStore()

    flags.add("room-1")
    flags.add("room-2")

    assert "room-1" in flags
    assert len(flags) == 2

    flags.discard("room-1")
    assert "room-1" not in flags
    assert len(flags) == 1

