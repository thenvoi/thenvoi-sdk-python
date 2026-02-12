"""Tests for bridge session management."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from bridge_core.session import InMemorySessionStore, SessionData


class TestSessionData:
    def test_creates_with_defaults(self) -> None:
        session = SessionData(room_id="room-1")
        assert session.room_id == "room-1"
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.last_activity, datetime)

    def test_timestamps_are_utc(self) -> None:
        session = SessionData(room_id="room-1")
        assert session.created_at.tzinfo == timezone.utc
        assert session.last_activity.tzinfo == timezone.utc


class TestInMemorySessionStore:
    @pytest.fixture
    def store(self) -> InMemorySessionStore:
        return InMemorySessionStore()

    async def test_get_or_create_new(self, store: InMemorySessionStore) -> None:
        session = await store.get_or_create("room-1")
        assert session.room_id == "room-1"

    async def test_get_or_create_existing(self, store: InMemorySessionStore) -> None:
        first = await store.get_or_create("room-1")
        second = await store.get_or_create("room-1")
        # Should return the same session, not create a new one
        assert first is second

    async def test_get_existing(self, store: InMemorySessionStore) -> None:
        await store.get_or_create("room-1")
        session = await store.get("room-1")
        assert session is not None
        assert session.room_id == "room-1"

    async def test_get_nonexistent(self, store: InMemorySessionStore) -> None:
        session = await store.get("room-999")
        assert session is None

    async def test_remove(self, store: InMemorySessionStore) -> None:
        await store.get_or_create("room-1")
        await store.remove("room-1")
        session = await store.get("room-1")
        assert session is None

    async def test_remove_nonexistent(self, store: InMemorySessionStore) -> None:
        # Should not raise
        await store.remove("room-999")

    async def test_list_sessions(self, store: InMemorySessionStore) -> None:
        await store.get_or_create("room-1")
        await store.get_or_create("room-2")
        sessions = await store.list_sessions()
        assert len(sessions) == 2
        room_ids = {s.room_id for s in sessions}
        assert room_ids == {"room-1", "room-2"}

    async def test_list_sessions_empty(self, store: InMemorySessionStore) -> None:
        sessions = await store.list_sessions()
        assert sessions == []

    async def test_concurrent_get_or_create(
        self, store: InMemorySessionStore
    ) -> None:
        """Concurrent get_or_create calls should not lose sessions."""
        room_ids = [f"room-{i}" for i in range(100)]
        await asyncio.gather(*[store.get_or_create(rid) for rid in room_ids])

        sessions = await store.list_sessions()
        assert len(sessions) == 100

    async def test_concurrent_get_or_create_same_room(
        self, store: InMemorySessionStore
    ) -> None:
        """Concurrent get_or_create for the same room returns the same session."""
        results = await asyncio.gather(
            *[store.get_or_create("room-1") for _ in range(50)]
        )
        # All results should be the same object
        assert all(r is results[0] for r in results)

    async def test_concurrent_create_and_remove(
        self, store: InMemorySessionStore
    ) -> None:
        """Concurrent creates and removes should not raise."""
        for i in range(20):
            await store.get_or_create(f"room-{i}")

        async def remove_half() -> None:
            for i in range(0, 20, 2):
                await store.remove(f"room-{i}")

        async def read_all() -> list[object]:
            return await store.list_sessions()

        await asyncio.gather(remove_half(), read_all())
        # Should not raise; final count depends on scheduling


class TestInMemorySessionStoreTTL:
    @pytest.fixture
    def store(self) -> InMemorySessionStore:
        return InMemorySessionStore(session_ttl=60.0)

    async def test_expired_session_evicted_on_get(
        self, store: InMemorySessionStore
    ) -> None:
        session = await store.get_or_create("room-1")
        # Backdate last_activity to exceed TTL
        session.last_activity = datetime.now(timezone.utc) - timedelta(seconds=120)

        result = await store.get("room-1")
        assert result is None

    async def test_active_session_not_evicted(
        self, store: InMemorySessionStore
    ) -> None:
        await store.get_or_create("room-1")

        result = await store.get("room-1")
        assert result is not None

    async def test_expired_evicted_from_list(self, store: InMemorySessionStore) -> None:
        s1 = await store.get_or_create("room-1")
        await store.get_or_create("room-2")

        # Expire room-1 only
        s1.last_activity = datetime.now(timezone.utc) - timedelta(seconds=120)

        sessions = await store.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].room_id == "room-2"

    async def test_expired_evicted_from_get_or_create(
        self, store: InMemorySessionStore
    ) -> None:
        s1 = await store.get_or_create("room-1")
        s1.last_activity = datetime.now(timezone.utc) - timedelta(seconds=120)

        # get_or_create should evict the expired one and create fresh
        s2 = await store.get_or_create("room-1")
        assert s2 is not s1
        assert s2.room_id == "room-1"

    async def test_no_ttl_means_no_eviction(self) -> None:
        store = InMemorySessionStore(session_ttl=None)
        session = await store.get_or_create("room-1")
        session.last_activity = datetime.now(timezone.utc) - timedelta(days=365)

        result = await store.get("room-1")
        assert result is not None
