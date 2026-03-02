"""Tests for bridge session management."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from thenvoi.integrations.a2a_bridge.session import InMemorySessionStore, SessionData


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
        # Should not create a new session record for the room.
        assert first.room_id == second.room_id == "room-1"
        assert first.created_at == second.created_at
        assert second.last_activity >= first.last_activity

    async def test_returned_session_is_frozen_for_callers(
        self, store: InMemorySessionStore
    ) -> None:
        session = await store.get_or_create("room-1")
        with pytest.raises(ValidationError, match="frozen_instance"):
            session.room_id = "room-2"  # type: ignore[misc]

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

    async def test_get_returns_identity_isolated_snapshot(
        self, store: InMemorySessionStore
    ) -> None:
        created = await store.get_or_create("room-1")
        fetched = await store.get("room-1")
        assert fetched is not None
        assert fetched == created
        assert fetched is not created

    async def test_external_model_copy_cannot_mutate_store_state(
        self, store: InMemorySessionStore
    ) -> None:
        created = await store.get_or_create("room-1")
        caller_copy = created.model_copy(
            update={"last_activity": created.last_activity + timedelta(days=1)}
        )
        assert caller_copy.last_activity > created.last_activity

        stored = await store.get("room-1")
        assert stored is not None
        assert stored.last_activity == created.last_activity

    async def test_list_sessions_returns_defensive_snapshots(
        self, store: InMemorySessionStore
    ) -> None:
        await store.get_or_create("room-1")
        first_list = await store.list_sessions()
        second_list = await store.list_sessions()

        assert len(first_list) == len(second_list) == 1
        assert first_list[0] == second_list[0]
        assert first_list[0] is not second_list[0]

    async def test_list_sessions_empty(self, store: InMemorySessionStore) -> None:
        sessions = await store.list_sessions()
        assert sessions == []

    async def test_concurrent_get_or_create(self, store: InMemorySessionStore) -> None:
        """Concurrent get_or_create calls should not lose sessions."""
        room_ids = [f"room-{i}" for i in range(100)]
        await asyncio.gather(*[store.get_or_create(rid) for rid in room_ids])

        sessions = await store.list_sessions()
        assert len(sessions) == 100

    async def test_concurrent_get_or_create_same_room(
        self, store: InMemorySessionStore
    ) -> None:
        """Concurrent get_or_create for the same room preserves one logical session."""
        results = await asyncio.gather(
            *[store.get_or_create("room-1") for _ in range(50)]
        )
        assert all(result.room_id == "room-1" for result in results)
        assert len({result.created_at for result in results}) == 1
        assert await store.count() == 1

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

    async def test_get_evicts_expired_session(
        self, store: InMemorySessionStore
    ) -> None:
        """get() should opportunistically evict expired sessions."""
        await store.get_or_create("room-1")
        await store.touch_session(
            "room-1",
            at=datetime.now(timezone.utc) - timedelta(seconds=120),
        )

        result = await store.get("room-1")
        assert result is None

    async def test_active_session_not_evicted(
        self, store: InMemorySessionStore
    ) -> None:
        await store.get_or_create("room-1")

        result = await store.get("room-1")
        assert result is not None

    async def test_expired_evicted_from_list(self, store: InMemorySessionStore) -> None:
        await store.get_or_create("room-1")
        await store.get_or_create("room-2")

        # Expire room-1 only
        await store.touch_session(
            "room-1",
            at=datetime.now(timezone.utc) - timedelta(seconds=120),
        )

        sessions = await store.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].room_id == "room-2"

    async def test_expired_evicted_from_count(
        self, store: InMemorySessionStore
    ) -> None:
        await store.get_or_create("room-1")
        await store.get_or_create("room-2")

        # Expire room-1 only
        await store.touch_session(
            "room-1",
            at=datetime.now(timezone.utc) - timedelta(seconds=120),
        )

        assert await store.count() == 1

    async def test_clock_injection_allows_deterministic_expiration(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)

        def _clock() -> datetime:
            return now

        store = InMemorySessionStore(session_ttl=60.0, clock=_clock)
        created = await store.get_or_create("room-1")
        assert created.last_activity == now

        now = now + timedelta(seconds=61)

        expired = await store.get("room-1")
        assert expired is None

    async def test_get_or_create_replaces_expired_session(
        self,
    ) -> None:
        """get_or_create() should evict stale sessions before returning a value."""
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)

        def _clock() -> datetime:
            return now

        store = InMemorySessionStore(session_ttl=60.0, clock=_clock)
        s1 = await store.get_or_create("room-1")
        now = now + timedelta(seconds=61)
        s2 = await store.get_or_create("room-1")
        assert s2.created_at > s1.created_at
        assert s2.room_id == "room-1"

    async def test_no_ttl_means_no_eviction(self) -> None:
        store = InMemorySessionStore(session_ttl=None)
        await store.get_or_create("room-1")
        await store.touch_session(
            "room-1",
            at=datetime.now(timezone.utc) - timedelta(days=365),
        )

        result = await store.get("room-1")
        assert result is not None

    async def test_count_empty(self) -> None:
        store = InMemorySessionStore()
        assert await store.count() == 0

    async def test_count_matches_list(self) -> None:
        store = InMemorySessionStore()
        await store.get_or_create("room-1")
        await store.get_or_create("room-2")
        await store.get_or_create("room-3")

        assert await store.count() == 3
        assert await store.count() == len(await store.list_sessions())

    async def test_high_session_warning_emitted_once_until_count_drops(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Crossing the threshold should warn once and re-arm after recovery."""
        store = InMemorySessionStore(session_ttl=60.0)
        monkeypatch.setattr(store, "_HIGH_SESSION_THRESHOLD", 2, raising=False)

        await store.get_or_create("room-1")
        await store.get_or_create("room-2")

        with patch("thenvoi.integrations.a2a_bridge.session.logger.warning") as warn:
            assert await store.count() == 2
            assert await store.count() == 2
            assert warn.call_count == 1

            await store.remove("room-2")
            assert await store.count() == 1

            await store.get_or_create("room-2")
            assert await store.count() == 2
            assert warn.call_count == 2

    async def test_high_session_warning_rearms_after_ttl_eviction(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """TTL eviction below threshold should reset warning suppression state."""
        store = InMemorySessionStore(session_ttl=60.0)
        monkeypatch.setattr(store, "_HIGH_SESSION_THRESHOLD", 2, raising=False)

        await store.get_or_create("room-1")
        await store.get_or_create("room-2")

        with patch("thenvoi.integrations.a2a_bridge.session.logger.warning") as warn:
            assert await store.count() == 2
            assert warn.call_count == 1

            await store.touch_session(
                "room-1",
                at=datetime.now(timezone.utc) - timedelta(seconds=120),
            )
            assert await store.count() == 1

            await store.get_or_create("room-3")
            assert await store.count() == 2
            assert warn.call_count == 2
