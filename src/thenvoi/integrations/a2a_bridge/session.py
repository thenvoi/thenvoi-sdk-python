"""Session management for bridge conversations."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class SessionData(BaseModel):
    """Data for an active bridge session."""

    model_config = ConfigDict(frozen=True)

    room_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionStore(Protocol):
    """Protocol for session storage backends."""

    async def get_or_create(self, room_id: str) -> SessionData:
        """Get existing session or create a new one.

        Args:
            room_id: The chat room ID.

        Returns:
            The session data.
        """
        ...

    async def get(self, room_id: str) -> SessionData | None:
        """Get a session by room ID.

        Args:
            room_id: The chat room ID.

        Returns:
            Session data if found, None otherwise.
        """
        ...

    async def remove(self, room_id: str) -> None:
        """Remove a session.

        Args:
            room_id: The chat room ID.
        """
        ...

    async def touch_session(
        self, room_id: str, *, at: datetime | None = None
    ) -> SessionData | None:
        """Update a session's last-activity timestamp.

        Args:
            room_id: The chat room ID.
            at: Optional explicit timestamp; defaults to store clock.

        Returns:
            Updated session data if found, None otherwise.
        """
        ...

    async def list_sessions(self) -> list[SessionData]:
        """List all active sessions.

        Returns:
            List of all session data.
        """
        ...

    async def count(self) -> int:
        """Return the number of active sessions.

        Returns:
            Count of active (non-expired) sessions.
        """
        ...


class InMemorySessionStore:
    """In-memory session store implementation.

    Eviction is opportunistic: expired sessions are removed on every public
    access path (``get_or_create``, ``get``, ``list_sessions``, ``count``).

    Args:
        session_ttl: Optional TTL in seconds. Sessions inactive longer than
            this are evicted automatically. None means no expiration.
    """

    def __init__(
        self,
        session_ttl: float | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._sessions: dict[str, SessionData] = {}
        self._lock = asyncio.Lock()
        self._session_ttl = session_ttl
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._high_session_warned = False

    _HIGH_SESSION_THRESHOLD = 10_000

    @staticmethod
    def _snapshot(session: SessionData) -> SessionData:
        """Return a defensive copy for callers.

        SessionData is immutable, but returning a copy avoids leaking internal
        object identity across module boundaries.
        """
        return session.model_copy(deep=True)

    def _update_last_activity_locked(
        self,
        room_id: str,
        *,
        at: datetime,
    ) -> SessionData | None:
        """Update last_activity under lock and return updated internal session."""
        session = self._sessions.get(room_id)
        if session is None:
            return None
        updated = session.model_copy(update={"last_activity": at})
        self._sessions[room_id] = updated
        return updated

    def _evict_expired(self) -> None:
        """Remove sessions that have exceeded the TTL. Must be called under lock.

        This is O(n) over all sessions and runs opportunistically on all read
        and create paths. Acceptable for expected session counts (hundreds);
        consider a background eviction task if this becomes a bottleneck.
        """
        if self._session_ttl is None:
            return
        now = self._clock()
        expired = [
            room_id
            for room_id, session in self._sessions.items()
            if (now - session.last_activity).total_seconds() > self._session_ttl
        ]
        for room_id in expired:
            del self._sessions[room_id]

        count = len(self._sessions)
        if count >= self._HIGH_SESSION_THRESHOLD:
            if not self._high_session_warned:
                logger.warning(
                    "Session count (%d) exceeds %d — consider a background eviction task",
                    count,
                    self._HIGH_SESSION_THRESHOLD,
                )
                self._high_session_warned = True
        else:
            self._high_session_warned = False

    async def get_or_create(self, room_id: str) -> SessionData:
        async with self._lock:
            self._evict_expired()
            if room_id in self._sessions:
                updated = self._update_last_activity_locked(room_id, at=self._clock())
                assert updated is not None
                return self._snapshot(updated)

            now = self._clock()
            session = SessionData(
                room_id=room_id,
                created_at=now,
                last_activity=now,
            )
            self._sessions[room_id] = session
            return self._snapshot(session)

    async def get(self, room_id: str) -> SessionData | None:
        async with self._lock:
            self._evict_expired()
            session = self._sessions.get(room_id)
            if session is None:
                return None
            return self._snapshot(session)

    async def remove(self, room_id: str) -> None:
        async with self._lock:
            self._sessions.pop(room_id, None)

    async def touch_session(
        self, room_id: str, *, at: datetime | None = None
    ) -> SessionData | None:
        async with self._lock:
            self._evict_expired()
            updated = self._update_last_activity_locked(
                room_id,
                at=at or self._clock(),
            )
            if updated is None:
                return None
            return self._snapshot(updated)

    async def list_sessions(self) -> list[SessionData]:
        async with self._lock:
            self._evict_expired()
            return [self._snapshot(session) for session in self._sessions.values()]

    async def count(self) -> int:
        async with self._lock:
            self._evict_expired()
            return len(self._sessions)
