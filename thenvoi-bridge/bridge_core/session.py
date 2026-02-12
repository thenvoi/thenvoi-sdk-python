"""Session management for bridge conversations."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Protocol

from pydantic import BaseModel, Field


class SessionData(BaseModel):
    """Data for an active bridge session."""

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

    async def update_activity(self, room_id: str) -> None:
        """Update the last_activity timestamp for a session.

        Args:
            room_id: The chat room ID.
        """
        ...

    async def remove(self, room_id: str) -> None:
        """Remove a session.

        Args:
            room_id: The chat room ID.
        """
        ...

    async def list_sessions(self) -> list[SessionData]:
        """List all active sessions.

        Returns:
            List of all session data.
        """
        ...


class InMemorySessionStore:
    """In-memory session store implementation.

    Eviction is lazy: expired sessions are removed only when accessed via
    ``get``, ``get_or_create``, or ``list_sessions``.

    .. important::

        Lazy eviction depends on periodic reads to prevent session
        accumulation.  In production, the Docker healthcheck (configured in
        ``docker-compose.yml``) polls ``GET /health`` every 30 s, which calls
        ``list_sessions`` and triggers eviction.  If you run the bridge
        without a healthcheck, consider adding a background eviction task.

    Args:
        session_ttl: Optional TTL in seconds. Sessions inactive longer than
            this are evicted automatically. None means no expiration.
    """

    def __init__(self, session_ttl: float | None = None) -> None:
        self._sessions: dict[str, SessionData] = {}
        self._lock = asyncio.Lock()
        self._session_ttl = session_ttl

    def _evict_expired(self) -> None:
        """Remove sessions that have exceeded the TTL. Must be called under lock.

        This is O(n) over all sessions and runs on every read operation (get,
        get_or_create, list_sessions). Acceptable for expected session counts
        (hundreds); consider a background eviction task if this becomes a bottleneck.
        """
        if self._session_ttl is None:
            return
        now = datetime.now(timezone.utc)
        expired = [
            room_id
            for room_id, session in self._sessions.items()
            if (now - session.last_activity).total_seconds() > self._session_ttl
        ]
        for room_id in expired:
            del self._sessions[room_id]

    async def get_or_create(self, room_id: str) -> SessionData:
        async with self._lock:
            self._evict_expired()
            if room_id in self._sessions:
                self._sessions[room_id].last_activity = datetime.now(timezone.utc)
                return self._sessions[room_id]

            session = SessionData(room_id=room_id)
            self._sessions[room_id] = session
            return session

    async def get(self, room_id: str) -> SessionData | None:
        async with self._lock:
            self._evict_expired()
            return self._sessions.get(room_id)

    async def update_activity(self, room_id: str) -> None:
        async with self._lock:
            session = self._sessions.get(room_id)
            if session:
                session.last_activity = datetime.now(timezone.utc)

    async def remove(self, room_id: str) -> None:
        async with self._lock:
            self._sessions.pop(room_id, None)

    async def list_sessions(self) -> list[SessionData]:
        async with self._lock:
            self._evict_expired()
            return list(self._sessions.values())
