"""CrewAI Flow adapter — Phase 1 surface (state source only).

This module is built up across phases. Phase 1 lands the
``CrewAIFlowStateSource`` protocol, ``RestCrewAIFlowStateSource`` (default),
and ``HistoryCrewAIFlowStateSource`` (test/bootstrap-only). The full
``CrewAIFlowAdapter`` is a stub here so test fixtures and downstream phases
can import it without circular dependencies. Phases 2–5 fill in the
adapter, runtime tools, side-effect executor, reply matching, and safety
policies.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from thenvoi.core.exceptions import ThenvoiConfigError, ThenvoiToolError
from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.converters.crewai_flow import (
    CrewAIFlowSessionState,
    CrewAIFlowStateConverter,
)

logger = logging.getLogger(__name__)


_DEFAULT_PAGE_SIZE = 100
_DEFAULT_CACHE_SIZE = 32


# ---------------------------------------------------------------------------
# State-source protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CrewAIFlowStateSource(Protocol):
    """Hydrates raw task-event dicts for the converter.

    Implementations decide where the events come from. The default routes
    through ``tools.fetch_room_context`` (REST per turn, with caching). The
    test-only history source reads ``AgentInput.history`` directly and is
    explicitly opt-in.
    """

    async def load_task_events(
        self,
        *,
        room_id: str,
        metadata_namespace: str,
        tools: AgentToolsProtocol,
        history: Any,
    ) -> list[dict[str, Any]]:
        """Return raw task-event dicts ordered ascending by inserted_at."""
        ...


# ---------------------------------------------------------------------------
# REST state source (default)
# ---------------------------------------------------------------------------


class _RoomCacheEntry:
    __slots__ = ("events", "latest_inserted_at", "latest_event_id")

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.latest_inserted_at: datetime | None = None
        self.latest_event_id: str = ""


class RestCrewAIFlowStateSource:
    """Default state source.

    Calls ``tools.fetch_room_context`` page by page, filters to task events
    that carry the configured metadata namespace, and orders by
    ``(inserted_at, message_id)`` ascending. Maintains an LRU cache keyed by
    ``(room_id, metadata_namespace)``; on cache hit the source pages from
    page 1 and stops as soon as it sees an event that is at or before the
    cached high-water mark.

    Args:
        page_size: page size for pagination (max 100 per platform contract).
        cache_size: LRU bound for room caches (default 32).
        retry_attempts: number of retries on a REST exception before raising
            ``ThenvoiToolError``. Default 1 (one retry beyond the initial
            attempt).
    """

    def __init__(
        self,
        *,
        page_size: int = _DEFAULT_PAGE_SIZE,
        cache_size: int = _DEFAULT_CACHE_SIZE,
        retry_attempts: int = 1,
    ) -> None:
        if page_size < 1 or page_size > 100:
            raise ThenvoiConfigError(
                "page_size must be in [1, 100] (platform max is 100)"
            )
        if cache_size < 1:
            raise ThenvoiConfigError("cache_size must be >= 1")
        self._page_size = page_size
        self._cache_size = cache_size
        self._retry_attempts = max(0, retry_attempts)
        self._cache: OrderedDict[tuple[str, str], _RoomCacheEntry] = OrderedDict()

    async def load_task_events(
        self,
        *,
        room_id: str,
        metadata_namespace: str,
        tools: AgentToolsProtocol,
        history: Any,
    ) -> list[dict[str, Any]]:
        cache_key = (room_id, metadata_namespace)
        entry = self._cache.get(cache_key)
        if entry is None:
            entry = _RoomCacheEntry()
            self._cache[cache_key] = entry
            self._evict_if_needed()
            return await self._full_fetch(
                room_id=room_id,
                metadata_namespace=metadata_namespace,
                tools=tools,
                entry=entry,
            )

        # Cache hit: refresh from the head and stop on the first known event.
        self._cache.move_to_end(cache_key)
        return await self._incremental_fetch(
            room_id=room_id,
            metadata_namespace=metadata_namespace,
            tools=tools,
            entry=entry,
        )

    async def _fetch_page(
        self,
        *,
        room_id: str,
        page: int,
        tools: AgentToolsProtocol,
    ) -> dict[str, Any]:
        attempt = 0
        last_exc: Exception | None = None
        while attempt <= self._retry_attempts:
            try:
                return await tools.fetch_room_context(
                    room_id=room_id, page=page, page_size=self._page_size
                )
            except Exception as exc:  # pragma: no cover - reraise after retries
                last_exc = exc
                attempt += 1
                if attempt > self._retry_attempts:
                    break
                await asyncio.sleep(0.1 * attempt)
        raise ThenvoiToolError(
            f"fetch_room_context failed for room {room_id}: {last_exc}"
        ) from last_exc

    @staticmethod
    def _is_task_event(item: dict[str, Any], namespace: str) -> bool:
        if item.get("message_type") != "task":
            return False
        metadata = item.get("metadata") or {}
        if not isinstance(metadata, dict):
            return False
        return namespace in metadata

    @staticmethod
    def _coerce_inserted_at(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        return None

    async def _full_fetch(
        self,
        *,
        room_id: str,
        metadata_namespace: str,
        tools: AgentToolsProtocol,
        entry: _RoomCacheEntry,
    ) -> list[dict[str, Any]]:
        page = 1
        collected: list[dict[str, Any]] = []
        while True:
            response = await self._fetch_page(room_id=room_id, page=page, tools=tools)
            data = response.get("data") or []
            if not data:
                break
            for item in data:
                if self._is_task_event(item, metadata_namespace):
                    collected.append(item)
            if len(data) < self._page_size:
                break
            page += 1

        collected.sort(
            key=lambda e: (
                self._coerce_inserted_at(e.get("inserted_at"))
                or datetime.fromtimestamp(0, tz=timezone.utc),
                str(e.get("id") or ""),
            )
        )
        entry.events = list(collected)
        if collected:
            last = collected[-1]
            entry.latest_inserted_at = self._coerce_inserted_at(last.get("inserted_at"))
            entry.latest_event_id = str(last.get("id") or "")
        return list(collected)

    async def _incremental_fetch(
        self,
        *,
        room_id: str,
        metadata_namespace: str,
        tools: AgentToolsProtocol,
        entry: _RoomCacheEntry,
    ) -> list[dict[str, Any]]:
        new_events: list[dict[str, Any]] = []
        page = 1
        stop = False
        while not stop:
            response = await self._fetch_page(room_id=room_id, page=page, tools=tools)
            data = response.get("data") or []
            if not data:
                break
            for item in data:
                inserted = self._coerce_inserted_at(item.get("inserted_at"))
                if (
                    entry.latest_inserted_at is not None
                    and inserted is not None
                    and inserted <= entry.latest_inserted_at
                ):
                    stop = True
                    continue
                if self._is_task_event(item, metadata_namespace):
                    new_events.append(item)
            if len(data) < self._page_size:
                break
            page += 1

        if new_events:
            new_events.sort(
                key=lambda e: (
                    self._coerce_inserted_at(e.get("inserted_at"))
                    or datetime.fromtimestamp(0, tz=timezone.utc),
                    str(e.get("id") or ""),
                )
            )
            entry.events.extend(new_events)
            last = new_events[-1]
            entry.latest_inserted_at = self._coerce_inserted_at(last.get("inserted_at"))
            entry.latest_event_id = str(last.get("id") or "")
        return list(entry.events)

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)


# ---------------------------------------------------------------------------
# History state source (test/bootstrap-only)
# ---------------------------------------------------------------------------


class HistoryCrewAIFlowStateSource:
    """Reads task events from ``AgentInput.history`` directly.

    This source fails closed on every non-bootstrap turn because the default
    preprocessor only hydrates history when ``is_session_bootstrap=True``.
    It is intentionally opt-in for tests and the rare deployment that runs
    only on the bootstrap turn.
    """

    def __init__(self, *, acknowledge_test_only: bool = False) -> None:
        if acknowledge_test_only is not True:
            raise ThenvoiConfigError(
                "HistoryCrewAIFlowStateSource is for tests and bootstrap-only "
                "deployments. Pass acknowledge_test_only=True to construct it. "
                "For production, use RestCrewAIFlowStateSource."
            )
        self._warned_empty = False

    async def load_task_events(
        self,
        *,
        room_id: str,
        metadata_namespace: str,
        tools: AgentToolsProtocol,
        history: Any,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        raw = self._extract_history(history)
        if not raw and not self._warned_empty:
            logger.warning(
                "HistoryCrewAIFlowStateSource: AgentInput.history is empty on "
                "a non-bootstrap turn. State will be lost. If you see this in "
                "production, switch to RestCrewAIFlowStateSource."
            )
            self._warned_empty = True
        for item in raw:
            if not isinstance(item, dict):
                continue
            if item.get("message_type") != "task":
                continue
            metadata = item.get("metadata") or {}
            if isinstance(metadata, dict) and metadata_namespace in metadata:
                events.append(item)
        return events

    @staticmethod
    def _extract_history(history: Any) -> list[dict[str, Any]]:
        if history is None:
            return []
        if isinstance(history, list):
            return history
        for attr in ("raw", "messages", "items"):
            value = getattr(history, attr, None)
            if isinstance(value, list):
                return value
        return []


# ---------------------------------------------------------------------------
# Adapter stub (filled in by Phase 2+)
# ---------------------------------------------------------------------------


class CrewAIFlowAdapter:
    """Stub. The full adapter lands in Phase 2.

    Phase 1 only needs the symbol importable so test fixtures can refer to
    the expected public surface without circular imports. Constructing the
    stub raises until Phase 2 lands.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "CrewAIFlowAdapter is not implemented yet. The Phase 1 module "
            "ships only the state contract and state source. Phase 2 adds "
            "the adapter."
        )


__all__ = [
    "CrewAIFlowAdapter",
    "CrewAIFlowSessionState",
    "CrewAIFlowStateConverter",
    "CrewAIFlowStateSource",
    "HistoryCrewAIFlowStateSource",
    "RestCrewAIFlowStateSource",
]
