"""CrewAI Flow adapter — message-scoped orchestration for Thenvoi rooms.

Experimental in v1. Use ``CrewAIAdapter`` for normal CrewAI agent turns; this
adapter exists for room routers that need parallel join, sequential
composition, tagged-peer enforcement, and explicit waiting turns without
relying on the model to track pending state from chat history.

Every inbound message creates one local Flow execution. Orchestration state
is stored in Thenvoi task events and reconstructed via
``CrewAIFlowStateSource`` on every turn. The adapter is deterministic on a
single worker through reserve-send-confirm side effects with bounded retry,
and fails closed on ambiguous state.

Phase 1 lands the state contract and state source. Phase 2 (this file)
adds the adapter API skeleton: constructor validation, lifecycle hooks,
and per-room async lock cache. Phases 3–5 add Flow execution, delegation,
reply matching, and safety policies.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Literal, Protocol, runtime_checkable

from thenvoi.converters.crewai_flow import (
    CrewAIFlowJoinPolicy,
    CrewAIFlowSessionState,
    CrewAIFlowStateConverter,
    CrewAIFlowTextOnlyBehavior,
)
from thenvoi.core.exceptions import ThenvoiConfigError, ThenvoiToolError
from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.core.simple_adapter import SimpleAdapter
from thenvoi.core.types import (
    AdapterFeatures,
    Capability,
    Emit,
    PlatformMessage,
)

try:  # pragma: no cover - optional dependency guard
    from crewai.flow.flow import Flow  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised only when crewai missing
    Flow = None  # type: ignore[assignment,misc]

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


# ---------------------------------------------------------------------------
# CrewAIFlowAdapter — Phase 2 skeleton
# ---------------------------------------------------------------------------


_VALID_JOIN_POLICIES = {"all", "first"}
_VALID_TEXT_ONLY = {"error_event", "fallback_send"}
_VALID_TAGGED_PEER = {"require_delegation_before_final", "off"}


class CrewAIFlowAdapter(SimpleAdapter[CrewAIFlowSessionState]):
    """Message-scoped CrewAI Flow adapter.

    Phase 2 lands the constructor, validation, and lifecycle hooks. The
    Flow execution pipeline (``on_message``) is added in Phase 3 — calling
    ``on_message`` before that phase raises ``NotImplementedError``.
    """

    SUPPORTED_EMIT = frozenset({Emit.EXECUTION})
    SUPPORTED_CAPABILITIES = frozenset({Capability.MEMORY, Capability.CONTACTS})

    def __init__(
        self,
        *,
        flow_factory: Callable[[], Any],
        state_source: CrewAIFlowStateSource | None = None,
        join_policy: Literal["all", "first"] = "all",
        metadata_namespace: str | None = None,
        max_delegation_rounds: int = 4,
        max_run_age: timedelta = timedelta(days=7),
        text_only_behavior: Literal["error_event", "fallback_send"] = "error_event",
        tagged_peer_policy: Literal[
            "require_delegation_before_final", "off"
        ] = "require_delegation_before_final",
        sequential_chains: Mapping[str, str] | None = None,
        history_converter: CrewAIFlowStateConverter | None = None,
        features: AdapterFeatures | None = None,
    ) -> None:
        # ---- flow_factory -------------------------------------------------
        if not callable(flow_factory):
            raise ThenvoiConfigError("flow_factory must be callable")
        # Constructor never calls flow_factory(). Runtime checks the returned
        # object's type when the first turn fires (Phase 3).

        # ---- state_source -------------------------------------------------
        if state_source is None:
            state_source = RestCrewAIFlowStateSource()
        if not hasattr(state_source, "load_task_events") or not callable(
            getattr(state_source, "load_task_events", None)
        ):
            raise ThenvoiConfigError(
                "state_source must implement an awaitable "
                "load_task_events(*, room_id, metadata_namespace, tools, history) method"
            )

        # ---- join_policy --------------------------------------------------
        if join_policy not in _VALID_JOIN_POLICIES:
            raise ThenvoiConfigError(
                f"join_policy must be one of {sorted(_VALID_JOIN_POLICIES)}"
            )

        # ---- metadata_namespace -------------------------------------------
        if metadata_namespace is not None:
            if not isinstance(metadata_namespace, str) or not metadata_namespace:
                raise ThenvoiConfigError(
                    "metadata_namespace must be a non-empty string or None"
                )

        # ---- max_delegation_rounds ----------------------------------------
        if not isinstance(max_delegation_rounds, int) or isinstance(
            max_delegation_rounds, bool
        ):
            raise ThenvoiConfigError("max_delegation_rounds must be an int")
        if not (1 <= max_delegation_rounds <= 20):
            raise ThenvoiConfigError(
                "max_delegation_rounds must be in [1, 20]"
            )

        # ---- max_run_age --------------------------------------------------
        if not isinstance(max_run_age, timedelta):
            raise ThenvoiConfigError("max_run_age must be a timedelta")
        if max_run_age <= timedelta(0):
            raise ThenvoiConfigError("max_run_age must be positive")

        # ---- text_only_behavior -------------------------------------------
        if text_only_behavior not in _VALID_TEXT_ONLY:
            raise ThenvoiConfigError(
                f"text_only_behavior must be one of {sorted(_VALID_TEXT_ONLY)}"
            )

        # ---- tagged_peer_policy -------------------------------------------
        if tagged_peer_policy not in _VALID_TAGGED_PEER:
            raise ThenvoiConfigError(
                f"tagged_peer_policy must be one of {sorted(_VALID_TAGGED_PEER)}"
            )

        # ---- sequential_chains --------------------------------------------
        if sequential_chains is not None:
            if not isinstance(sequential_chains, Mapping):
                raise ThenvoiConfigError(
                    "sequential_chains must be a Mapping[str, str]"
                )
            for k, v in sequential_chains.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    raise ThenvoiConfigError(
                        "sequential_chains keys and values must be strings"
                    )

        # ---- history_converter --------------------------------------------
        # Default converter picks up max_run_age and the namespace once
        # on_started resolves the namespace.
        converter = history_converter or CrewAIFlowStateConverter(
            max_run_age=max_run_age
        )

        super().__init__(history_converter=converter, features=features)

        self._flow_factory = flow_factory
        self._state_source = state_source
        self._join_policy: CrewAIFlowJoinPolicy = CrewAIFlowJoinPolicy(join_policy)
        self._configured_metadata_namespace = metadata_namespace
        self.metadata_namespace: str = metadata_namespace or ""
        self._max_delegation_rounds = max_delegation_rounds
        self._max_run_age = max_run_age
        self._text_only_behavior: CrewAIFlowTextOnlyBehavior = (
            CrewAIFlowTextOnlyBehavior(text_only_behavior)
        )
        self._tagged_peer_policy = tagged_peer_policy
        self._sequential_chains: dict[str, str] = dict(sequential_chains or {})

        # Per-room async locks and transient caches. Cleared on on_cleanup.
        self._room_locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_started(
        self, agent_name: str, agent_description: str
    ) -> None:
        await super().on_started(agent_name, agent_description)
        if self._configured_metadata_namespace is None:
            self.metadata_namespace = f"crewai_flow:{agent_name}"
        else:
            self.metadata_namespace = self._configured_metadata_namespace
        # Propagate namespace to the converter so it filters by it.
        if isinstance(self.history_converter, CrewAIFlowStateConverter):
            self.history_converter.metadata_namespace = self.metadata_namespace

    async def on_cleanup(self, room_id: str) -> None:
        self._room_locks.pop(room_id, None)

    # ------------------------------------------------------------------
    # Internal accessors
    # ------------------------------------------------------------------

    def _get_room_lock(self, room_id: str) -> asyncio.Lock:
        lock = self._room_locks.get(room_id)
        if lock is None:
            lock = asyncio.Lock()
            self._room_locks[room_id] = lock
        return lock

    # ------------------------------------------------------------------
    # on_message — filled in Phase 3
    # ------------------------------------------------------------------

    async def on_message(
        self,
        msg: PlatformMessage,
        tools: AgentToolsProtocol,
        history: CrewAIFlowSessionState,
        participants_msg: str | None,
        contacts_msg: str | None,
        *,
        is_session_bootstrap: bool,
        room_id: str,
    ) -> None:
        raise NotImplementedError(
            "CrewAIFlowAdapter.on_message is implemented in Phase 3."
        )


__all__ = [
    "CrewAIFlowAdapter",
    "CrewAIFlowSessionState",
    "CrewAIFlowStateConverter",
    "CrewAIFlowStateSource",
    "HistoryCrewAIFlowStateSource",
    "RestCrewAIFlowStateSource",
]
