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
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import OrderedDict
from collections.abc import Mapping
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Literal, Protocol, Union, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from thenvoi.converters.crewai_flow import (
    CrewAIFlowAmbiguousIdentityError,
    CrewAIFlowDelegationState,
    CrewAIFlowDelegationStatus,
    CrewAIFlowError,
    CrewAIFlowJoinPolicy,
    CrewAIFlowMetadata,
    CrewAIFlowParticipantSnapshot,
    CrewAIFlowRunStatus,
    CrewAIFlowSessionState,
    CrewAIFlowStage,
    CrewAIFlowStateConverter,
    CrewAIFlowTextOnlyBehavior,
    normalize_participant_key,
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
    ``(room_id, metadata_namespace)``; on cache hit the source scans until the
    tail because the platform endpoint is oldest-first and has no cursor.

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

        # Cache hit: the platform endpoint is oldest-first and has no cursor, so
        # scan pages until the short page and append only events past the cached
        # high-water mark.
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
        while True:
            response = await self._fetch_page(room_id=room_id, page=page, tools=tools)
            data = response.get("data") or []
            if not data:
                break
            for item in data:
                inserted = self._coerce_inserted_at(item.get("inserted_at"))
                event_id = str(item.get("id") or item.get("message_id") or "")
                if entry.latest_inserted_at is not None and inserted is not None:
                    if inserted < entry.latest_inserted_at:
                        continue
                    if (
                        inserted == entry.latest_inserted_at
                        and event_id <= entry.latest_event_id
                    ):
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
        if isinstance(history, CrewAIFlowSessionState):
            return [
                {
                    "id": f"history:{run_id}",
                    "message_type": "task",
                    "inserted_at": datetime.now(timezone.utc).isoformat(),
                    "metadata": {
                        metadata_namespace: run.model_dump(
                            mode="json", exclude_none=True
                        )
                    },
                }
                for run_id, run in history.runs.items()
            ]

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
# Decision Pydantic models (Flow output discriminator)
# ---------------------------------------------------------------------------


_DECISION_FORBID = ConfigDict(extra="forbid")


class DelegateItem(BaseModel):
    model_config = _DECISION_FORBID

    delegation_id: str
    target: str
    content: str
    mentions: list[str]


class DirectResponseDecision(BaseModel):
    model_config = _DECISION_FORBID

    decision: Literal["direct_response"]
    content: str
    mentions: list[str] = Field(default_factory=list)


class DelegateDecision(BaseModel):
    model_config = _DECISION_FORBID

    decision: Literal["delegate"]
    delegations: list[DelegateItem] = Field(min_length=1)


class WaitingDecision(BaseModel):
    model_config = _DECISION_FORBID

    decision: Literal["waiting"]
    reason: str


class SynthesizeDecision(BaseModel):
    model_config = _DECISION_FORBID

    decision: Literal["synthesize"]
    content: str
    mentions: list[str] = Field(default_factory=list)


class FailedDecision(BaseModel):
    model_config = _DECISION_FORBID

    decision: Literal["failed"]
    error: CrewAIFlowError


FlowDecision = Union[
    DirectResponseDecision,
    DelegateDecision,
    WaitingDecision,
    SynthesizeDecision,
    FailedDecision,
]


def _validate_decision(raw: Any) -> FlowDecision:
    if not isinstance(raw, dict):
        raise ValueError("decision payload must be a dict")
    kind = raw.get("decision")
    if not isinstance(kind, str):
        raise ValueError(f"unknown decision kind: {kind!r}")
    model_map: dict[str, type[BaseModel]] = {
        "direct_response": DirectResponseDecision,
        "delegate": DelegateDecision,
        "waiting": WaitingDecision,
        "synthesize": SynthesizeDecision,
        "failed": FailedDecision,
    }
    model = model_map.get(kind)
    if model is None:
        raise ValueError(f"unknown decision kind: {kind!r}")
    return model.model_validate(raw)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Runtime tools (read-only Flow surface)
# ---------------------------------------------------------------------------


_current_flow_runtime: ContextVar["CrewAIFlowRuntimeTools | None"] = ContextVar(
    "thenvoi_crewai_flow_runtime", default=None
)


def get_current_flow_runtime() -> "CrewAIFlowRuntimeTools | None":
    """Return the active Flow runtime, or None outside a kickoff_async call."""
    return _current_flow_runtime.get()


class CrewAIFlowRuntimeTools:
    """Read-only Flow runtime surface.

    Flow code that needs to delegate, synthesize, fail, or wait expresses
    that through the Flow's terminal return value. ``create_crewai_tools``
    is the only escape hatch for sub-Crews; the tools it returns route
    through the adapter-owned ``SideEffectExecutor`` so reserve-send-confirm
    semantics apply uniformly.
    """

    def __init__(
        self,
        *,
        room_id: str,
        agent_name: str,
        agent_description: str,
        participants: list[CrewAIFlowParticipantSnapshot],
        tools: AgentToolsProtocol,
        executor: "SideEffectExecutor",
        run_id: str,
        features: AdapterFeatures,
    ) -> None:
        self._room_id = room_id
        self._agent_name = agent_name
        self._agent_description = agent_description
        self._participants = participants
        self._tools = tools
        self._executor = executor
        self._run_id = run_id
        self._features = features

    @property
    def room_id(self) -> str:
        return self._room_id

    @property
    def agent_name(self) -> str:
        return self._agent_name

    @property
    def agent_description(self) -> str:
        return self._agent_description

    @property
    def participants(self) -> list[CrewAIFlowParticipantSnapshot]:
        return list(self._participants)

    @property
    def run_id(self) -> str:
        return self._run_id

    async def lookup_peers(self, page: int = 1, page_size: int = 50) -> Any:
        return await self._tools.lookup_peers(page=page, page_size=page_size)

    async def get_participants(self) -> Any:
        return await self._tools.get_participants()

    def create_crewai_tools(
        self,
        *,
        capabilities: frozenset[Capability] | None = None,
        custom_tools: list[Any] | None = None,
    ) -> list[Any]:
        """Returns CrewAI BaseTool instances bound to the active executor.

        Use inside a ``@listen`` method when spawning a sub-Crew that needs
        to call platform tools. The returned tools enforce the adapter's
        reserve-send-confirm sequence for visible writes.
        """
        from thenvoi.integrations.crewai.tools import (
            CrewAIToolContext,
            build_thenvoi_crewai_tools,
        )

        caps = capabilities if capabilities is not None else self._features.capabilities
        features = AdapterFeatures(
            capabilities=caps,
            emit=self._features.emit,
            include_tools=self._features.include_tools,
            exclude_tools=self._features.exclude_tools,
            include_categories=self._features.include_categories,
        )
        ctx = CrewAIToolContext(room_id=self._room_id, tools=self._tools)
        reporter = CrewAIFlowSubCrewReporter(
            executor=self._executor, run_id=self._run_id
        )

        def _get_context() -> CrewAIToolContext:
            return ctx

        return build_thenvoi_crewai_tools(
            get_context=_get_context,
            reporter=reporter,
            features=features,
            custom_tools=custom_tools,
        )


# ---------------------------------------------------------------------------
# SideEffectExecutor + sub-Crew reporter
# ---------------------------------------------------------------------------


class SideEffectExecutor:
    """Executor for visible writes under the reserve-send-confirm protocol.

    Holds the only reference to ``tools.send_message``. Records reservation
    and confirmation task events under the configured ``metadata_namespace``.
    The executor is created per ``on_message`` turn and discarded when the
    turn returns; it captures ``run_id`` and the active ``CrewAIFlowMetadata``
    builder so all events carry consistent envelope fields.
    """

    def __init__(
        self,
        *,
        tools: AgentToolsProtocol,
        room_id: str,
        run_id: str,
        parent_message_id: str,
        metadata_namespace: str,
        join_policy: CrewAIFlowJoinPolicy,
        text_only_behavior: CrewAIFlowTextOnlyBehavior,
        tagged_peer_keys: list[str] | None = None,
        delegation_rounds: int = 0,
        confirm_retry_attempts: int = 2,
    ) -> None:
        self._tools = tools
        self._room_id = room_id
        self._run_id = run_id
        self._parent_message_id = parent_message_id
        self._namespace = metadata_namespace
        self._join_policy = join_policy
        self._text_only_behavior = text_only_behavior
        self._tagged_peer_keys = list(tagged_peer_keys or [])
        self._delegation_rounds = delegation_rounds
        self._confirm_retry_attempts = confirm_retry_attempts
        self._subcrew_counter = 0
        self._side_effect_aborted = False

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def metadata_namespace(self) -> str:
        return self._namespace

    @property
    def side_effect_aborted(self) -> bool:
        return self._side_effect_aborted

    def next_subcrew_key(self) -> str:
        self._subcrew_counter += 1
        return f"{self._run_id}:subcrew:{self._subcrew_counter}"

    def _envelope(
        self,
        *,
        status: CrewAIFlowRunStatus,
        stage: CrewAIFlowStage,
        delegations: list[CrewAIFlowDelegationState] | None = None,
        final_side_effect_key: str | None = None,
        final_message_id: str | None = None,
        final_reserved_event_id: str | None = None,
        final_sent_event_id: str | None = None,
        error: CrewAIFlowError | None = None,
        delegation_rounds: int | None = None,
    ) -> dict[str, Any]:
        meta = CrewAIFlowMetadata(
            room_id=self._room_id,
            run_id=self._run_id,
            parent_message_id=self._parent_message_id,
            status=status,
            stage=stage,
            join_policy=self._join_policy,
            text_only_behavior=self._text_only_behavior,
            delegations=delegations or [],
            tagged_peer_keys=list(self._tagged_peer_keys),
            delegation_rounds=(
                self._delegation_rounds
                if delegation_rounds is None
                else delegation_rounds
            ),
            final_side_effect_key=final_side_effect_key,
            final_message_id=final_message_id,
            final_reserved_event_id=final_reserved_event_id,
            final_sent_event_id=final_sent_event_id,
            error=error,
        )
        return {self._namespace: meta.model_dump(mode="json", exclude_none=True)}

    async def _send_event(
        self,
        *,
        content: str,
        message_type: str,
        metadata: dict[str, Any],
        retry_attempts: int,
    ) -> Any:
        last_exc: Exception | None = None
        for attempt in range(retry_attempts + 1):
            try:
                return await self._tools.send_event(
                    content=content,
                    message_type=message_type,
                    metadata=metadata,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < retry_attempts:
                    await asyncio.sleep(0.1 * (attempt + 1))
        raise ThenvoiToolError(
            f"send_event ({message_type}) failed after {retry_attempts + 1} attempts: {last_exc}"
        ) from last_exc

    async def record_observed(self) -> None:
        await self._send_event(
            content=f"observed:{self._run_id}",
            message_type="task",
            metadata=self._envelope(
                status=CrewAIFlowRunStatus.OBSERVED,
                stage=CrewAIFlowStage.INITIAL,
            ),
            retry_attempts=2,
        )

    async def record_waiting(self, reason: str) -> None:
        await self._send_event(
            content=f"waiting:{reason}"[:500],
            message_type="task",
            metadata=self._envelope(
                status=CrewAIFlowRunStatus.WAITING,
                stage=CrewAIFlowStage.WAITING_FOR_REPLIES,
            ),
            retry_attempts=2,
        )

    async def record_failed(self, error: CrewAIFlowError) -> None:
        # Best-effort error event for visibility, then the task event.
        try:
            await self._tools.send_event(
                content=f"flow error: {error.code}: {error.message}"[:500],
                message_type="error",
                metadata={"error": error.model_dump()},
            )
        except Exception:  # noqa: BLE001
            logger.warning("Failed to emit error event", exc_info=True)
        await self._send_event(
            content=f"failed:{error.code}",
            message_type="task",
            metadata=self._envelope(
                status=CrewAIFlowRunStatus.FAILED,
                stage=CrewAIFlowStage.FAILED,
                error=error,
            ),
            retry_attempts=2,
        )

    async def execute_direct_response(
        self, *, content: str, mentions: list[str], state: CrewAIFlowSessionState
    ) -> None:
        side_effect_key = f"{self._run_id}:final"

        # Idempotency: if already finalized for this run, suppress.
        existing = state.runs.get(self._run_id)
        if existing is not None and existing.status == CrewAIFlowRunStatus.FINALIZED:
            logger.debug("Skipping duplicate finalization for run %s", self._run_id)
            return
        if existing is not None and existing.final_side_effect_key == side_effect_key:
            if existing.final_message_id is not None:
                logger.debug(
                    "Side effect %s already sent; suppressing", side_effect_key
                )
                return
            # reservation without sent: indeterminate
            await self._record_indeterminate(side_effect_key=side_effect_key)
            return

        # Step 4: write reservation.
        reservation = await self._send_event(
            content=f"reserve:{side_effect_key}",
            message_type="task",
            metadata=self._envelope(
                status=CrewAIFlowRunStatus.SIDE_EFFECT_RESERVED,
                stage=CrewAIFlowStage.SYNTHESIZING,
                final_side_effect_key=side_effect_key,
            ),
            retry_attempts=2,
        )

        # Step 5: visible send.
        try:
            message_response = await self._tools.send_message(
                content=content,
                mentions=mentions or None,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "send_message failed for final side effect %s",
                side_effect_key,
                exc_info=True,
            )
            await self._record_indeterminate(side_effect_key=side_effect_key)
            return
        message_id = (
            getattr(message_response, "id", None)
            if message_response is not None
            else None
        )
        if message_id is None and isinstance(message_response, dict):
            message_id = message_response.get("id")

        # Step 6: confirm with bounded retry (3 attempts total per spec).
        try:
            confirmation_metadata = self._envelope(
                status=CrewAIFlowRunStatus.FINALIZED,
                stage=CrewAIFlowStage.DONE,
                final_side_effect_key=side_effect_key,
                final_message_id=str(message_id) if message_id else None,
                final_reserved_event_id=str(getattr(reservation, "id", "")) or None,
            )
            await self._send_event(
                content=f"finalized:{side_effect_key}",
                message_type="task",
                metadata=confirmation_metadata,
                retry_attempts=self._confirm_retry_attempts,
            )
        except ThenvoiToolError:
            logger.warning(
                "Confirmation event for %s failed after retries; recording indeterminate",
                side_effect_key,
            )
            await self._record_indeterminate(side_effect_key=side_effect_key)

    async def _record_indeterminate(self, *, side_effect_key: str) -> None:
        self._side_effect_aborted = True
        try:
            await self._send_event(
                content=f"indeterminate:{side_effect_key}",
                message_type="task",
                metadata=self._envelope(
                    status=CrewAIFlowRunStatus.INDETERMINATE,
                    stage=CrewAIFlowStage.INDETERMINATE,
                    final_side_effect_key=side_effect_key,
                ),
                retry_attempts=1,
            )
        except ThenvoiToolError:
            logger.error(
                "Could not persist indeterminate state for %s after retries",
                side_effect_key,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Delegation (phase 4)
    # ------------------------------------------------------------------

    async def execute_delegations(
        self,
        *,
        items: list["DelegateItem"],
        state: CrewAIFlowSessionState,
        participants: list[CrewAIFlowParticipantSnapshot],
    ) -> None:
        """Reserve, send, confirm one delegation per item.

        Same-target collision in this batch (or across batch + existing
        pending state) prepends ``[ref:{token}]`` to the visible content
        per the v1 correlation token format.
        """
        participant_dicts = [
            {"id": p.participant_id, "handle": p.handle, "name": p.handle}
            for p in participants
        ]
        existing = state.runs.get(self._run_id)
        existing_delegations = existing.delegations if existing else []

        # Build target counts so duplicates get a correlation token.
        target_counts: dict[str, int] = {}
        normalized_targets: dict[str, str] = {}
        for d in existing_delegations:
            target_counts[d.target.normalized_key] = (
                target_counts.get(d.target.normalized_key, 0) + 1
            )
        for item in items:
            try:
                key = normalize_participant_key(
                    item.target,
                    participants=participant_dicts,
                )
            except CrewAIFlowAmbiguousIdentityError as exc:
                await self.record_failed(
                    CrewAIFlowError(
                        code="ambiguous_participant",
                        message=str(exc),
                    )
                )
                return
            normalized_targets[item.delegation_id] = key
            target_counts[key] = target_counts.get(key, 0) + 1

        # Process items in order.
        accumulated: list[CrewAIFlowDelegationState] = list(existing_delegations)
        for item in items:
            normalized = normalized_targets[item.delegation_id]

            participant_id = ""
            handle: str | None = None
            for p in participants:
                if p.normalized_key == normalized:
                    participant_id = p.participant_id
                    handle = p.handle
                    break
            if not participant_id:
                await self.record_failed(
                    CrewAIFlowError(
                        code="unknown_participant",
                        message=f"Unknown delegation target: {item.target}",
                    )
                )
                return

            side_effect_key = f"{self._run_id}:delegate:{item.delegation_id}"

            # Idempotency: if this delegation_id already has a sent record,
            # suppress the duplicate.
            already = next(
                (
                    d
                    for d in existing_delegations
                    if d.delegation_id == item.delegation_id
                ),
                None,
            )
            if already is not None:
                if already.delegation_message_id is not None:
                    logger.debug(
                        "Delegation %s already sent; suppressing", item.delegation_id
                    )
                    continue
                # Reservation without sent: indeterminate.
                await self._record_indeterminate(side_effect_key=side_effect_key)
                return

            # Reservation event.
            reservation = await self._send_event(
                content=f"reserve:{side_effect_key}",
                message_type="task",
                metadata=self._envelope(
                    status=CrewAIFlowRunStatus.SIDE_EFFECT_RESERVED,
                    stage=CrewAIFlowStage.DELEGATED,
                    delegations=accumulated
                    + [
                        CrewAIFlowDelegationState(
                            delegation_id=item.delegation_id,
                            target=CrewAIFlowParticipantSnapshot(
                                participant_id=participant_id,
                                handle=handle,
                                normalized_key=normalized,
                            ),
                            status=CrewAIFlowDelegationStatus.RESERVED,
                            side_effect_key=side_effect_key,
                        )
                    ],
                ),
                retry_attempts=2,
            )

            # Decide whether to prefix correlation token.
            content = item.content
            if target_counts.get(normalized, 0) > 1:
                import hashlib

                token = hashlib.sha256(side_effect_key.encode()).hexdigest()[:8]
                content = f"[ref:{token}] {content}"

            # Visible send.
            try:
                response = await self._tools.send_message(
                    content=content,
                    mentions=item.mentions or None,
                )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "send_message failed for delegation %s",
                    item.delegation_id,
                    exc_info=True,
                )
                await self._record_indeterminate(side_effect_key=side_effect_key)
                return
            message_id = getattr(response, "id", None) if response is not None else None
            if message_id is None and isinstance(response, dict):
                message_id = response.get("id")

            # Confirmation event.
            try:
                sent_event = await self._send_event(
                    content=f"delegated:{side_effect_key}",
                    message_type="task",
                    metadata=self._envelope(
                        status=CrewAIFlowRunStatus.DELEGATED_PENDING,
                        stage=CrewAIFlowStage.DELEGATED,
                        delegations=accumulated
                        + [
                            CrewAIFlowDelegationState(
                                delegation_id=item.delegation_id,
                                target=CrewAIFlowParticipantSnapshot(
                                    participant_id=participant_id,
                                    handle=handle,
                                    normalized_key=normalized,
                                ),
                                status=CrewAIFlowDelegationStatus.PENDING,
                                side_effect_key=side_effect_key,
                                reserved_event_id=str(getattr(reservation, "id", ""))
                                or None,
                                delegation_message_id=str(message_id)
                                if message_id
                                else None,
                            )
                        ],
                    ),
                    retry_attempts=self._confirm_retry_attempts,
                )
                accumulated.append(
                    CrewAIFlowDelegationState(
                        delegation_id=item.delegation_id,
                        target=CrewAIFlowParticipantSnapshot(
                            participant_id=participant_id,
                            handle=handle,
                            normalized_key=normalized,
                        ),
                        status=CrewAIFlowDelegationStatus.PENDING,
                        side_effect_key=side_effect_key,
                        reserved_event_id=str(getattr(reservation, "id", "")) or None,
                        delegation_message_id=str(message_id) if message_id else None,
                        sent_event_id=str(getattr(sent_event, "id", "")) or None,
                    )
                )
            except ThenvoiToolError:
                logger.warning(
                    "Confirmation event for delegation %s failed; recording indeterminate",
                    item.delegation_id,
                )
                await self._record_indeterminate(side_effect_key=side_effect_key)
                return

    # ------------------------------------------------------------------
    # Reply matching (phase 4)
    # ------------------------------------------------------------------

    async def record_reply(
        self,
        *,
        run_id: str,
        delegation_id: str,
        reply_message_id: str,
        delegations: list[CrewAIFlowDelegationState],
    ) -> CrewAIFlowMetadata:
        # Build updated delegations list with this one set to replied.
        updated: list[CrewAIFlowDelegationState] = []
        for d in delegations:
            if d.delegation_id == delegation_id:
                updated.append(
                    d.model_copy(
                        update={
                            "status": CrewAIFlowDelegationStatus.REPLIED,
                            "reply_message_id": reply_message_id,
                        }
                    )
                )
            else:
                updated.append(d)
        metadata = self._envelope(
            status=CrewAIFlowRunStatus.REPLY_RECORDED,
            stage=CrewAIFlowStage.WAITING_FOR_REPLIES,
            delegations=updated,
        )
        await self._send_event(
            content=f"reply_recorded:{delegation_id}",
            message_type="task",
            metadata=metadata,
            retry_attempts=2,
        )
        return CrewAIFlowMetadata.model_validate(metadata[self._namespace])

    async def record_reply_ambiguous(
        self,
        *,
        run_id: str,
        reason: str,
    ) -> None:
        await self._send_event(
            content=f"reply_ambiguous:{reason}"[:500],
            message_type="task",
            metadata=self._envelope(
                status=CrewAIFlowRunStatus.REPLY_AMBIGUOUS,
                stage=CrewAIFlowStage.WAITING_FOR_REPLIES,
            ),
            retry_attempts=2,
        )

    async def record_buffered(
        self,
        *,
        source_message_id: str,
        content: str,
    ) -> None:
        """Record a buffered synthesis snippet under the run.

        Stored as a task event with status=waiting and a single
        ``buffered_syntheses`` entry. The converter merges entries by
        ``source_message_id``, so multiple turns accumulate into one list.
        """
        from thenvoi.converters.crewai_flow import CrewAIFlowBufferedSynthesis

        envelope = self._envelope(
            status=CrewAIFlowRunStatus.WAITING,
            stage=CrewAIFlowStage.WAITING_FOR_REPLIES,
        )
        envelope[self._namespace]["buffered_syntheses"] = [
            CrewAIFlowBufferedSynthesis(
                source_message_id=source_message_id,
                content=content,
            ).model_dump(mode="json")
        ]
        await self._send_event(
            content=f"buffered:{source_message_id}",
            message_type="task",
            metadata=envelope,
            retry_attempts=2,
        )


# Imported lazily so module loads without crewai installed.
class CrewAIFlowSubCrewReporter:
    """``CrewAIToolReporter`` that routes sub-Crew visible sends via executor."""

    def __init__(self, *, executor: SideEffectExecutor, run_id: str) -> None:
        self._executor = executor
        self._run_id = run_id

    async def execute_send_message(
        self,
        tools: AgentToolsProtocol,
        content: str,
        mentions: list[str],
    ) -> None:
        key = self._executor.next_subcrew_key()
        reservation = await self._executor._send_event(
            content=f"reserve:{key}",
            message_type="task",
            metadata=self._subcrew_envelope(
                status=CrewAIFlowRunStatus.SIDE_EFFECT_RESERVED,
                stage=CrewAIFlowStage.DELEGATED,
                side_effect_key=key,
            ),
            retry_attempts=2,
        )
        try:
            message_response = await self._executor._tools.send_message(
                content=content,
                mentions=mentions or None,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "send_message failed for sub-Crew side effect %s",
                key,
                exc_info=True,
            )
            await self._executor._record_indeterminate(side_effect_key=key)
            return
        message_id = (
            getattr(message_response, "id", None)
            if message_response is not None
            else None
        )
        if message_id is None and isinstance(message_response, dict):
            message_id = message_response.get("id")
        try:
            await self._executor._send_event(
                content=f"subcrew_sent:{key}",
                message_type="task",
                metadata=self._subcrew_envelope(
                    status=CrewAIFlowRunStatus.WAITING,
                    stage=CrewAIFlowStage.WAITING_FOR_REPLIES,
                    side_effect_key=key,
                    message_id=str(message_id) if message_id else None,
                    reserved_event_id=str(getattr(reservation, "id", "")) or None,
                ),
                retry_attempts=self._executor._confirm_retry_attempts,
            )
        except ThenvoiToolError:
            logger.warning("Confirmation event for sub-Crew side effect %s failed", key)
            await self._executor._record_indeterminate(side_effect_key=key)

    def _subcrew_envelope(
        self,
        *,
        status: CrewAIFlowRunStatus,
        stage: CrewAIFlowStage,
        side_effect_key: str,
        message_id: str | None = None,
        reserved_event_id: str | None = None,
    ) -> dict[str, Any]:
        envelope = self._executor._envelope(status=status, stage=stage)
        payload = envelope[self._executor.metadata_namespace]
        payload["final_side_effect_key"] = side_effect_key
        if message_id is not None:
            payload["final_message_id"] = message_id
        if reserved_event_id is not None:
            payload["final_reserved_event_id"] = reserved_event_id
        return envelope

    async def report_call(
        self,
        tools: AgentToolsProtocol,
        tool_name: str,
        input_data: dict[str, Any],
    ) -> None:
        return None

    async def report_result(
        self,
        tools: AgentToolsProtocol,
        tool_name: str,
        result: Any,
        is_error: bool = False,
    ) -> None:
        return None


# ---------------------------------------------------------------------------
# CrewAIFlowAdapter
# ---------------------------------------------------------------------------


_VALID_JOIN_POLICIES = {"all", "first"}
_VALID_TEXT_ONLY = {"error_event", "fallback_send"}
_VALID_TAGGED_PEER = {"require_delegation_before_final", "off"}


class _AmbiguousReply:
    def __init__(
        self,
        *,
        run_id: str,
        parent_message_id: str,
        reason: str,
    ) -> None:
        self.run_id = run_id
        self.parent_message_id = parent_message_id
        self.reason = reason


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
        load_task_events = getattr(state_source, "load_task_events", None)
        if not callable(load_task_events) or not inspect.iscoroutinefunction(
            load_task_events
        ):
            raise ThenvoiConfigError(
                "state_source must implement an awaitable "
                "load_task_events(*, room_id, metadata_namespace, tools, history) method"
            )
        required_state_source_params = {
            "room_id",
            "metadata_namespace",
            "tools",
            "history",
        }
        try:
            state_source_params = set(inspect.signature(load_task_events).parameters)
        except (TypeError, ValueError) as exc:
            raise ThenvoiConfigError(
                "state_source must expose an inspectable load_task_events signature"
            ) from exc
        if not required_state_source_params.issubset(state_source_params):
            raise ThenvoiConfigError(
                "state_source.load_task_events must accept keyword arguments "
                "room_id, metadata_namespace, tools, and history"
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
            raise ThenvoiConfigError("max_delegation_rounds must be in [1, 20]")

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

    async def on_started(self, agent_name: str, agent_description: str) -> None:
        await super().on_started(agent_name, agent_description)
        if self._configured_metadata_namespace is None:
            agent_id = getattr(self, "_thenvoi_agent_id", None) or agent_name
            self.metadata_namespace = f"crewai_flow:{agent_id}"
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
        if not self.metadata_namespace:
            # on_started has not run yet — fail closed.
            raise ThenvoiConfigError(
                "CrewAIFlowAdapter.on_message called before on_started; "
                "metadata_namespace is unresolved."
            )

        lock = self._get_room_lock(room_id)
        async with lock:
            await self._process_one_turn(
                msg=msg,
                tools=tools,
                history=history,
                participants_msg=participants_msg,
                contacts_msg=contacts_msg,
                room_id=room_id,
            )

    async def _process_one_turn(
        self,
        *,
        msg: PlatformMessage,
        tools: AgentToolsProtocol,
        history: CrewAIFlowSessionState,
        participants_msg: str | None,
        contacts_msg: str | None,
        room_id: str,
    ) -> None:
        run_id = msg.id
        executor = SideEffectExecutor(
            tools=tools,
            room_id=room_id,
            run_id=run_id,
            parent_message_id=msg.id,
            metadata_namespace=self.metadata_namespace,
            join_policy=self._join_policy,
            text_only_behavior=self._text_only_behavior,
        )

        # Load durable state via the configured state source.
        try:
            raw_events = await self._state_source.load_task_events(
                room_id=room_id,
                metadata_namespace=self.metadata_namespace,
                tools=tools,
                history=history,
            )
        except ThenvoiToolError as exc:
            logger.warning("State source unavailable: %s", exc)
            await self._safe_record_failed(
                executor,
                CrewAIFlowError(
                    code="state_source_unavailable",
                    message=str(exc)[:500],
                ),
            )
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("State source raised unexpectedly")
            await self._safe_record_failed(
                executor,
                CrewAIFlowError(
                    code="state_source_error",
                    message=f"{type(exc).__name__}: {exc}"[:500],
                ),
            )
            return

        converter = self.history_converter
        if not isinstance(converter, CrewAIFlowStateConverter):
            converter = CrewAIFlowStateConverter(
                metadata_namespace=self.metadata_namespace,
                max_run_age=self._max_run_age,
            )
        state = converter.convert(raw_events)

        # Idempotency: if this run is already terminal, do nothing.
        if state.is_terminal(run_id):
            existing = state.runs.get(run_id)
            logger.debug(
                "Run %s is terminal (%s); skipping",
                run_id,
                existing.status if existing else "unknown",
            )
            return

        # Build the participant snapshot for input + runtime tools.
        participants = self._snapshot_participants(tools)
        current_run = state.runs.get(run_id)
        if current_run is not None:
            executor._tagged_peer_keys = list(current_run.tagged_peer_keys)
            executor._delegation_rounds = current_run.delegation_rounds
        else:
            executor._tagged_peer_keys = self._extract_tagged_peer_keys(
                content=msg.content,
                participants=participants,
            )

        # ------------------------------------------------------------------
        # Reply matching: if this inbound message matches a pending delegation
        # in another run, record the reply, update the in-memory snapshot, and
        # continue into Flow execution for that run.
        # ------------------------------------------------------------------
        matched = self._match_reply_to_delegation(
            msg=msg,
            state=state,
            participants=participants,
        )
        if isinstance(matched, _AmbiguousReply):
            ambiguous_executor = SideEffectExecutor(
                tools=tools,
                room_id=room_id,
                run_id=matched.run_id,
                parent_message_id=matched.parent_message_id,
                metadata_namespace=self.metadata_namespace,
                join_policy=(
                    state.runs[matched.run_id].join_policy
                    if matched.run_id in state.runs
                    else self._join_policy
                ),
                text_only_behavior=self._text_only_behavior,
            )
            ambiguous_run = state.runs.get(matched.run_id)
            if ambiguous_run is not None:
                ambiguous_executor._tagged_peer_keys = list(
                    ambiguous_run.tagged_peer_keys
                )
                ambiguous_executor._delegation_rounds = ambiguous_run.delegation_rounds
            await ambiguous_executor.record_reply_ambiguous(
                run_id=matched.run_id,
                reason=matched.reason,
            )
            return

        if matched is not None:
            other_run_id, delegation_id, run_meta = matched
            run_executor = SideEffectExecutor(
                tools=tools,
                room_id=room_id,
                run_id=other_run_id,
                parent_message_id=run_meta.parent_message_id,
                metadata_namespace=self.metadata_namespace,
                join_policy=run_meta.join_policy,
                text_only_behavior=self._text_only_behavior,
                tagged_peer_keys=run_meta.tagged_peer_keys,
                delegation_rounds=run_meta.delegation_rounds,
            )
            updated_run = await run_executor.record_reply(
                run_id=other_run_id,
                delegation_id=delegation_id,
                reply_message_id=msg.id,
                delegations=run_meta.delegations,
            )
            merged_run = run_meta.model_copy(
                update={
                    "status": updated_run.status,
                    "stage": updated_run.stage,
                    "delegations": updated_run.delegations,
                }
            )
            state = state.model_copy(
                update={
                    "runs": {
                        **state.runs,
                        other_run_id: merged_run,
                    }
                }
            )
            run_id = other_run_id
            executor = run_executor

        # If no match, only User-typed senders start a new run; Agent-typed
        # senders are discarded with a debug log.
        sender_type = getattr(msg, "sender_type", "User")
        if sender_type == "Agent" and matched is None:
            logger.debug(
                "Agent-typed sender %s did not match a pending delegation; discarding",
                msg.sender_id,
            )
            return

        # Construct Flow.
        try:
            flow = self._flow_factory()
        except Exception as exc:  # noqa: BLE001
            logger.exception("flow_factory raised")
            await self._safe_record_failed(
                executor,
                CrewAIFlowError(
                    code="flow_factory_error",
                    message=f"{type(exc).__name__}: {exc}"[:500],
                ),
            )
            return

        if not callable(getattr(flow, "kickoff_async", None)):
            await self._safe_record_failed(
                executor,
                CrewAIFlowError(
                    code="flow_factory_error",
                    message=(
                        f"flow_factory returned {type(flow).__name__}, "
                        "expected an object with an awaitable kickoff_async method"
                    ),
                ),
            )
            return

        inputs = self._build_inputs(
            msg=msg,
            state=state,
            participants=participants,
            participants_msg=participants_msg,
            contacts_msg=contacts_msg,
            room_id=room_id,
        )

        runtime = CrewAIFlowRuntimeTools(
            room_id=room_id,
            agent_name=self.agent_name,
            agent_description=self.agent_description,
            participants=participants,
            tools=tools,
            executor=executor,
            run_id=run_id,
            features=self.features,
        )
        token = _current_flow_runtime.set(runtime)
        try:
            try:
                result = await flow.kickoff_async(inputs)
            except Exception as exc:  # noqa: BLE001
                logger.exception("kickoff_async raised")
                await self._safe_record_failed(
                    executor,
                    CrewAIFlowError(
                        code="flow_runtime_error",
                        message=f"{type(exc).__name__}: {exc}"[:500],
                    ),
                )
                return
        finally:
            _current_flow_runtime.reset(token)

        if executor.side_effect_aborted:
            return

        await self._handle_result(
            result=result,
            executor=executor,
            state=state,
            msg=msg,
        )

    async def _handle_result(
        self,
        *,
        result: Any,
        executor: SideEffectExecutor,
        state: CrewAIFlowSessionState,
        msg: PlatformMessage,
    ) -> None:
        # Reject FlowStreamingOutput (string class match avoids the import).
        if type(result).__name__ == "FlowStreamingOutput":
            await self._safe_record_failed(
                executor,
                CrewAIFlowError(
                    code="streaming_output_unsupported",
                    message="FlowStreamingOutput is not supported in v1",
                ),
            )
            return

        # Validate decision shape.
        try:
            decision = _validate_decision(result)
        except (ValueError, ValidationError) as exc:
            await self._handle_malformed(
                executor=executor,
                state=state,
                raw=result,
                error=str(exc)[:500],
            )
            return

        if isinstance(decision, DirectResponseDecision):
            await executor.execute_direct_response(
                content=decision.content,
                mentions=decision.mentions,
                state=state,
            )
        elif isinstance(decision, WaitingDecision):
            await executor.record_waiting(decision.reason)
        elif isinstance(decision, FailedDecision):
            await executor.record_failed(decision.error)
        elif isinstance(decision, DelegateDecision):
            run = state.runs.get(executor.run_id)
            current_rounds = run.delegation_rounds if run is not None else 0
            if current_rounds >= self._max_delegation_rounds:
                await executor.record_failed(
                    CrewAIFlowError(
                        code="max_delegation_rounds_exceeded",
                        message=(
                            f"Run exceeded max_delegation_rounds="
                            f"{self._max_delegation_rounds}"
                        ),
                    )
                )
                return
            executor._delegation_rounds = current_rounds + 1
            await executor.execute_delegations(
                items=decision.delegations,
                state=state,
                participants=self._snapshot_participants(executor._tools),
            )
        elif isinstance(decision, SynthesizeDecision):
            await self._handle_synthesize(
                executor=executor,
                state=state,
                decision=decision,
                msg=msg,
            )

    async def _handle_synthesize(
        self,
        *,
        executor: SideEffectExecutor,
        state: CrewAIFlowSessionState,
        decision: "SynthesizeDecision",
        msg: PlatformMessage,
    ) -> None:
        """Apply join policy + safety gates.

        - Join policy: ``all`` blocks until every delegation is replied;
          ``first`` requires at least one reply.
        - Tagged peer policy: any ``@handle`` token in the *original parent*
          message that resolves to a participant must have a recorded
          delegation before finalization.
        - Sequential chains: an upstream reply blocks finalization until the
          mapped downstream key has a delegation.
        - Buffered syntheses: when the policy is not satisfied, store the
          partial synthesis content keyed by ``msg.id`` so a future
          finalization can concatenate it.
        """
        run = state.runs.get(executor.run_id)
        pending_count = 0
        replied_count = 0
        if run is not None:
            for d in run.delegations:
                if d.status == CrewAIFlowDelegationStatus.PENDING:
                    pending_count += 1
                elif d.status == CrewAIFlowDelegationStatus.REPLIED:
                    replied_count += 1

        join_policy = run.join_policy if run is not None else self._join_policy
        join_satisfied = (
            run is None
            or (join_policy == CrewAIFlowJoinPolicy.FIRST and replied_count >= 1)
            or (join_policy == CrewAIFlowJoinPolicy.ALL and pending_count == 0)
        )

        # Tagged-peer gate.
        tagged_blocked: list[str] = []
        if self._tagged_peer_policy == "require_delegation_before_final":
            tagged_keys = (
                run.tagged_peer_keys if run is not None else executor._tagged_peer_keys
            )
            delegated_keys = (
                {d.target.normalized_key for d in run.delegations}
                if run is not None
                else set()
            )
            tagged_blocked = [k for k in tagged_keys if k not in delegated_keys]

        # Sequential-chain gate: if any upstream key has a reply but the
        # mapped downstream key has no delegation, block.
        sequential_blocked: list[str] = []
        if run is not None and self._sequential_chains:
            replied_keys = {
                d.target.normalized_key
                for d in run.delegations
                if d.status == CrewAIFlowDelegationStatus.REPLIED
            }
            delegated_keys = {d.target.normalized_key for d in run.delegations}
            for upstream_key, downstream_key in self._sequential_chains.items():
                if (
                    upstream_key in replied_keys
                    and downstream_key not in delegated_keys
                ):
                    sequential_blocked.append(downstream_key)

        if not join_satisfied or tagged_blocked or sequential_blocked:
            # Buffer the partial synthesis for later concatenation.
            await executor.record_buffered(
                source_message_id=msg.id,
                content=decision.content,
            )
            return

        # All gates clear. Concatenate buffered + current and finalize.
        buffered = "\n\n".join(
            b.content for b in (run.buffered_syntheses if run is not None else [])
        )
        final_content = (
            f"{buffered}\n\n{decision.content}".strip()
            if buffered
            else decision.content
        )
        await executor.execute_direct_response(
            content=final_content,
            mentions=decision.mentions,
            state=state,
        )

    @staticmethod
    def _extract_tagged_peer_keys(
        content: str,
        participants: list[CrewAIFlowParticipantSnapshot],
    ) -> list[str]:
        """Return normalized keys of room participants tagged via @handle."""
        import re

        if not content:
            return []
        tokens = re.findall(r"@([A-Za-z0-9_./-]+)", content)
        out: list[str] = []
        seen: set[str] = set()
        for raw in tokens:
            normalized = raw.strip().lower()
            if "/" in normalized:
                normalized = normalized.rsplit("/", 1)[-1]
            for p in participants:
                if p.normalized_key == normalized:
                    if normalized not in seen:
                        out.append(normalized)
                        seen.add(normalized)
                    break
        return out

    async def _handle_malformed(
        self,
        *,
        executor: SideEffectExecutor,
        state: CrewAIFlowSessionState,
        raw: Any,
        error: str,
    ) -> None:
        if (
            self._text_only_behavior == CrewAIFlowTextOnlyBehavior.FALLBACK_SEND
            and isinstance(raw, str)
        ):
            run = state.runs.get(executor.run_id)
            has_pending = any(
                d.status
                in (
                    CrewAIFlowDelegationStatus.PENDING,
                    CrewAIFlowDelegationStatus.RESERVED,
                )
                for d in (run.delegations if run is not None else [])
            )
            if not has_pending:
                await executor.execute_direct_response(
                    content=raw,
                    mentions=[],
                    state=state,
                )
                return
        await self._safe_record_failed(
            executor,
            CrewAIFlowError(
                code="malformed_flow_output",
                message=error,
            ),
        )

    async def _safe_record_failed(
        self,
        executor: SideEffectExecutor,
        error: CrewAIFlowError,
    ) -> None:
        try:
            await executor.record_failed(error)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to record failed task event for run %s", executor.run_id
            )

    def _match_reply_to_delegation(
        self,
        *,
        msg: PlatformMessage,
        state: CrewAIFlowSessionState,
        participants: list[CrewAIFlowParticipantSnapshot],
    ) -> tuple[str, str, "CrewAIFlowMetadata"] | _AmbiguousReply | None:
        """Try to match an inbound message to a pending delegation.

        Returns ``(run_id, delegation_id, run_metadata)`` on a unique match.
        Returns ``None`` for: known echoes, already-recorded replies, no
        candidate set, ambiguous matches (which also record a
        ``reply_ambiguous`` event side-effect).
        """
        from thenvoi.converters.crewai_flow import (
            CrewAIFlowAmbiguousIdentityError,
            normalize_participant_key,
        )

        # Compute sender's normalized key against the participant snapshot.
        try:
            sender_key = normalize_participant_key(
                msg.sender_id,
                participants=[
                    {"id": p.participant_id, "handle": p.handle, "name": p.handle}
                    for p in participants
                ],
            )
        except CrewAIFlowAmbiguousIdentityError:
            logger.warning(
                "Ambiguous sender identity for %s; treating as ambiguous", msg.sender_id
            )
            for run_id, run in state.runs.items():
                if run.status in (
                    CrewAIFlowRunStatus.FINALIZED,
                    CrewAIFlowRunStatus.FAILED,
                    CrewAIFlowRunStatus.INDETERMINATE,
                ):
                    continue
                for d in run.delegations:
                    if d.status in (
                        CrewAIFlowDelegationStatus.PENDING,
                        CrewAIFlowDelegationStatus.RESERVED,
                    ):
                        return _AmbiguousReply(
                            run_id=run_id,
                            parent_message_id=run.parent_message_id,
                            reason="ambiguous_sender_identity",
                        )
            return None

        # Build candidate set: pending delegations across active runs.
        candidates: list[tuple[str, str, "CrewAIFlowMetadata"]] = []
        for run_id, run in state.runs.items():
            if run.status in (
                CrewAIFlowRunStatus.FINALIZED,
                CrewAIFlowRunStatus.FAILED,
                CrewAIFlowRunStatus.INDETERMINATE,
            ):
                continue
            for d in run.delegations:
                # Rule 2: ignore the router's own delegation echoing back.
                if d.delegation_message_id == msg.id:
                    return None
                # Rule 3: ignore already-processed replies.
                if d.reply_message_id == msg.id:
                    return None
                if d.status not in (
                    CrewAIFlowDelegationStatus.PENDING,
                    CrewAIFlowDelegationStatus.RESERVED,
                ):
                    continue
                if d.target.normalized_key and d.target.normalized_key == sender_key:
                    candidates.append((run_id, d.delegation_id, run))

        if not candidates:
            return None

        # Rule 5: visible correlation token narrows the candidate set.
        token_hits = self._candidates_matching_token(msg.content, candidates)
        if token_hits is not None:
            if len(token_hits) == 1:
                return token_hits[0]
            run_id, _delegation_id, run = candidates[0]
            return _AmbiguousReply(
                run_id=run_id,
                parent_message_id=run.parent_message_id,
                reason="correlation_token_mismatch",
            )

        if len(candidates) == 1:
            return candidates[0]

        # Multiple candidates with no token: ambiguous.
        logger.warning(
            "Sender %s matches %d pending delegations without a correlation token",
            msg.sender_id,
            len(candidates),
        )
        run_id, _delegation_id, run = candidates[0]
        return _AmbiguousReply(
            run_id=run_id,
            parent_message_id=run.parent_message_id,
            reason="multiple_pending_delegations",
        )

    @staticmethod
    def _candidates_matching_token(
        content: str,
        candidates: list[tuple[str, str, "CrewAIFlowMetadata"]],
    ) -> list[tuple[str, str, "CrewAIFlowMetadata"]] | None:
        import hashlib
        import re

        if not content:
            return None
        match = re.search(r"\[ref:([0-9a-f]{8})\]", content)
        if not match:
            return None
        token = match.group(1)
        hits: list[tuple[str, str, "CrewAIFlowMetadata"]] = []
        for run_id, delegation_id, run in candidates:
            for d in run.delegations:
                if d.delegation_id != delegation_id:
                    continue
                expected = hashlib.sha256(d.side_effect_key.encode()).hexdigest()[:8]
                if expected == token:
                    hits.append((run_id, delegation_id, run))
        return hits

    async def _maybe_finalize_after_reply(
        self,
        *,
        tools: AgentToolsProtocol,
        room_id: str,
        run_id: str,
    ) -> None:
        """Re-load state for the run and finalize if join policy is satisfied.

        Phase 4 finalization is deferred to the next router turn for join
        policy ``all`` (the run waits until the Flow is invoked again with
        a fresh inbound that triggers a synthesize decision). This method
        only handles ``join_policy="first"`` where the first reply alone
        satisfies the policy — but even then the spec requires the Flow to
        produce the final synthesis content. So this method is a no-op for
        v1 and exists as the seam for future ``automatic finalization``.
        """
        # The reply is recorded; the next time the Flow runs (next turn or
        # explicit synthesize decision) it will see the replied state and
        # produce the synthesize. Phase 5 adds optional automatic
        # synthesize-after-join-satisfied behaviour.
        return None

    def _snapshot_participants(
        self, tools: AgentToolsProtocol
    ) -> list[CrewAIFlowParticipantSnapshot]:
        participants_attr = getattr(tools, "participants", None)
        raw = participants_attr() if callable(participants_attr) else participants_attr
        if not isinstance(raw, list):
            return []
        snapshot: list[CrewAIFlowParticipantSnapshot] = []
        for p in raw:
            if not isinstance(p, dict):
                continue
            participant_id = str(p.get("id") or "")
            handle = p.get("handle") or p.get("name") or ""
            if not participant_id:
                continue
            normalized = (handle or "").strip().lower().lstrip("@")
            if "/" in normalized:
                normalized = normalized.rsplit("/", 1)[-1]
            snapshot.append(
                CrewAIFlowParticipantSnapshot(
                    participant_id=participant_id,
                    handle=handle or None,
                    normalized_key=normalized,
                )
            )
        return snapshot

    def _build_inputs(
        self,
        *,
        msg: PlatformMessage,
        state: CrewAIFlowSessionState,
        participants: list[CrewAIFlowParticipantSnapshot],
        participants_msg: str | None,
        contacts_msg: str | None,
        room_id: str,
    ) -> dict[str, Any]:
        return {
            "room_id": room_id,
            "message": {
                "id": msg.id,
                "content": msg.content,
                "sender_id": msg.sender_id,
                "sender_name": getattr(msg, "sender_name", None),
                "sender_type": getattr(msg, "sender_type", None),
                "created_at": (
                    msg.created_at.isoformat()
                    if getattr(msg, "created_at", None) is not None
                    else None
                ),
            },
            "state": state.model_dump(mode="json"),
            "agent": {
                "name": self.agent_name,
                "description": self.agent_description,
            },
            "participants": [p.model_dump(mode="json") for p in participants],
            "participants_msg": participants_msg,
            "contacts_msg": contacts_msg,
        }


__all__ = [
    "CrewAIFlowAdapter",
    "CrewAIFlowRuntimeTools",
    "CrewAIFlowSessionState",
    "CrewAIFlowStateConverter",
    "CrewAIFlowStateSource",
    "CrewAIFlowSubCrewReporter",
    "DelegateDecision",
    "DelegateItem",
    "DirectResponseDecision",
    "FailedDecision",
    "FlowDecision",
    "HistoryCrewAIFlowStateSource",
    "RestCrewAIFlowStateSource",
    "SideEffectExecutor",
    "SynthesizeDecision",
    "WaitingDecision",
    "get_current_flow_runtime",
]
