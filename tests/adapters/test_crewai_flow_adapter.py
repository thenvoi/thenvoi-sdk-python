"""Phase 2 tests for ``CrewAIFlowAdapter`` constructor and lifecycle."""

from __future__ import annotations

import sys
from datetime import timedelta
from unittest.mock import MagicMock

import pytest


# Mock crewai imports BEFORE importing the adapter module so the optional
# dependency guard doesn't pin us to a real install.
@pytest.fixture(autouse=True)
def _mock_crewai(monkeypatch: pytest.MonkeyPatch):
    fake_module = MagicMock()
    fake_flow_module = MagicMock()
    fake_flow_module.Flow = type("Flow", (), {})
    fake_flow_module.start = MagicMock()
    fake_flow_module.listen = MagicMock()
    fake_flow_module.router = MagicMock()
    fake_flow_module.and_ = MagicMock()
    fake_flow_module.or_ = MagicMock()
    fake_module.flow = MagicMock()
    fake_module.flow.flow = fake_flow_module

    monkeypatch.setitem(sys.modules, "crewai", fake_module)
    monkeypatch.setitem(sys.modules, "crewai.flow", fake_module.flow)
    monkeypatch.setitem(sys.modules, "crewai.flow.flow", fake_flow_module)
    yield


from thenvoi.adapters.crewai_flow import (  # noqa: E402
    CrewAIFlowAdapter,
    HistoryCrewAIFlowStateSource,
    RestCrewAIFlowStateSource,
)
from thenvoi.core.exceptions import ThenvoiConfigError  # noqa: E402


def _factory():
    return MagicMock()


# ---------------------------------------------------------------------------
# Constructor defaults and validation
# ---------------------------------------------------------------------------


class TestInit:
    def test_init_defaults(self) -> None:
        adapter = CrewAIFlowAdapter(flow_factory=_factory)
        assert isinstance(adapter._state_source, RestCrewAIFlowStateSource)
        assert adapter._max_delegation_rounds == 4
        assert adapter._max_run_age == timedelta(days=7)
        assert adapter.metadata_namespace == ""

    def test_init_does_not_call_flow_factory(self) -> None:
        called = []

        def factory():
            called.append(1)
            return MagicMock()

        CrewAIFlowAdapter(flow_factory=factory)
        assert called == []

    def test_init_accepts_history_state_source(self) -> None:
        adapter = CrewAIFlowAdapter(
            flow_factory=_factory,
            state_source=HistoryCrewAIFlowStateSource(acknowledge_test_only=True),
        )
        assert isinstance(adapter._state_source, HistoryCrewAIFlowStateSource)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_flow_factory_must_be_callable(self) -> None:
        with pytest.raises(ThenvoiConfigError):
            CrewAIFlowAdapter(flow_factory="not callable")  # type: ignore[arg-type]

    def test_state_source_must_have_load_task_events(self) -> None:
        with pytest.raises(ThenvoiConfigError):
            CrewAIFlowAdapter(
                flow_factory=_factory,
                state_source=object(),  # type: ignore[arg-type]
            )

    def test_join_policy_validation(self) -> None:
        with pytest.raises(ThenvoiConfigError):
            CrewAIFlowAdapter(
                flow_factory=_factory,
                join_policy="some_other",  # type: ignore[arg-type]
            )

    def test_metadata_namespace_must_be_non_empty(self) -> None:
        with pytest.raises(ThenvoiConfigError):
            CrewAIFlowAdapter(flow_factory=_factory, metadata_namespace="")

    def test_max_delegation_rounds_bounds(self) -> None:
        with pytest.raises(ThenvoiConfigError):
            CrewAIFlowAdapter(flow_factory=_factory, max_delegation_rounds=0)
        with pytest.raises(ThenvoiConfigError):
            CrewAIFlowAdapter(flow_factory=_factory, max_delegation_rounds=21)

    def test_max_run_age_must_be_positive_timedelta(self) -> None:
        with pytest.raises(ThenvoiConfigError):
            CrewAIFlowAdapter(flow_factory=_factory, max_run_age=timedelta(0))
        with pytest.raises(ThenvoiConfigError):
            CrewAIFlowAdapter(
                flow_factory=_factory, max_run_age=timedelta(seconds=-1)
            )
        with pytest.raises(ThenvoiConfigError):
            CrewAIFlowAdapter(flow_factory=_factory, max_run_age=86400)  # type: ignore[arg-type]

    def test_text_only_behavior_validation(self) -> None:
        with pytest.raises(ThenvoiConfigError):
            CrewAIFlowAdapter(
                flow_factory=_factory,
                text_only_behavior="bogus",  # type: ignore[arg-type]
            )

    def test_tagged_peer_policy_validation(self) -> None:
        with pytest.raises(ThenvoiConfigError):
            CrewAIFlowAdapter(
                flow_factory=_factory,
                tagged_peer_policy="something_else",  # type: ignore[arg-type]
            )

    def test_sequential_chains_must_be_str_to_str(self) -> None:
        with pytest.raises(ThenvoiConfigError):
            CrewAIFlowAdapter(
                flow_factory=_factory,
                sequential_chains={"k": 123},  # type: ignore[dict-item]
            )


# ---------------------------------------------------------------------------
# on_started / namespace resolution
# ---------------------------------------------------------------------------


class TestOnStartedAndNamespace:
    @pytest.mark.asyncio
    async def test_default_namespace_resolves_with_agent_id(self) -> None:
        adapter = CrewAIFlowAdapter(flow_factory=_factory)
        await adapter.on_started("agent-A", "desc")
        assert adapter.metadata_namespace == "crewai_flow:agent-A"

    @pytest.mark.asyncio
    async def test_two_adapters_get_distinct_namespaces(self) -> None:
        a = CrewAIFlowAdapter(flow_factory=_factory)
        b = CrewAIFlowAdapter(flow_factory=_factory)
        await a.on_started("agent-X", "")
        await b.on_started("agent-Y", "")
        assert a.metadata_namespace != b.metadata_namespace

    @pytest.mark.asyncio
    async def test_explicit_namespace_overrides_default(self) -> None:
        adapter = CrewAIFlowAdapter(
            flow_factory=_factory, metadata_namespace="custom_ns"
        )
        await adapter.on_started("agent-A", "")
        assert adapter.metadata_namespace == "custom_ns"

    @pytest.mark.asyncio
    async def test_on_started_does_not_call_flow_factory(self) -> None:
        called = []

        def factory():
            called.append(1)
            return MagicMock()

        adapter = CrewAIFlowAdapter(flow_factory=factory)
        await adapter.on_started("agent-A", "")
        assert called == []


# ---------------------------------------------------------------------------
# on_cleanup
# ---------------------------------------------------------------------------


class TestOnCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_removes_only_target_room(self) -> None:
        adapter = CrewAIFlowAdapter(flow_factory=_factory)
        # Prime locks for two rooms.
        adapter._get_room_lock("room-A")
        adapter._get_room_lock("room-B")
        assert "room-A" in adapter._room_locks
        assert "room-B" in adapter._room_locks

        await adapter.on_cleanup("room-A")
        assert "room-A" not in adapter._room_locks
        assert "room-B" in adapter._room_locks
