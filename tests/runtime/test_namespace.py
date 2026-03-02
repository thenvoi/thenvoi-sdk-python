"""Compatibility tests for the ``thenvoi.runtime`` namespace exports."""

from __future__ import annotations

import pytest

from thenvoi.runtime import AgentRuntime, ExecutionContext, RoomPresence, SessionConfig
from thenvoi.runtime.execution import ExecutionContext as ExecutionContextFromExecution
from thenvoi.runtime.presence import RoomPresence as RoomPresenceFromPresence
from thenvoi.runtime.runtime import AgentRuntime as AgentRuntimeFromRuntime
from thenvoi.runtime.types import SessionConfig as SessionConfigFromTypes

pytestmark = pytest.mark.contract_gate


def test_runtime_namespace_exports_lazy_symbols() -> None:
    assert AgentRuntime is AgentRuntimeFromRuntime
    assert ExecutionContext is ExecutionContextFromExecution
    assert RoomPresence is RoomPresenceFromPresence
    assert SessionConfig is SessionConfigFromTypes
