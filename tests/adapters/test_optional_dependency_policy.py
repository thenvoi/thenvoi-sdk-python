"""Contract tests for optional adapter dependency handling."""

from __future__ import annotations

import pytest

from thenvoi.adapters import claude_sdk, crewai, langgraph
from thenvoi.adapters.optional_dependencies import ensure_optional_dependency

pytestmark = pytest.mark.contract_gate


def test_optional_dependency_helper_formats_consistent_message() -> None:
    with pytest.raises(ImportError) as exc_info:
        ensure_optional_dependency(
            ImportError("missing"),
            package="sample-pkg",
            integration="SampleAdapter",
            install_commands=("pip install sample-pkg", "uv add sample-pkg"),
        )

    message = str(exc_info.value)
    assert "sample-pkg is required for SampleAdapter integrations." in message
    assert "Install with: pip install sample-pkg" in message
    assert "Install with: uv add sample-pkg" in message


def test_claude_sdk_adapter_defers_missing_dependency_to_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(claude_sdk, "_CLAUDE_SDK_IMPORT_ERROR", ImportError("missing"))

    with pytest.raises(ImportError, match="claude-agent-sdk is required"):
        claude_sdk.ClaudeSDKAdapter()


def test_langgraph_adapter_defers_missing_dependency_to_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(langgraph, "_LANGGRAPH_IMPORT_ERROR", ImportError("missing"))

    with pytest.raises(ImportError, match="langgraph is required"):
        langgraph.LangGraphAdapter(graph=object())


def test_crewai_adapter_defers_missing_dependency_to_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(crewai, "_CREWAI_IMPORT_ERROR", ImportError("missing"))

    with pytest.raises(ImportError, match="crewai is required"):
        crewai.CrewAIAdapter()
