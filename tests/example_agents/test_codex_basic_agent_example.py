"""Tests for examples/codex/01_basic_agent.py."""

from __future__ import annotations

import importlib
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import examples.codex.basic_agent as codex_basic_agent_module


def _load_example_module() -> ModuleType:
    return importlib.reload(codex_basic_agent_module)


def test_env_bool_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_example_module()

    monkeypatch.delenv("FLAG", raising=False)
    assert module._env_bool("FLAG", default=True) is True

    monkeypatch.setenv("FLAG", "true")
    assert module._env_bool("FLAG", default=False) is True

    monkeypatch.setenv("FLAG", "0")
    assert module._env_bool("FLAG", default=True) is False


@pytest.mark.asyncio
async def test_main_rejects_invalid_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_example_module()
    monkeypatch.setenv("CODEX_TRANSPORT", "invalid")

    with pytest.raises(ValueError, match="CODEX_TRANSPORT must be 'stdio' or 'ws'"):
        await module.main()


@pytest.mark.asyncio
async def test_main_builds_adapter_and_runs_session(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_example_module()
    monkeypatch.setenv("AGENT_KEY", "agent-key")
    monkeypatch.setenv("CODEX_TRANSPORT", "ws")
    monkeypatch.setenv("CODEX_WS_URL", "ws://codex.example")
    monkeypatch.setenv("CODEX_MODEL", "gpt-test")
    monkeypatch.setenv("CODEX_APPROVAL_POLICY", "never")
    monkeypatch.setenv("CODEX_APPROVAL_MODE", "manual")
    monkeypatch.setenv("CODEX_TURN_TASK_MARKERS", "true")
    monkeypatch.setenv("CODEX_ROLE", "planner")

    mock_warning = MagicMock()
    mock_config_cls = MagicMock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
    adapter = object()
    mock_adapter_cls = MagicMock(return_value=adapter)
    session = SimpleNamespace(agent=SimpleNamespace(run=AsyncMock()))
    mock_bootstrap = MagicMock(return_value=session)

    monkeypatch.setattr(module.logger, "warning", mock_warning)
    monkeypatch.setattr(module, "CodexAdapterConfig", mock_config_cls)
    monkeypatch.setattr(module, "CodexAdapter", mock_adapter_cls)
    monkeypatch.setattr(module, "bootstrap_agent", mock_bootstrap)

    await module.main()

    config_kwargs = mock_config_cls.call_args.kwargs
    assert config_kwargs["transport"] == "ws"
    assert config_kwargs["codex_ws_url"] == "ws://codex.example"
    assert config_kwargs["model"] == "gpt-test"
    assert config_kwargs["approval_policy"] == "never"
    assert config_kwargs["approval_mode"] == "manual"
    assert config_kwargs["emit_turn_task_markers"] is True
    assert config_kwargs["custom_section"].startswith("You are a helpful assistant.")

    mock_warning.assert_called_once()
    mock_adapter_cls.assert_called_once()
    mock_bootstrap.assert_called_once_with(agent_key="agent-key", adapter=adapter)
    session.agent.run.assert_awaited_once()
