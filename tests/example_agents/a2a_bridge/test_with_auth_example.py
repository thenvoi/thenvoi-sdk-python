"""Tests for examples/a2a_bridge/02_with_auth.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _load_example_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "examples_a2a_bridge_with_auth_test",
        Path("examples/a2a_bridge/02_with_auth.py"),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load examples/a2a_bridge/02_with_auth.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_main_builds_adapter_without_auth_when_no_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_example_module()
    monkeypatch.delenv("A2A_API_KEY", raising=False)
    monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("A2A_AGENT_URL", raising=False)

    mock_setup_logging = MagicMock()
    mock_auth_cls = MagicMock()
    adapter_instance = object()
    mock_adapter_cls = MagicMock(return_value=adapter_instance)
    session = SimpleNamespace(agent=SimpleNamespace(run=AsyncMock()))
    mock_bootstrap = MagicMock(return_value=session)

    monkeypatch.setattr(module, "setup_logging", mock_setup_logging)
    monkeypatch.setattr(module, "A2AAuth", mock_auth_cls)
    monkeypatch.setattr(module, "A2AAdapter", mock_adapter_cls)
    monkeypatch.setattr(module, "bootstrap_agent", mock_bootstrap)

    await module.main()

    mock_setup_logging.assert_called_once()
    mock_auth_cls.assert_not_called()
    mock_adapter_cls.assert_called_once_with(
        remote_url="http://localhost:10000",
        auth=None,
        streaming=True,
    )
    mock_bootstrap.assert_called_once_with(agent_key="a2a_agent", adapter=adapter_instance)
    session.agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_main_builds_adapter_with_auth_when_credentials_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_example_module()
    monkeypatch.setenv("A2A_AGENT_URL", "http://a2a.example")
    monkeypatch.setenv("A2A_API_KEY", "api-key")
    monkeypatch.setenv("A2A_BEARER_TOKEN", "bearer-token")

    mock_setup_logging = MagicMock()
    auth_instance = object()
    mock_auth_cls = MagicMock(return_value=auth_instance)
    adapter_instance = object()
    mock_adapter_cls = MagicMock(return_value=adapter_instance)
    session = SimpleNamespace(agent=SimpleNamespace(run=AsyncMock()))
    mock_bootstrap = MagicMock(return_value=session)
    mock_logger = MagicMock()

    monkeypatch.setattr(module, "setup_logging", mock_setup_logging)
    monkeypatch.setattr(module, "A2AAuth", mock_auth_cls)
    monkeypatch.setattr(module, "A2AAdapter", mock_adapter_cls)
    monkeypatch.setattr(module, "bootstrap_agent", mock_bootstrap)
    monkeypatch.setattr(module, "logger", mock_logger)

    await module.main()

    mock_auth_cls.assert_called_once_with(
        api_key="api-key",
        bearer_token="bearer-token",
    )
    mock_adapter_cls.assert_called_once_with(
        remote_url="http://a2a.example",
        auth=auth_instance,
        streaming=True,
    )
    session.agent.run.assert_awaited_once()
    mock_logger.info.assert_any_call("Using authentication for A2A agent")
