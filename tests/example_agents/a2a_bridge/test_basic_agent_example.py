"""Tests for examples/a2a_bridge/01_basic_agent.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _load_example_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    setup_logging_stub = ModuleType("setup_logging")
    setup_logging_stub.setup_logging = MagicMock()
    monkeypatch.setitem(sys.modules, "setup_logging", setup_logging_stub)

    spec = importlib.util.spec_from_file_location(
        "examples_a2a_bridge_basic_agent_test",
        Path("examples/a2a_bridge/01_basic_agent.py"),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load examples/a2a_bridge/01_basic_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_main_requires_thenvoi_ws_url(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_example_module(monkeypatch)
    monkeypatch.delenv("THENVOI_WS_URL", raising=False)
    monkeypatch.setenv("THENVOI_REST_URL", "https://rest.example")
    monkeypatch.setattr(module, "load_dotenv", MagicMock())

    with pytest.raises(ValueError, match="THENVOI_WS_URL environment variable is required"):
        await module.main()


@pytest.mark.asyncio
async def test_main_requires_thenvoi_rest_url(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_example_module(monkeypatch)
    monkeypatch.setenv("THENVOI_WS_URL", "wss://ws.example")
    monkeypatch.delenv("THENVOI_REST_URL", raising=False)
    monkeypatch.setattr(module, "load_dotenv", MagicMock())

    with pytest.raises(ValueError, match="THENVOI_REST_URL environment variable is required"):
        await module.main()


@pytest.mark.asyncio
async def test_main_creates_agent_and_runs_with_configured_a2a_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_example_module(monkeypatch)
    monkeypatch.setenv("THENVOI_WS_URL", "wss://ws.example")
    monkeypatch.setenv("THENVOI_REST_URL", "https://rest.example")
    monkeypatch.setenv("A2A_AGENT_URL", "http://a2a.example")

    mock_load_dotenv = MagicMock()
    mock_load_agent_config = MagicMock(return_value=("agent-123", "api-key"))
    adapter_instance = object()
    mock_adapter_cls = MagicMock(return_value=adapter_instance)
    runtime_agent = SimpleNamespace(run=AsyncMock())
    mock_agent_create = MagicMock(return_value=runtime_agent)

    monkeypatch.setattr(module, "load_dotenv", mock_load_dotenv)
    monkeypatch.setattr(module, "load_agent_config", mock_load_agent_config)
    monkeypatch.setattr(module, "A2AAdapter", mock_adapter_cls)
    monkeypatch.setattr(module.Agent, "create", mock_agent_create)

    await module.main()

    mock_load_dotenv.assert_called_once()
    mock_load_agent_config.assert_called_once_with("a2a_agent")
    mock_adapter_cls.assert_called_once_with(
        remote_url="http://a2a.example",
        streaming=True,
    )
    mock_agent_create.assert_called_once_with(
        adapter=adapter_instance,
        agent_id="agent-123",
        api_key="api-key",
        ws_url="wss://ws.example",
        rest_url="https://rest.example",
    )
    runtime_agent.run.assert_awaited_once()
