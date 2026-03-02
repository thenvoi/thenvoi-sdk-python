"""Tests for examples/pydantic_ai/*.py scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _load_module(filename: str) -> ModuleType:
    module_name = f"examples_pydantic_ai_{filename.replace('.', '_')}_test"
    spec = importlib.util.spec_from_file_location(
        module_name,
        Path("examples/pydantic_ai") / filename,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load examples/pydantic_ai/{filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_basic_agent_main_bootstraps_and_runs() -> None:
    module = _load_module("01_basic_agent.py")
    adapter_instance = object()
    mock_adapter_cls = MagicMock(return_value=adapter_instance)
    session = SimpleNamespace(agent=SimpleNamespace(run=AsyncMock()))
    mock_bootstrap = MagicMock(return_value=session)

    module.PydanticAIAdapter = mock_adapter_cls
    module.bootstrap_agent = mock_bootstrap

    await module.main()

    mock_adapter_cls.assert_called_once_with(
        model="openai:gpt-4o",
        custom_section="You are a helpful assistant. Be concise and friendly.",
    )
    mock_bootstrap.assert_called_once_with(agent_key="pydantic_agent", adapter=adapter_instance)
    session.agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_custom_instructions_main_bootstraps_and_runs() -> None:
    module = _load_module("02_custom_instructions.py")
    adapter_instance = object()
    mock_adapter_cls = MagicMock(return_value=adapter_instance)
    session = SimpleNamespace(agent=SimpleNamespace(run=AsyncMock()))
    mock_bootstrap = MagicMock(return_value=session)

    module.PydanticAIAdapter = mock_adapter_cls
    module.bootstrap_agent = mock_bootstrap

    await module.main()

    mock_adapter_cls.assert_called_once_with(
        model="anthropic:claude-3-5-sonnet-latest",
        custom_section=module.CUSTOM_PROMPT,
    )
    mock_bootstrap.assert_called_once_with(agent_key="support_agent", adapter=adapter_instance)
    session.agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_tom_agent_requires_ws_url(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module("03_tom_agent.py")
    monkeypatch.delenv("THENVOI_WS_URL", raising=False)
    monkeypatch.setenv("THENVOI_REST_URL", "https://rest.example")
    monkeypatch.setattr(module, "load_dotenv", MagicMock())

    with pytest.raises(ValueError, match="THENVOI_WS_URL environment variable is required"):
        await module.main()


@pytest.mark.asyncio
async def test_tom_agent_creates_adapter_and_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module("03_tom_agent.py")
    monkeypatch.setenv("THENVOI_WS_URL", "wss://ws.example")
    monkeypatch.setenv("THENVOI_REST_URL", "https://rest.example")

    monkeypatch.setattr(module, "load_dotenv", MagicMock())
    monkeypatch.setattr(module, "load_agent_config", MagicMock(return_value=("tom-id", "tom-key")))
    monkeypatch.setattr(module, "generate_tom_prompt", MagicMock(return_value="tom-prompt"))
    adapter_instance = object()
    mock_adapter_cls = MagicMock(return_value=adapter_instance)
    runtime_agent = SimpleNamespace(run=AsyncMock())
    mock_create = MagicMock(return_value=runtime_agent)
    monkeypatch.setattr(module, "PydanticAIAdapter", mock_adapter_cls)
    monkeypatch.setattr(module.Agent, "create", mock_create)

    await module.main()

    mock_adapter_cls.assert_called_once_with(
        model="openai:gpt-4o",
        custom_section="tom-prompt",
    )
    mock_create.assert_called_once_with(
        adapter=adapter_instance,
        agent_id="tom-id",
        api_key="tom-key",
        ws_url="wss://ws.example",
        rest_url="https://rest.example",
    )
    runtime_agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_jerry_agent_main_bootstraps_and_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module("04_jerry_agent.py")
    monkeypatch.setattr(module, "generate_jerry_prompt", MagicMock(return_value="jerry-prompt"))
    adapter_instance = object()
    mock_adapter_cls = MagicMock(return_value=adapter_instance)
    session = SimpleNamespace(agent=SimpleNamespace(run=AsyncMock()))
    mock_bootstrap = MagicMock(return_value=session)
    monkeypatch.setattr(module, "PydanticAIAdapter", mock_adapter_cls)
    monkeypatch.setattr(module, "bootstrap_agent", mock_bootstrap)

    await module.main()

    mock_adapter_cls.assert_called_once_with(
        model="openai:gpt-4o",
        custom_section="jerry-prompt",
    )
    mock_bootstrap.assert_called_once_with(agent_key="jerry_agent", adapter=adapter_instance)
    session.agent.run.assert_awaited_once()
