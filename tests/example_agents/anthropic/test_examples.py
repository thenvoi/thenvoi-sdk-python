"""Tests for examples/anthropic/*.py scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _load_module(filename: str) -> ModuleType:
    module_name = f"examples_anthropic_{filename.replace('.', '_')}_test"
    spec = importlib.util.spec_from_file_location(
        module_name,
        Path("examples/anthropic") / filename,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load examples/anthropic/{filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_basic_agent_main_bootstraps_session_and_runs() -> None:
    module = _load_module("01_basic_agent.py")
    adapter_instance = object()
    mock_adapter_cls = MagicMock(return_value=adapter_instance)
    session = SimpleNamespace(agent=SimpleNamespace(run=AsyncMock()))
    mock_bootstrap = MagicMock(return_value=session)

    module.AnthropicAdapter = mock_adapter_cls
    module.bootstrap_agent = mock_bootstrap

    await module.main()

    mock_adapter_cls.assert_called_once_with(
        model="claude-sonnet-4-5-20250929",
        custom_section="You are a helpful assistant. Be concise and friendly.",
    )
    mock_bootstrap.assert_called_once_with(
        agent_key="anthropic_agent",
        adapter=adapter_instance,
    )
    session.agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_custom_instructions_requires_ws_url(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module("02_custom_instructions.py")
    monkeypatch.delenv("THENVOI_WS_URL", raising=False)
    monkeypatch.setenv("THENVOI_REST_URL", "https://rest.example")
    monkeypatch.setattr(module, "load_dotenv", MagicMock())

    with pytest.raises(ValueError, match="THENVOI_WS_URL environment variable is required"):
        await module.main()


@pytest.mark.asyncio
async def test_custom_instructions_creates_agent_and_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module("02_custom_instructions.py")
    monkeypatch.setenv("THENVOI_WS_URL", "wss://ws.example")
    monkeypatch.setenv("THENVOI_REST_URL", "https://rest.example")

    mock_load_dotenv = MagicMock()
    mock_load_agent_config = MagicMock(return_value=("agent-1", "api-key"))
    adapter_instance = object()
    mock_adapter_cls = MagicMock(return_value=adapter_instance)
    runtime_agent = SimpleNamespace(run=AsyncMock())
    mock_create = MagicMock(return_value=runtime_agent)

    monkeypatch.setattr(module, "load_dotenv", mock_load_dotenv)
    monkeypatch.setattr(module, "load_agent_config", mock_load_agent_config)
    monkeypatch.setattr(module, "AnthropicAdapter", mock_adapter_cls)
    monkeypatch.setattr(module.Agent, "create", mock_create)

    await module.main()

    mock_load_dotenv.assert_called_once()
    mock_load_agent_config.assert_called_once_with("support_agent")
    mock_adapter_cls.assert_called_once_with(
        model="claude-sonnet-4-5-20250929",
        custom_section=module.CUSTOM_PROMPT,
        enable_execution_reporting=True,
    )
    mock_create.assert_called_once_with(
        adapter=adapter_instance,
        agent_id="agent-1",
        api_key="api-key",
        ws_url="wss://ws.example",
        rest_url="https://rest.example",
    )
    runtime_agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_tom_agent_uses_generated_prompt_and_runs() -> None:
    module = _load_module("03_tom_agent.py")
    mock_generate_tom_prompt = MagicMock(return_value="tom-prompt")
    adapter_instance = object()
    mock_adapter_cls = MagicMock(return_value=adapter_instance)
    session = SimpleNamespace(agent=SimpleNamespace(run=AsyncMock()))
    mock_bootstrap = MagicMock(return_value=session)

    module.generate_tom_prompt = mock_generate_tom_prompt
    module.AnthropicAdapter = mock_adapter_cls
    module.bootstrap_agent = mock_bootstrap

    await module.main()

    mock_generate_tom_prompt.assert_called_once_with("Tom")
    mock_adapter_cls.assert_called_once_with(
        model="claude-sonnet-4-5-20250929",
        custom_section="tom-prompt",
    )
    mock_bootstrap.assert_called_once_with(agent_key="tom_agent", adapter=adapter_instance)
    session.agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_jerry_agent_requires_rest_url(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module("04_jerry_agent.py")
    monkeypatch.setenv("THENVOI_WS_URL", "wss://ws.example")
    monkeypatch.delenv("THENVOI_REST_URL", raising=False)
    monkeypatch.setattr(module, "load_dotenv", MagicMock())

    with pytest.raises(ValueError, match="THENVOI_REST_URL environment variable is required"):
        await module.main()


@pytest.mark.asyncio
async def test_jerry_agent_creates_adapter_and_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module("04_jerry_agent.py")
    monkeypatch.setenv("THENVOI_WS_URL", "wss://ws.example")
    monkeypatch.setenv("THENVOI_REST_URL", "https://rest.example")

    mock_load_dotenv = MagicMock()
    mock_generate_jerry_prompt = MagicMock(return_value="jerry-prompt")
    mock_load_agent_config = MagicMock(return_value=("jerry-id", "jerry-key"))
    adapter_instance = object()
    mock_adapter_cls = MagicMock(return_value=adapter_instance)
    runtime_agent = SimpleNamespace(run=AsyncMock())
    mock_create = MagicMock(return_value=runtime_agent)

    monkeypatch.setattr(module, "load_dotenv", mock_load_dotenv)
    monkeypatch.setattr(module, "generate_jerry_prompt", mock_generate_jerry_prompt)
    monkeypatch.setattr(module, "load_agent_config", mock_load_agent_config)
    monkeypatch.setattr(module, "AnthropicAdapter", mock_adapter_cls)
    monkeypatch.setattr(module.Agent, "create", mock_create)

    await module.main()

    mock_load_dotenv.assert_called_once()
    mock_generate_jerry_prompt.assert_called_once_with("Jerry")
    mock_load_agent_config.assert_called_once_with("jerry_agent")
    mock_adapter_cls.assert_called_once_with(
        model="claude-sonnet-4-5-20250929",
        custom_section="jerry-prompt",
    )
    mock_create.assert_called_once_with(
        adapter=adapter_instance,
        agent_id="jerry-id",
        api_key="jerry-key",
        ws_url="wss://ws.example",
        rest_url="https://rest.example",
    )
    runtime_agent.run.assert_awaited_once()
