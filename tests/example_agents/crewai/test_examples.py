"""Tests for examples/crewai/*.py scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _load_module(filename: str) -> ModuleType:
    module_name = f"examples_crewai_{filename.replace('.', '_')}_test"
    spec = importlib.util.spec_from_file_location(
        module_name,
        Path("examples/crewai") / filename,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load examples/crewai/{filename}")
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

    module.CrewAIAdapter = mock_adapter_cls
    module.bootstrap_agent = mock_bootstrap

    await module.main()

    mock_adapter_cls.assert_called_once_with(
        model="gpt-4o",
        custom_section="You are a helpful assistant. Be concise and friendly.",
    )
    mock_bootstrap.assert_called_once_with(agent_key="crewai_agent", adapter=adapter_instance)
    session.agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_role_based_agent_requires_ws_url(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module("02_role_based_agent.py")
    monkeypatch.delenv("THENVOI_WS_URL", raising=False)
    monkeypatch.setenv("THENVOI_REST_URL", "https://rest.example")
    monkeypatch.setattr(module, "load_dotenv", MagicMock())

    with pytest.raises(ValueError, match="THENVOI_WS_URL environment variable is required"):
        await module.main()


@pytest.mark.asyncio
async def test_role_based_agent_creates_adapter_and_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module("02_role_based_agent.py")
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
    monkeypatch.setattr(module, "CrewAIAdapter", mock_adapter_cls)
    monkeypatch.setattr(module.Agent, "create", mock_create)

    await module.main()

    mock_load_agent_config.assert_called_once_with("crewai_agent")
    mock_adapter_cls.assert_called_once()
    assert mock_adapter_cls.call_args.kwargs["role"] == "Research Assistant"
    assert mock_adapter_cls.call_args.kwargs["enable_execution_reporting"] is True
    assert mock_adapter_cls.call_args.kwargs["verbose"] is True
    mock_create.assert_called_once_with(
        adapter=adapter_instance,
        agent_id="agent-1",
        api_key="api-key",
        ws_url="wss://ws.example",
        rest_url="https://rest.example",
    )
    runtime_agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_coordinator_agent_creates_adapter_and_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module("03_coordinator_agent.py")
    monkeypatch.setenv("THENVOI_WS_URL", "wss://ws.example")
    monkeypatch.setenv("THENVOI_REST_URL", "https://rest.example")

    monkeypatch.setattr(module, "load_dotenv", MagicMock())
    monkeypatch.setattr(module, "load_agent_config", MagicMock(return_value=("coord-id", "coord-key")))
    adapter_instance = object()
    mock_adapter_cls = MagicMock(return_value=adapter_instance)
    runtime_agent = SimpleNamespace(run=AsyncMock())
    mock_create = MagicMock(return_value=runtime_agent)
    monkeypatch.setattr(module, "CrewAIAdapter", mock_adapter_cls)
    monkeypatch.setattr(module.Agent, "create", mock_create)

    await module.main()

    assert mock_adapter_cls.call_args.kwargs["role"] == "Team Coordinator"
    assert "thenvoi_lookup_peers" in mock_adapter_cls.call_args.kwargs["backstory"]
    assert mock_adapter_cls.call_args.kwargs["enable_execution_reporting"] is True
    assert mock_adapter_cls.call_args.kwargs["verbose"] is True
    runtime_agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_research_crew_requires_role_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module("04_research_crew.py")
    monkeypatch.setattr(module.sys, "argv", ["04_research_crew.py"])
    monkeypatch.setattr(module, "load_dotenv", MagicMock())

    with pytest.raises(ValueError, match="Usage: uv run examples/crewai/04_research_crew.py <role>"):
        await module.main()


@pytest.mark.asyncio
async def test_research_crew_rejects_unknown_role(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module("04_research_crew.py")
    monkeypatch.setattr(module.sys, "argv", ["04_research_crew.py", "invalid"])
    monkeypatch.setattr(module, "load_dotenv", MagicMock())

    with pytest.raises(ValueError, match="Unknown role: invalid"):
        await module.main()


@pytest.mark.asyncio
async def test_research_crew_builds_selected_member_and_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module("04_research_crew.py")
    monkeypatch.setattr(module.sys, "argv", ["04_research_crew.py", "researcher"])
    monkeypatch.setenv("THENVOI_WS_URL", "wss://ws.example")
    monkeypatch.setenv("THENVOI_REST_URL", "https://rest.example")

    member = module.CREW_MEMBERS["researcher"]
    monkeypatch.setattr(module, "load_dotenv", MagicMock())
    monkeypatch.setattr(module, "load_agent_config", MagicMock(return_value=("res-id", "res-key")))
    adapter_instance = object()
    mock_adapter_cls = MagicMock(return_value=adapter_instance)
    runtime_agent = SimpleNamespace(run=AsyncMock())
    mock_create = MagicMock(return_value=runtime_agent)
    monkeypatch.setattr(module, "CrewAIAdapter", mock_adapter_cls)
    monkeypatch.setattr(module.Agent, "create", mock_create)

    await module.main()

    mock_adapter_cls.assert_called_once_with(
        model="gpt-4o",
        role=member["role"],
        goal=member["goal"],
        backstory=member["backstory"],
        custom_section=member["custom_section"],
        enable_execution_reporting=True,
    )
    mock_create.assert_called_once_with(
        adapter=adapter_instance,
        agent_id="res-id",
        api_key="res-key",
        ws_url="wss://ws.example",
        rest_url="https://rest.example",
    )
    runtime_agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_tom_agent_uses_generated_prompt_and_runs() -> None:
    module = _load_module("05_tom_agent.py")
    mock_generate_tom_prompt = MagicMock(return_value="tom-prompt")
    adapter_instance = object()
    mock_adapter_cls = MagicMock(return_value=adapter_instance)
    session = SimpleNamespace(agent=SimpleNamespace(run=AsyncMock()))
    mock_bootstrap = MagicMock(return_value=session)

    module.generate_tom_prompt = mock_generate_tom_prompt
    module.CrewAIAdapter = mock_adapter_cls
    module.bootstrap_agent = mock_bootstrap

    await module.main()

    mock_generate_tom_prompt.assert_called_once_with("Tom")
    mock_adapter_cls.assert_called_once_with(model="gpt-4o", custom_section="tom-prompt")
    mock_bootstrap.assert_called_once_with(agent_key="tom_agent", adapter=adapter_instance)
    session.agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_jerry_agent_requires_rest_url(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module("06_jerry_agent.py")
    monkeypatch.setenv("THENVOI_WS_URL", "wss://ws.example")
    monkeypatch.delenv("THENVOI_REST_URL", raising=False)
    monkeypatch.setattr(module, "load_dotenv", MagicMock())

    with pytest.raises(ValueError, match="THENVOI_REST_URL environment variable is required"):
        await module.main()


@pytest.mark.asyncio
async def test_jerry_agent_creates_adapter_and_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module("06_jerry_agent.py")
    monkeypatch.setenv("THENVOI_WS_URL", "wss://ws.example")
    monkeypatch.setenv("THENVOI_REST_URL", "https://rest.example")

    monkeypatch.setattr(module, "load_dotenv", MagicMock())
    monkeypatch.setattr(module, "generate_jerry_prompt", MagicMock(return_value="jerry-prompt"))
    monkeypatch.setattr(module, "load_agent_config", MagicMock(return_value=("jerry-id", "jerry-key")))
    adapter_instance = object()
    mock_adapter_cls = MagicMock(return_value=adapter_instance)
    runtime_agent = SimpleNamespace(run=AsyncMock())
    mock_create = MagicMock(return_value=runtime_agent)

    monkeypatch.setattr(module, "CrewAIAdapter", mock_adapter_cls)
    monkeypatch.setattr(module.Agent, "create", mock_create)

    await module.main()

    mock_adapter_cls.assert_called_once_with(model="gpt-4o", custom_section="jerry-prompt")
    mock_create.assert_called_once_with(
        adapter=adapter_instance,
        agent_id="jerry-id",
        api_key="jerry-key",
        ws_url="wss://ws.example",
        rest_url="https://rest.example",
    )
    runtime_agent.run.assert_awaited_once()
