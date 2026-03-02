"""Tests for examples/parlant/*.py scripts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _load_module(monkeypatch: pytest.MonkeyPatch, filename: str) -> ModuleType:
    parlant_pkg = ModuleType("parlant")
    parlant_sdk = ModuleType("parlant.sdk")
    parlant_sdk.NLPServices = SimpleNamespace(openai="openai")
    parlant_pkg.sdk = parlant_sdk
    monkeypatch.setitem(sys.modules, "parlant", parlant_pkg)
    monkeypatch.setitem(sys.modules, "parlant.sdk", parlant_sdk)

    module_name = f"examples_parlant_{filename.replace('.', '_')}_test"
    spec = importlib.util.spec_from_file_location(
        module_name,
        Path("examples/parlant") / filename,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load examples/parlant/{filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_basic_setup_agent_with_guidelines_creates_expected_guidelines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch, "01_basic_agent.py")
    fake_agent = SimpleNamespace(create_guideline=AsyncMock())
    fake_server = SimpleNamespace(create_agent=AsyncMock(return_value=fake_agent))
    tools = [object()]

    agent = await module.setup_agent_with_guidelines(fake_server, tools)

    assert agent is fake_agent
    fake_server.create_agent.assert_awaited_once_with(
        name="Parlant",
        description=module.AGENT_DESCRIPTION,
    )
    assert fake_agent.create_guideline.await_count == 5


@pytest.mark.asyncio
async def test_basic_main_builds_adapter_and_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch, "01_basic_agent.py")
    parlant_agent = SimpleNamespace(id="parlant-1")
    server_instance = SimpleNamespace()

    class _Server:
        def __init__(self, nlp_service: str | None = None) -> None:
            self.nlp_service = nlp_service

        async def __aenter__(self) -> SimpleNamespace:
            return server_instance

        async def __aexit__(self, _exc_type, _exc, _tb) -> bool:
            return False

    session = SimpleNamespace(agent=SimpleNamespace(run=AsyncMock()))
    tools = [SimpleNamespace(tool=SimpleNamespace(name="thenvoi_send_message"))]
    monkeypatch.setitem(module.p.__dict__, "Server", _Server)
    monkeypatch.setattr(module, "create_parlant_tools", MagicMock(return_value=tools))
    monkeypatch.setattr(
        module,
        "setup_agent_with_guidelines",
        AsyncMock(return_value=parlant_agent),
    )
    adapter_instance = object()
    monkeypatch.setattr(module, "ParlantAdapter", MagicMock(return_value=adapter_instance))
    monkeypatch.setattr(module, "bootstrap_agent", MagicMock(return_value=session))

    await module.main()

    module.setup_agent_with_guidelines.assert_awaited_once_with(server_instance, tools)
    module.bootstrap_agent.assert_called_once_with(
        agent_key="parlant_agent",
        adapter=adapter_instance,
    )
    session.agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_with_guidelines_main_requires_ws_url(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch, "02_with_guidelines.py")
    monkeypatch.delenv("THENVOI_WS_URL", raising=False)
    monkeypatch.setenv("THENVOI_REST_URL", "https://rest.example")
    monkeypatch.setattr(module, "load_dotenv", MagicMock())

    with pytest.raises(ValueError, match="THENVOI_WS_URL environment variable is required"):
        await module.main()


@pytest.mark.asyncio
async def test_with_guidelines_main_creates_thenvoi_agent_and_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch, "02_with_guidelines.py")
    monkeypatch.setenv("THENVOI_WS_URL", "wss://ws.example")
    monkeypatch.setenv("THENVOI_REST_URL", "https://rest.example")
    parlant_agent = SimpleNamespace(id="parlant-2")
    server_instance = SimpleNamespace()

    class _Server:
        def __init__(self, nlp_service: str | None = None) -> None:
            self.nlp_service = nlp_service

        async def __aenter__(self) -> SimpleNamespace:
            return server_instance

        async def __aexit__(self, _exc_type, _exc, _tb) -> bool:
            return False

    monkeypatch.setitem(module.p.__dict__, "Server", _Server)
    monkeypatch.setattr(module, "load_dotenv", MagicMock())
    monkeypatch.setattr(module, "load_agent_config", MagicMock(return_value=("agent-1", "api-key")))
    tools = [SimpleNamespace(tool=SimpleNamespace(name="thenvoi_send_message"))]
    monkeypatch.setattr(module, "create_parlant_tools", MagicMock(return_value=tools))
    monkeypatch.setattr(
        module,
        "setup_agent_with_guidelines",
        AsyncMock(return_value=parlant_agent),
    )
    adapter_instance = object()
    monkeypatch.setattr(module, "ParlantAdapter", MagicMock(return_value=adapter_instance))
    runtime_agent = SimpleNamespace(run=AsyncMock())
    monkeypatch.setattr(module.Agent, "create", MagicMock(return_value=runtime_agent))

    await module.main()

    module.setup_agent_with_guidelines.assert_awaited_once_with(server_instance, tools)
    module.Agent.create.assert_called_once_with(
        adapter=adapter_instance,
        agent_id="agent-1",
        api_key="api-key",
        ws_url="wss://ws.example",
        rest_url="https://rest.example",
    )
    runtime_agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_support_setup_agent_creates_guidelines(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch, "03_support_agent.py")
    fake_agent = SimpleNamespace(create_guideline=AsyncMock())
    fake_server = SimpleNamespace(create_agent=AsyncMock(return_value=fake_agent))

    agent = await module.setup_support_agent(fake_server)

    assert agent is fake_agent
    fake_server.create_agent.assert_awaited_once_with(
        name="Support",
        description=module.SUPPORT_DESCRIPTION,
    )
    assert fake_agent.create_guideline.await_count == 6


@pytest.mark.asyncio
async def test_tom_main_creates_parlant_agent_and_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch, "04_tom_agent.py")
    monkeypatch.setenv("THENVOI_WS_URL", "wss://ws.example")
    monkeypatch.setenv("THENVOI_REST_URL", "https://rest.example")
    parlant_agent = SimpleNamespace(id="tom-id", create_guideline=AsyncMock())

    class _Server:
        def __init__(self, nlp_service: str | None = None) -> None:
            self.nlp_service = nlp_service

        async def __aenter__(self) -> "_Server":
            return self

        async def __aexit__(self, _exc_type, _exc, _tb) -> bool:
            return False

        async def create_agent(self, *, name: str, description: str):
            assert name == "Tom"
            assert description == "tom-prompt"
            return parlant_agent

    monkeypatch.setitem(module.p.__dict__, "Server", _Server)
    monkeypatch.setattr(module, "load_dotenv", MagicMock())
    monkeypatch.setattr(module, "generate_tom_prompt", MagicMock(return_value="tom-prompt"))
    monkeypatch.setattr(module, "load_agent_config", MagicMock(return_value=("tom-agent", "tom-key")))
    monkeypatch.setattr(module, "create_parlant_tools", MagicMock(return_value=["tool"]))
    adapter_instance = object()
    monkeypatch.setattr(module, "ParlantAdapter", MagicMock(return_value=adapter_instance))
    runtime_agent = SimpleNamespace(run=AsyncMock())
    monkeypatch.setattr(module.Agent, "create", MagicMock(return_value=runtime_agent))

    await module.main()

    parlant_agent.create_guideline.assert_awaited_once()
    runtime_agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_jerry_main_creates_parlant_agent_and_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch, "05_jerry_agent.py")
    monkeypatch.setenv("THENVOI_WS_URL", "wss://ws.example")
    monkeypatch.setenv("THENVOI_REST_URL", "https://rest.example")
    parlant_agent = SimpleNamespace(id="jerry-id", create_guideline=AsyncMock())

    class _Server:
        def __init__(self, nlp_service: str | None = None) -> None:
            self.nlp_service = nlp_service

        async def __aenter__(self) -> "_Server":
            return self

        async def __aexit__(self, _exc_type, _exc, _tb) -> bool:
            return False

        async def create_agent(self, *, name: str, description: str):
            assert name == "Jerry"
            assert description == "jerry-prompt"
            return parlant_agent

    monkeypatch.setitem(module.p.__dict__, "Server", _Server)
    monkeypatch.setattr(module, "load_dotenv", MagicMock())
    monkeypatch.setattr(module, "generate_jerry_prompt", MagicMock(return_value="jerry-prompt"))
    monkeypatch.setattr(module, "load_agent_config", MagicMock(return_value=("jerry-agent", "jerry-key")))
    monkeypatch.setattr(module, "create_parlant_tools", MagicMock(return_value=["tool"]))
    adapter_instance = object()
    monkeypatch.setattr(module, "ParlantAdapter", MagicMock(return_value=adapter_instance))
    runtime_agent = SimpleNamespace(run=AsyncMock())
    monkeypatch.setattr(module.Agent, "create", MagicMock(return_value=runtime_agent))

    await module.main()

    parlant_agent.create_guideline.assert_awaited_once()
    runtime_agent.run.assert_awaited_once()
