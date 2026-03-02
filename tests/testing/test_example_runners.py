"""Focused tests for shared example runner executors."""

from __future__ import annotations

import logging
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock

import pytest

from thenvoi.runtime.types import ContactEventConfig, ContactEventStrategy
from thenvoi.testing import example_runners


@pytest.mark.asyncio
async def test_run_pydantic_ai_agent_applies_contact_strategy_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_adapter_kwargs: dict[str, object] = {}

    class _FakeAdapter:
        def __init__(self, **kwargs: object) -> None:
            captured_adapter_kwargs.update(kwargs)

    adapters_stub = ModuleType("thenvoi.adapters")
    adapters_stub.PydanticAIAdapter = _FakeAdapter
    monkeypatch.setitem(sys.modules, "thenvoi.adapters", adapters_stub)

    fake_agent = MagicMock()
    fake_agent.run = AsyncMock()
    create_mock = MagicMock(return_value=fake_agent)
    monkeypatch.setattr(example_runners.Agent, "create", create_mock)

    contact_config = ContactEventConfig(
        strategy=ContactEventStrategy.CALLBACK,
        on_event=AsyncMock(),
        broadcast_changes=True,
    )
    logger = MagicMock()

    await example_runners.run_pydantic_ai_agent(
        agent_id="agent-1",
        api_key="api-key",
        rest_url="https://rest.example",
        ws_url="wss://ws.example",
        model="anthropic:claude-sonnet-4-5",
        custom_section="Base instructions.",
        enable_streaming=True,
        contact_config=contact_config,
        logger=logger,
    )

    section = str(captured_adapter_kwargs["custom_section"])
    assert "Contact requests are handled automatically." in section
    assert "contacts are added or removed" in section
    assert captured_adapter_kwargs["enable_execution_reporting"] is True

    create_kwargs = create_mock.call_args.kwargs
    assert create_kwargs["contact_config"] is contact_config
    fake_agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_a2a_gateway_agent_sets_debug_and_gateway_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_adapter_kwargs: dict[str, object] = {}

    class _FakeGatewayAdapter:
        def __init__(self, **kwargs: object) -> None:
            captured_adapter_kwargs.update(kwargs)

    adapters_stub = ModuleType("thenvoi.adapters")
    adapters_stub.A2AGatewayAdapter = _FakeGatewayAdapter
    monkeypatch.setitem(sys.modules, "thenvoi.adapters", adapters_stub)

    fake_agent = MagicMock()
    fake_agent.run = AsyncMock()
    create_mock = MagicMock(return_value=fake_agent)
    monkeypatch.setattr(example_runners.Agent, "create", create_mock)

    adapter_logger = MagicMock()
    get_logger_mock = MagicMock(return_value=adapter_logger)
    monkeypatch.setattr(example_runners.logging, "getLogger", get_logger_mock)

    logger = MagicMock()
    await example_runners.run_a2a_gateway_agent(
        agent_id="agent-1",
        api_key="api-key",
        rest_url="https://rest.example",
        ws_url="wss://ws.example",
        gateway_port=8765,
        enable_debug=True,
        logger=logger,
    )

    assert captured_adapter_kwargs["gateway_url"] == "http://localhost:8765"
    assert captured_adapter_kwargs["port"] == 8765
    get_logger_mock.assert_called_with("thenvoi.integrations.a2a.gateway")
    adapter_logger.setLevel.assert_called_once_with(logging.DEBUG)
    fake_agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_parlant_agent_uses_server_and_agent_constructor_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_adapter_kwargs: dict[str, object] = {}

    class _FakeParlantAdapter:
        def __init__(self, **kwargs: object) -> None:
            captured_adapter_kwargs.update(kwargs)

    adapters_stub = ModuleType("thenvoi.adapters")
    adapters_stub.ParlantAdapter = _FakeParlantAdapter
    monkeypatch.setitem(sys.modules, "thenvoi.adapters", adapters_stub)

    parlant_agent = object()

    class _FakeServer:
        def __init__(self) -> None:
            self.create_agent = AsyncMock(return_value=parlant_agent)

        async def __aenter__(self) -> "_FakeServer":
            return self

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

    fake_server = _FakeServer()
    parlant_sdk_stub = ModuleType("parlant.sdk")
    parlant_sdk_stub.Server = lambda: fake_server
    parlant_pkg_stub = ModuleType("parlant")
    parlant_pkg_stub.sdk = parlant_sdk_stub
    monkeypatch.setitem(sys.modules, "parlant", parlant_pkg_stub)
    monkeypatch.setitem(sys.modules, "parlant.sdk", parlant_sdk_stub)

    fake_agent = MagicMock()
    fake_agent.run = AsyncMock()
    create_mock = MagicMock(return_value=fake_agent)
    monkeypatch.setattr(example_runners.Agent, "create", create_mock)

    logger = MagicMock()
    await example_runners.run_parlant_agent(
        agent_id="agent-1",
        api_key="api-key",
        rest_url="https://rest.example",
        ws_url="wss://ws.example",
        model="gpt-4o",
        custom_section="Parlant custom section",
        enable_streaming=True,
        logger=logger,
    )

    fake_server.create_agent.assert_awaited_once_with(
        name="Thenvoi Parlant Agent",
        description="Parlant custom section",
    )
    assert captured_adapter_kwargs["server"] is fake_server
    assert captured_adapter_kwargs["parlant_agent"] is parlant_agent
    assert captured_adapter_kwargs["custom_section"] == "Parlant custom section"
    create_mock.assert_called_once()
    fake_agent.run.assert_awaited_once()
