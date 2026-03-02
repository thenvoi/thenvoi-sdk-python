"""Tests for examples/a2a_gateway/02_with_demo_agent.py."""

from __future__ import annotations

import importlib
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import examples.a2a_gateway.with_demo_agent as with_demo_agent_module


def _load_example_module() -> ModuleType:
    return importlib.reload(with_demo_agent_module)


@pytest.mark.asyncio
async def test_run_gateway_uses_env_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("THENVOI_API_KEY", "api-key")
    monkeypatch.setenv("THENVOI_AGENT_ID", "gateway-agent")
    monkeypatch.setenv("THENVOI_WS_URL", "ws://example")
    monkeypatch.setenv("THENVOI_REST_URL", "https://example")
    monkeypatch.setenv("GATEWAY_PORT", "12000")

    module = _load_example_module()

    agent = SimpleNamespace(run=AsyncMock())
    mock_adapter_cls = MagicMock(return_value=object())
    mock_agent_create = MagicMock(return_value=agent)
    mock_load_config = MagicMock()

    monkeypatch.setattr(module, "A2AGatewayAdapter", mock_adapter_cls)
    monkeypatch.setattr(module.Agent, "create", mock_agent_create)
    monkeypatch.setattr(module, "load_agent_config", mock_load_config)

    await module.run_gateway()

    mock_load_config.assert_not_called()
    mock_adapter_cls.assert_called_once_with(
        rest_url="https://example",
        api_key="api-key",
        gateway_url="http://localhost:12000",
        port=12000,
    )
    mock_agent_create.assert_called_once_with(
        adapter=mock_adapter_cls.return_value,
        agent_id="gateway-agent",
        api_key="api-key",
        ws_url="ws://example",
        rest_url="https://example",
    )
    agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_gateway_logs_and_returns_when_config_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("THENVOI_API_KEY", raising=False)

    module = _load_example_module()
    mock_load_config = MagicMock(side_effect=RuntimeError("missing config"))
    mock_agent_create = MagicMock()
    mock_logger_error = MagicMock()

    monkeypatch.setattr(module, "load_agent_config", mock_load_config)
    monkeypatch.setattr(module.Agent, "create", mock_agent_create)
    monkeypatch.setattr(module.logger, "error", mock_logger_error)

    await module.run_gateway()

    mock_load_config.assert_called_once_with("gateway_agent")
    mock_logger_error.assert_called_once()
    mock_agent_create.assert_not_called()


def test_run_orchestrator_returns_when_openai_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    module = _load_example_module()
    mock_logger_error = MagicMock()
    mock_uvicorn_run = MagicMock()

    monkeypatch.setattr(module.logger, "error", mock_logger_error)
    monkeypatch.setattr(module.uvicorn, "run", mock_uvicorn_run)

    module.run_orchestrator()

    mock_logger_error.assert_called_once()
    mock_uvicorn_run.assert_not_called()


def test_run_orchestrator_builds_server_and_runs_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("AVAILABLE_PEERS", "weather, data , ,search")
    monkeypatch.setenv("GATEWAY_PORT", "13000")
    monkeypatch.setenv("ORCHESTRATOR_PORT", "13001")

    module = _load_example_module()

    class _FakeOrchestratorAgent:
        SUPPORTED_CONTENT_TYPES = ("text/plain",)

        def __init__(
            self,
            gateway_url: str,
            available_peers: list[str],
            model: str,
        ) -> None:
            self.gateway_url = gateway_url
            self.available_peers = available_peers
            self.model = model

    mock_executor_cls = MagicMock(return_value=object())
    mock_handler_cls = MagicMock(return_value=object())
    server = MagicMock()
    server.build.return_value = "asgi-app"
    mock_server_cls = MagicMock(return_value=server)
    mock_uvicorn_run = MagicMock()

    monkeypatch.setattr(module, "OrchestratorAgent", _FakeOrchestratorAgent)
    monkeypatch.setattr(module, "OrchestratorAgentExecutor", mock_executor_cls)
    monkeypatch.setattr(module, "DefaultRequestHandler", mock_handler_cls)
    monkeypatch.setattr(module, "A2AStarletteApplication", mock_server_cls)
    monkeypatch.setattr(module.uvicorn, "run", mock_uvicorn_run)

    module.run_orchestrator()

    mock_executor_cls.assert_called_once()
    mock_handler_cls.assert_called_once()
    mock_server_cls.assert_called_once()
    mock_uvicorn_run.assert_called_once_with(
        "asgi-app",
        host="localhost",
        port=13001,
    )


@pytest.mark.asyncio
async def test_main_starts_gateway_and_orchestrator_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_example_module()
    mock_setup_logging = MagicMock()
    mock_load_dotenv = MagicMock()
    mock_sleep = AsyncMock()
    mock_run_gateway = AsyncMock(return_value=None)
    mock_run_orchestrator = MagicMock()
    thread_instance = MagicMock()
    thread_cls = MagicMock(return_value=thread_instance)

    monkeypatch.setattr(module, "setup_logging_profile", mock_setup_logging)
    monkeypatch.setattr(module, "load_dotenv", mock_load_dotenv)
    monkeypatch.setattr(module, "run_gateway", mock_run_gateway)
    monkeypatch.setattr(module, "run_orchestrator", mock_run_orchestrator)
    monkeypatch.setattr(module.asyncio, "sleep", mock_sleep)
    monkeypatch.setattr(module.threading, "Thread", thread_cls)

    await module.main()

    mock_setup_logging.assert_called_once_with("a2a_gateway")
    mock_load_dotenv.assert_called_once()
    mock_run_gateway.assert_awaited_once()
    mock_sleep.assert_awaited_once_with(2)
    thread_cls.assert_called_once_with(target=mock_run_orchestrator, daemon=True)
    thread_instance.start.assert_called_once()
