"""Unit tests for orchestrator CLI runtime."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

import thenvoi.integrations.a2a_gateway.orchestrator.cli as cli_module
from thenvoi.integrations.a2a_gateway.orchestrator.remote_agent import (
    GatewayDiscoveryError,
)


def test_parse_available_peers_trims_and_filters_empty() -> None:
    peers = cli_module._parse_available_peers(" weather, data ,,search,  ")

    assert peers == ["weather", "data", "search"]


def test_run_orchestrator_server_returns_error_without_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    mock_logger_error = MagicMock()
    mock_uvicorn_run = MagicMock()
    monkeypatch.setattr(cli_module.logger, "error", mock_logger_error)
    monkeypatch.setattr(cli_module, "load_dotenv", MagicMock())
    monkeypatch.setattr(cli_module.uvicorn, "run", mock_uvicorn_run)

    result = cli_module.run_orchestrator_server(
        host="localhost",
        port=10001,
        gateway_url="http://localhost:10000",
        peers="",
        model="gpt-4o",
    )

    assert result == 1
    mock_logger_error.assert_called_once()
    mock_uvicorn_run.assert_not_called()


def test_run_orchestrator_server_discovers_peers_and_runs_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

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

    mock_discover = AsyncMock(return_value=["weather", "search"])
    mock_executor_cls = MagicMock(return_value=object())
    mock_handler_cls = MagicMock(return_value=object())
    mock_server = MagicMock()
    mock_server.build.return_value = "asgi-app"
    mock_app_cls = MagicMock(return_value=mock_server)
    mock_uvicorn_run = MagicMock()

    monkeypatch.setattr(cli_module, "OrchestratorAgent", _FakeOrchestratorAgent)
    monkeypatch.setattr(cli_module, "OrchestratorAgentExecutor", mock_executor_cls)
    monkeypatch.setattr(cli_module, "DefaultRequestHandler", mock_handler_cls)
    monkeypatch.setattr(cli_module, "A2AStarletteApplication", mock_app_cls)
    monkeypatch.setattr(cli_module, "_discover_peers", mock_discover)
    monkeypatch.setattr(cli_module, "load_dotenv", MagicMock())
    monkeypatch.setattr(cli_module.uvicorn, "run", mock_uvicorn_run)

    result = cli_module.run_orchestrator_server(
        host="127.0.0.1",
        port=7777,
        gateway_url="http://gateway.example",
        peers="",
        model="gpt-test",
    )

    assert result == 0
    mock_discover.assert_called_once_with("http://gateway.example")
    mock_executor_cls.assert_called_once()
    mock_handler_cls.assert_called_once()
    mock_app_cls.assert_called_once()
    mock_uvicorn_run.assert_called_once_with("asgi-app", host="127.0.0.1", port=7777)


def test_run_orchestrator_server_continues_when_discovery_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

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

    mock_discover = AsyncMock(
        side_effect=GatewayDiscoveryError(
            "boom",
            code="list_peers_failed",
            retryable=True,
        )
    )
    mock_warning = MagicMock()
    mock_server = MagicMock()
    mock_server.build.return_value = "asgi-app"

    monkeypatch.setattr(cli_module, "OrchestratorAgent", _FakeOrchestratorAgent)
    monkeypatch.setattr(cli_module, "OrchestratorAgentExecutor", MagicMock(return_value=object()))
    monkeypatch.setattr(cli_module, "DefaultRequestHandler", MagicMock(return_value=object()))
    monkeypatch.setattr(cli_module, "A2AStarletteApplication", MagicMock(return_value=mock_server))
    monkeypatch.setattr(cli_module, "_discover_peers", mock_discover)
    monkeypatch.setattr(cli_module, "load_dotenv", MagicMock())
    monkeypatch.setattr(cli_module.logger, "warning", mock_warning)
    mock_uvicorn_run = MagicMock()
    monkeypatch.setattr(cli_module.uvicorn, "run", mock_uvicorn_run)

    result = cli_module.run_orchestrator_server(
        host="localhost",
        port=10001,
        gateway_url="http://gateway.example",
        peers="",
        model="gpt-test",
    )

    assert result == 0
    mock_discover.assert_called_once_with("http://gateway.example")
    mock_warning.assert_called()
    mock_uvicorn_run.assert_called_once()
