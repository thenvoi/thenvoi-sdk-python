"""Tests for src/thenvoi/config/__init__.py namespace wrappers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import thenvoi.config as config_module


def test_get_config_path_delegates_to_loader(monkeypatch) -> None:
    expected_path = Path("/tmp/agent_config.yaml")
    monkeypatch.setattr(
        config_module, "_get_config_path", MagicMock(return_value=expected_path)
    )

    assert config_module.get_config_path() == expected_path


def test_load_agent_config_delegates_to_loader(monkeypatch) -> None:
    mock_loader = MagicMock(return_value=("agent-id", "api-key"))
    monkeypatch.setattr(config_module, "_load_agent_config", mock_loader)

    result = config_module.load_agent_config("planner", config_path="agent.yaml")

    assert result == ("agent-id", "api-key")
    mock_loader.assert_called_once_with("planner", config_path="agent.yaml")


def test_resolve_agent_credentials_delegates_to_runtime(monkeypatch) -> None:
    mock_resolver = MagicMock(return_value=("runtime-id", "runtime-key"))
    monkeypatch.setattr(config_module, "_resolve_agent_credentials", mock_resolver)

    result = config_module.resolve_agent_credentials(
        "planner", config_path="agent.yaml"
    )

    assert result == ("runtime-id", "runtime-key")
    mock_resolver.assert_called_once_with("planner", config_path="agent.yaml")


def test_resolve_platform_urls_delegates_to_runtime(monkeypatch) -> None:
    mock_resolver = MagicMock(
        return_value=("wss://ws.thenvoi.test", "https://api.thenvoi.test")
    )
    monkeypatch.setattr(config_module, "_resolve_platform_urls", mock_resolver)

    result = config_module.resolve_platform_urls()

    assert result == ("wss://ws.thenvoi.test", "https://api.thenvoi.test")
    mock_resolver.assert_called_once_with(
        ws_env_key="THENVOI_WS_URL",
        rest_env_key="THENVOI_REST_URL",
        ws_default=None,
        rest_default=None,
    )
