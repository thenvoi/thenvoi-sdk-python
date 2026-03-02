"""Tests for runtime config resolution helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from thenvoi.config import runtime as runtime_config


def test_resolve_platform_urls_reads_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("THENVOI_WS_URL", "wss://ws.thenvoi.test")
    monkeypatch.setenv("THENVOI_REST_URL", "https://api.thenvoi.test")

    ws_url, rest_url = runtime_config.resolve_platform_urls()

    assert ws_url == "wss://ws.thenvoi.test"
    assert rest_url == "https://api.thenvoi.test"


def test_resolve_platform_urls_supports_custom_env_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUSTOM_WS", "wss://custom-ws.example")
    monkeypatch.setenv("CUSTOM_REST", "https://custom-rest.example")

    ws_url, rest_url = runtime_config.resolve_platform_urls(
        ws_env_key="CUSTOM_WS",
        rest_env_key="CUSTOM_REST",
    )

    assert ws_url == "wss://custom-ws.example"
    assert rest_url == "https://custom-rest.example"


def test_resolve_platform_urls_raises_for_missing_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("THENVOI_WS_URL", raising=False)
    monkeypatch.delenv("THENVOI_REST_URL", raising=False)

    with pytest.raises(
        ValueError, match="THENVOI_WS_URL environment variable is required"
    ):
        runtime_config.resolve_platform_urls()

    monkeypatch.setenv("THENVOI_WS_URL", "wss://ok.example")
    with pytest.raises(
        ValueError,
        match="THENVOI_REST_URL environment variable is required",
    ):
        runtime_config.resolve_platform_urls()


def test_resolve_agent_credentials_resolves_path_before_delegating(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    load_agent_config = MagicMock(return_value=("agent-id", "api-key"))
    monkeypatch.setattr(runtime_config, "load_agent_config", load_agent_config)
    config_path = tmp_path / "agent.yml"

    result = runtime_config.resolve_agent_credentials(
        "planner", config_path=config_path
    )

    assert result == ("agent-id", "api-key")
    load_agent_config.assert_called_once_with(
        "planner",
        config_path=config_path.resolve(),
    )
