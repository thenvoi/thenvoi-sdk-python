"""Tests for shared testing config loader helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from thenvoi.testing.config_loader import (
    load_runner_config,
    load_yaml_mapping,
    resolve_agent_credentials,
    resolve_platform_urls,
)


def test_load_yaml_mapping_requires_top_level_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(yaml.safe_dump(["not", "a", "mapping"]), encoding="utf-8")

    with pytest.raises(ValueError, match="top level"):
        load_yaml_mapping(config_path)


def test_load_runner_config_supports_keyed_config(tmp_path: Path) -> None:
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "agent": {
                    "agent_id": "agent-123",
                    "api_key": "api-key-456",
                    "model": "claude-sonnet",
                },
                "workspace": "/tmp/workspace",
            }
        ),
        encoding="utf-8",
    )

    config = load_runner_config(config_path, "agent")
    assert config["agent_id"] == "agent-123"
    assert config["api_key"] == "api-key-456"
    assert config["model"] == "claude-sonnet"
    assert "workspace" not in config


def test_load_runner_config_supports_flat_config(tmp_path: Path) -> None:
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "agent_id": "agent-flat-1",
                "api_key": "flat-key-1",
                "role": "planner",
            }
        ),
        encoding="utf-8",
    )

    config = load_runner_config(config_path, "ignored")
    assert config["agent_id"] == "agent-flat-1"
    assert config["api_key"] == "flat-key-1"
    assert config["role"] == "planner"


def test_resolve_platform_urls_with_env_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("THENVOI_WS_URL", raising=False)
    monkeypatch.delenv("THENVOI_REST_URL", raising=False)

    ws_url, rest_url = resolve_platform_urls(
        ws_default="wss://default/ws",
        rest_default="https://default/rest",
    )
    assert ws_url == "wss://default/ws"
    assert rest_url == "https://default/rest"

    monkeypatch.setenv("THENVOI_WS_URL", "wss://from-env/ws")
    monkeypatch.setenv("THENVOI_REST_URL", "https://from-env/rest")
    ws_url, rest_url = resolve_platform_urls(
        ws_default="wss://default/ws",
        rest_default="https://default/rest",
    )
    assert ws_url == "wss://from-env/ws"
    assert rest_url == "https://from-env/rest"


def test_resolve_agent_credentials_with_explicit_config_path(tmp_path: Path) -> None:
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "runner_agent": {
                    "agent_id": "agent-xyz",
                    "api_key": "api-xyz",
                }
            }
        ),
        encoding="utf-8",
    )

    agent_id, api_key = resolve_agent_credentials(
        "runner_agent",
        config_path=config_path,
    )
    assert agent_id == "agent-xyz"
    assert api_key == "api-xyz"
