"""Direct module tests for thenvoi.config.loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from thenvoi.config.loader import get_config_path, load_agent_config


def test_get_config_path_uses_current_working_directory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    assert get_config_path() == tmp_path / "agent_config.yaml"


def test_load_agent_config_accepts_flat_file(tmp_path: Path) -> None:
    config_path = tmp_path / "agent_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "agent_id": "agent-123",
                "api_key": "api-key-456",
                "role": "reviewer",
            }
        ),
        encoding="utf-8",
    )

    agent_id, api_key = load_agent_config("ignored-key", config_path=config_path)
    assert agent_id == "agent-123"
    assert api_key == "api-key-456"


def test_load_agent_config_reports_missing_required_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "agent_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "agent_id": "agent-123",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing required fields"):
        load_agent_config("ignored-key", config_path=config_path)
