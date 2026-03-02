"""Compatibility tests for canonical runner config loader aliases."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import thenvoi.testing.config_loader as config_loader


def test_load_config_alias_delegates_to_runner_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_loader = MagicMock(return_value={"agent_id": "agent-123"})
    monkeypatch.setattr(config_loader, "load_runner_config", mock_loader)

    result = config_loader.load_config("agent.yml", "agent")

    assert result == {"agent_id": "agent-123"}
    mock_loader.assert_called_once_with("agent.yml", "agent")
