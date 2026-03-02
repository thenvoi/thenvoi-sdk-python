"""Tests for shared dynamic tool loading helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from thenvoi.testing.tool_loading import load_custom_tools


def test_load_custom_tools_loads_selected_registry_entries(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    tools_dir = config_dir / "tools"
    tools_dir.mkdir()

    (tools_dir / "__init__.py").write_text(
        "TOOL_REGISTRY = {'one': object(), 'two': object(), 'three': object()}",
        encoding="utf-8",
    )
    logger = MagicMock()

    loaded = load_custom_tools(tools_dir, config_dir, ["two", "missing", "one"], logger=logger)

    assert len(loaded) == 2
    logger.warning.assert_not_called()
    logger.exception.assert_not_called()


def test_load_custom_tools_rejects_tools_outside_allowed_path(tmp_path: Path) -> None:
    config_dir = tmp_path / "config" / "runner"
    config_dir.mkdir(parents=True)
    tools_dir = tmp_path / "outside-tools"
    tools_dir.mkdir()
    (tools_dir / "__init__.py").write_text("TOOL_REGISTRY = {}", encoding="utf-8")

    logger = MagicMock()
    loaded = load_custom_tools(tools_dir, config_dir, ["tool"], logger=logger)

    assert loaded == []
    logger.warning.assert_called_once()


def test_load_custom_tools_returns_empty_when_init_missing(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    tools_dir = config_dir / "tools"
    tools_dir.mkdir()

    loaded = load_custom_tools(tools_dir, config_dir, ["tool"], logger=MagicMock())
    assert loaded == []


def test_load_custom_tools_logs_exception_when_import_fails(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    tools_dir = config_dir / "tools"
    tools_dir.mkdir()
    (tools_dir / "__init__.py").write_text("raise RuntimeError('boom')", encoding="utf-8")

    logger = MagicMock()
    loaded = load_custom_tools(tools_dir, config_dir, ["tool"], logger=logger)

    assert loaded == []
    logger.exception.assert_called_once_with("Could not load custom tools")
