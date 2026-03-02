"""Tests for shared example logging helpers."""

from __future__ import annotations

import logging
import sys
from unittest.mock import MagicMock

import pytest

from thenvoi.testing.example_logging import (
    build_logging_setup,
    setup_example_logging,
    setup_logging_profile,
)


def test_setup_example_logging_configures_basic_and_named_loggers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_basic_config = MagicMock()
    monkeypatch.setattr(logging, "basicConfig", mock_basic_config)

    setup_example_logging(
        level=logging.INFO,
        root_level=logging.ERROR,
        extra_level_loggers=("custom.extra",),
        warning_loggers=("custom.warning",),
        debug_loggers=("custom.debug",),
        stream_stdout=True,
        date_format=None,
    )

    mock_basic_config.assert_called_once_with(
        level=logging.ERROR,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    assert logging.getLogger("thenvoi").level == logging.INFO
    assert logging.getLogger("custom.extra").level == logging.INFO
    assert logging.getLogger("custom.warning").level == logging.WARNING
    assert logging.getLogger("custom.debug").level == logging.DEBUG


def test_setup_logging_profile_unknown_profile_raises() -> None:
    with pytest.raises(ValueError, match="Unknown logging profile"):
        setup_logging_profile("not-a-profile")


def test_setup_logging_profile_applies_dynamic_profile_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _capture(level: int, **kwargs: object) -> None:
        captured["level"] = level
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "thenvoi.testing.example_logging.setup_example_logging",
        _capture,
    )

    setup_logging_profile("codex", logging.DEBUG)

    assert captured["level"] == logging.DEBUG
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["root_level"] == logging.DEBUG
    assert kwargs["stream_stdout"] is True
    assert kwargs["date_format"] is None


def test_build_logging_setup_returns_callable_bound_to_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_setup_profile = MagicMock()
    monkeypatch.setattr(
        "thenvoi.testing.example_logging.setup_logging_profile",
        mock_setup_profile,
    )

    setup_fn = build_logging_setup("parlant")
    setup_fn(logging.WARNING)

    mock_setup_profile.assert_called_once_with("parlant", logging.WARNING)
