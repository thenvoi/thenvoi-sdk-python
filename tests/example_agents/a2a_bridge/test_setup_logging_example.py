"""Tests for examples/a2a_bridge/setup_logging.py."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import examples.a2a_bridge.setup_logging as setup_logging_module


def test_setup_logging_delegates_to_shared_profile(monkeypatch) -> None:
    mock_setup_profile = MagicMock()
    monkeypatch.setattr(setup_logging_module, "setup_logging_profile", mock_setup_profile)

    setup_logging_module.setup_logging(level=logging.DEBUG, a2a_debug=True)

    mock_setup_profile.assert_called_once_with(
        "a2a_bridge",
        logging.DEBUG,
        a2a_debug=True,
    )
