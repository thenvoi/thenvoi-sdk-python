"""Shared logging configuration for examples."""

from __future__ import annotations

import logging

from thenvoi.testing.example_logging import setup_logging_profile


def setup_logging(level: int = logging.INFO, a2a_debug: bool = False) -> None:
    """Configure example logging for A2A bridge examples."""
    setup_logging_profile("a2a_bridge", level, a2a_debug=a2a_debug)
