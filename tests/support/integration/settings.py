"""Shared settings boundary for integration support fixtures."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from thenvoi_testing.settings import ThenvoiTestSettings


class TestSettings(ThenvoiTestSettings):
    """Settings for integration tests, loaded from .env.test."""

    _env_file_path = Path(__file__).parent.parent / ".env.test"


@lru_cache(maxsize=1)
def get_test_settings() -> TestSettings:
    """Load integration settings lazily without mutable module globals."""
    return TestSettings()


__all__ = ["TestSettings", "get_test_settings"]
