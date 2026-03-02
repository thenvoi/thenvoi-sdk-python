"""Global pytest configuration hooks."""

from __future__ import annotations

from thenvoi_testing.markers import pytest_ignore_collect_in_ci as _ignore_collect_in_ci

pytest_plugins = (
    "tests.support.fixtures",
)


def pytest_ignore_collect(collection_path):
    """Skip integration tests in CI environment."""
    return _ignore_collect_in_ci(str(collection_path), "integration")
