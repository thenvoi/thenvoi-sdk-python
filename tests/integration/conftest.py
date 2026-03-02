"""Integration test package hooks."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = (
    "tests.support.integration.plugin",
    "tests.support.integration.fixtures",
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark all tests in this package as integration tests."""
    integration_root = Path(__file__).resolve().parent
    for item in items:
        item_path = Path(item.path).resolve()
        if item_path.is_relative_to(integration_root):
            item.add_marker("integration")
