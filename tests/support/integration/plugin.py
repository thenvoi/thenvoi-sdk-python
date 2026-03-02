"""Pytest plugin hooks that bind integration policy into test runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from tests.support.integration.contracts.markers import (
    enforce_live_fixture_environment,
    enforce_live_fixture_policy,
    is_integration_mode,
    is_truthy_env,
)

if TYPE_CHECKING:
    from _pytest.config.argparsing import Parser

def pytest_addoption(parser: Parser) -> None:
    """Add --no-clean option to pytest."""
    parser.addoption(
        "--no-clean",
        action="store_true",
        default=False,
        help="Skip cleanup of test-created agents and chats (useful for limit testing)",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Enforce marker discipline for tests that use live API fixtures."""
    enforce_live_fixture_policy(item)


# Backward-compatible aliases for existing imports.
_enforce_live_fixture_environment = enforce_live_fixture_environment
_enforce_live_fixture_policy = enforce_live_fixture_policy
_is_integration_mode = is_integration_mode
_is_truthy_env = is_truthy_env


__all__ = [
    "enforce_live_fixture_environment",
    "enforce_live_fixture_policy",
    "is_integration_mode",
    "is_truthy_env",
    "_enforce_live_fixture_environment",
    "_enforce_live_fixture_policy",
    "_is_integration_mode",
    "_is_truthy_env",
    "pytest_addoption",
    "pytest_runtest_setup",
]
