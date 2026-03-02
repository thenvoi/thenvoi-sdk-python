"""Canonical marker-policy contract for integration fixtures."""

from __future__ import annotations

from tests.support.integration.policy import (
    enforce_live_fixture_environment,
    enforce_live_fixture_policy,
    is_integration_mode,
    is_truthy_env,
)

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
]
