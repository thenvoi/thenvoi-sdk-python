"""Public policy interface for integration marker and fixture guard behavior."""

from __future__ import annotations

import os
import re
from functools import partial
from itertools import product

import pytest
from _pytest.mark.expression import Expression

from tests.support.integration.marker_registry import (
    INTEGRATION_MARKER_SPECS,
    LIVE_INTEGRATION_FIXTURE_MARKERS,
)

_MARK_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_MARK_LOGICAL_TOKENS = {"and", "or", "not", "true", "false"}
_MAX_MARKER_ASSIGNMENTS = 256


def is_truthy_env(var_name: str) -> bool:
    """Parse common truthy env var values."""
    return os.environ.get(var_name, "").strip().lower() in {"1", "true", "yes", "on"}


def _marker_names(mark_expression: str) -> set[str]:
    return {
        token.lower()
        for token in _MARK_TOKEN_RE.findall(mark_expression)
        if token.lower() not in _MARK_LOGICAL_TOKENS
    }


def _assignment_matcher(
    assignments: dict[str, bool],
    name: str,
    **_kwargs: object,
) -> bool:
    return assignments.get(name.lower(), False)


def _expression_allows_integration(mark_expression: str) -> bool:
    marker_names = _marker_names(mark_expression)
    if "integration" not in marker_names:
        return False

    other_markers = sorted(marker_names - {"integration"})
    if len(other_markers) > 8:
        return "not integration" not in mark_expression.lower()

    try:
        compiled = Expression.compile(mark_expression)
    except Exception:
        return False

    combinations = product((False, True), repeat=len(other_markers))
    for index, values in enumerate(combinations):
        if index >= _MAX_MARKER_ASSIGNMENTS:
            break
        assignments = dict(zip(other_markers, values))
        assignments["integration"] = True
        matcher = partial(_assignment_matcher, assignments)
        if compiled.evaluate(matcher):
            return True
    return False


def is_integration_mode(config: pytest.Config) -> bool:
    """True when integration tests are explicitly enabled for this run."""
    if is_truthy_env("THENVOI_RUN_INTEGRATION"):
        return True

    mark_expression = str(config.getoption("-m") or "").strip()
    if not mark_expression:
        return False

    return _expression_allows_integration(mark_expression)


def _has_any_marker(item: pytest.Item, marker_names: tuple[str, ...]) -> bool:
    return any(item.get_closest_marker(marker) is not None for marker in marker_names)


def enforce_live_fixture_policy(item: pytest.Item) -> None:
    """Reject tests using live fixtures unless marker policy is satisfied."""
    used_live_fixtures = sorted(
        {
            fixture_name
            for fixture_name in item.fixturenames
            if fixture_name in LIVE_INTEGRATION_FIXTURE_MARKERS
        }
    )
    if not used_live_fixtures:
        return

    if item.get_closest_marker("integration") is None:
        pytest.fail(
            f"{item.nodeid} uses live integration fixtures {used_live_fixtures} "
            "but is not marked with @pytest.mark.integration",
            pytrace=False,
        )

    if not is_integration_mode(item.config):
        pytest.fail(
            f"{item.nodeid} uses live integration fixtures {used_live_fixtures} "
            "outside integration mode. Run with `-m integration` or set "
            "`THENVOI_RUN_INTEGRATION=1`.",
            pytrace=False,
        )

    for fixture_name in used_live_fixtures:
        required_markers = LIVE_INTEGRATION_FIXTURE_MARKERS[fixture_name]
        if not _has_any_marker(item, required_markers):
            required = ", ".join(required_markers)
            pytest.fail(
                f"{item.nodeid} uses live fixture `{fixture_name}` but is missing one "
                f"of the required markers: {required}",
                pytrace=False,
            )


def _required_env_vars_for_fixture(
    item: pytest.Item,
    fixture_name: str,
) -> tuple[str, ...]:
    allowed_markers = LIVE_INTEGRATION_FIXTURE_MARKERS.get(fixture_name, ())
    active_markers = [
        marker_name
        for marker_name in allowed_markers
        if item.get_closest_marker(marker_name) is not None
    ]
    marker_names = active_markers or list(allowed_markers)

    required: list[str] = []
    for marker_name in marker_names:
        marker_spec = INTEGRATION_MARKER_SPECS.get(marker_name)
        if marker_spec is None:
            continue
        required.extend(marker_spec.env_vars)
    return tuple(dict.fromkeys(required))


def enforce_live_fixture_environment(
    request: pytest.FixtureRequest,
    fixture_name: str,
) -> None:
    """Apply centralized environment admission for live integration fixtures.

    This must be called inside the fixture body before creating live clients.
    """
    enforce_live_fixture_policy(request.node)

    required_env_vars = _required_env_vars_for_fixture(request.node, fixture_name)
    if not required_env_vars:
        return

    missing = [name for name in required_env_vars if not os.environ.get(name)]
    if not missing:
        return

    if request.node.get_closest_marker("integration") is not None:
        missing_list = ", ".join(missing)
        pytest.skip(
            f"Missing required integration env vars for fixture `{fixture_name}`: "
            f"{missing_list}"
        )

    pytest.fail(
        f"{request.node.nodeid} requested live fixture `{fixture_name}` with missing "
        f"env vars: {', '.join(missing)}",
        pytrace=False,
    )


__all__ = [
    "enforce_live_fixture_environment",
    "enforce_live_fixture_policy",
    "is_integration_mode",
    "is_truthy_env",
]
