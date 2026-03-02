"""Contract tests for integration marker registry and pytest marker declarations."""

from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from tests.support.integration.marker_registry import (
    INTEGRATION_MARKER_SPECS,
    LIVE_INTEGRATION_FIXTURE_MARKERS,
)
from tests.support.integration.markers import (
    requires_api,
    requires_multi_agent,
    requires_user_api,
)

pytestmark = pytest.mark.contract_gate


def _pyproject_marker_names() -> set[str]:
    pyproject = Path("pyproject.toml")
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    marker_entries = data["tool"]["pytest"]["ini_options"]["markers"]
    return {
        entry.split(":", maxsplit=1)[0].strip()
        for entry in marker_entries
        if ":" in entry
    }


def test_registry_markers_declared_in_pyproject() -> None:
    declared = _pyproject_marker_names()
    expected = set(INTEGRATION_MARKER_SPECS)
    assert expected <= declared


def test_live_fixture_policy_references_known_markers() -> None:
    known_markers = set(INTEGRATION_MARKER_SPECS)
    referenced_markers = {
        marker
        for marker_group in LIVE_INTEGRATION_FIXTURE_MARKERS.values()
        for marker in marker_group
    }
    assert referenced_markers <= known_markers


@pytest.mark.parametrize(
    ("decorator", "expected"),
    [
        (requires_api, {"integration", "requires_api"}),
        (requires_multi_agent, {"integration", "requires_multi_agent"}),
        (requires_user_api, {"integration", "requires_user_api"}),
    ],
)
def test_integration_markers_do_not_embed_env_skip_logic(
    decorator,
    expected: set[str],
) -> None:
    @decorator
    def _probe() -> None:
        return None

    marks = {mark.name for mark in getattr(_probe, "pytestmark", [])}
    assert marks == expected
