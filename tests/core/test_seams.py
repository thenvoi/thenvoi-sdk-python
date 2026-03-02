"""Tests for typed seam boundary contracts."""

from __future__ import annotations

import pytest

from thenvoi.core.seams import BoundaryResult


def test_boundary_result_success_unwraps_value() -> None:
    result = BoundaryResult.success("ok")
    assert result.is_ok is True
    assert result.unwrap(operation="test op") == "ok"


def test_boundary_result_failure_preserves_error_details() -> None:
    result = BoundaryResult.failure(
        code="bad_input",
        message="input invalid",
        details={"field": "agent_id"},
    )

    assert result.is_ok is False
    assert result.error is not None
    assert result.error.code == "bad_input"
    assert result.error.details == {"field": "agent_id"}


def test_boundary_result_failure_unwrap_raises_runtime_error() -> None:
    result = BoundaryResult.failure(
        code="runner_failed",
        message="boom",
    )

    with pytest.raises(RuntimeError, match="runner_failed"):
        result.unwrap(operation="runner setup")
