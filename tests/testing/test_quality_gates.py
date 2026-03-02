"""Tests for the canonical quality-gate command wrapper."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from thenvoi.testing import quality_gates


def test_run_quality_gates_invokes_pytest_with_contract_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_mock = MagicMock(return_value=SimpleNamespace(returncode=7))
    monkeypatch.setattr(quality_gates.subprocess, "run", run_mock)

    exit_code = quality_gates.run_quality_gates(["tests/docs/test_quality_gate_contract.py"])

    run_mock.assert_called_once_with(
        [
            quality_gates.sys.executable,
            "-m",
            "pytest",
            "-m",
            "contract_gate",
            "-v",
            "tests/docs/test_quality_gate_contract.py",
        ],
        check=False,
        timeout=quality_gates.QUALITY_GATE_TIMEOUT_S,
    )
    assert exit_code == 7


def test_run_quality_gates_returns_timeout_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_mock = MagicMock(
        side_effect=subprocess.TimeoutExpired(
            cmd=["pytest"],
            timeout=quality_gates.QUALITY_GATE_TIMEOUT_S,
        )
    )
    monkeypatch.setattr(quality_gates.subprocess, "run", run_mock)

    exit_code = quality_gates.run_quality_gates()

    assert exit_code == 124
    run_mock.assert_called_once()
