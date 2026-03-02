"""Canonical repository quality-gate command."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence

QUALITY_GATE_PYTEST_ARGS: tuple[str, ...] = ("-m", "contract_gate", "-v")
QUALITY_GATE_TIMEOUT_S = 30 * 60


def run_quality_gates(
    extra_args: Sequence[str] = (),
    *,
    timeout_s: int = QUALITY_GATE_TIMEOUT_S,
) -> int:
    """Run contract-gate tests and return pytest's exit code."""
    command = [sys.executable, "-m", "pytest", *QUALITY_GATE_PYTEST_ARGS, *extra_args]
    try:
        completed = subprocess.run(command, check=False, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return 124
    return int(completed.returncode)


def main() -> None:
    """CLI entrypoint for the canonical quality-gate command."""
    raise SystemExit(run_quality_gates(sys.argv[1:]))
