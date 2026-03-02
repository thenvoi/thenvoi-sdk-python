"""Executable contract tests for documented CI quality-gate guarantees."""

from __future__ import annotations

from pathlib import Path
import tomllib
import pytest

_QUALITY_GATE_COMMAND = "uv run thenvoi-quality-gates"

pytestmark = pytest.mark.contract_gate


def test_ci_contract_checks_use_canonical_command() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert _QUALITY_GATE_COMMAND in workflow


def test_readme_quality_gate_section_is_present_and_specific() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    lowered = readme.lower()

    assert "Quality Gate Matrix (CI)" in readme
    assert "contract checks" in lowered
    assert "integration marker policy" in lowered
    assert _QUALITY_GATE_COMMAND in readme
    assert "uv run pytest tests/integration/ -m integration -v" in readme


def test_pyproject_declares_quality_gate_script_and_marker() -> None:
    pyproject = Path("pyproject.toml")
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    scripts = data["project"]["scripts"]
    assert scripts["thenvoi-quality-gates"] == "thenvoi.testing.quality_gates:main"

    marker_entries = data["tool"]["pytest"]["ini_options"]["markers"]
    marker_names = {
        entry.split(":", maxsplit=1)[0].strip()
        for entry in marker_entries
        if ":" in entry
    }
    assert "contract_gate" in marker_names
