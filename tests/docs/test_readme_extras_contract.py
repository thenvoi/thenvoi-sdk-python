"""Validate README extras references against pyproject optional dependencies."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

_README_EXTRAS_RE = re.compile(r"thenvoi-sdk-python\.git\[([^\]]+)\]")


def _readme_referenced_extras() -> set[str]:
    readme = README_PATH.read_text(encoding="utf-8")
    extras: set[str] = set()
    for match in _README_EXTRAS_RE.findall(readme):
        parts = [part.strip() for part in match.split(",")]
        extras.update(part for part in parts if part)
    return extras


def _pyproject_optional_extras() -> set[str]:
    with PYPROJECT_PATH.open("rb") as handle:
        data = tomllib.load(handle)
    project = data.get("project", {})
    optional = project.get("optional-dependencies", {})
    return set(optional.keys())


def test_readme_extras_exist_in_pyproject() -> None:
    readme_extras = _readme_referenced_extras()
    assert readme_extras, "README contains no install extras to validate"

    pyproject_extras = _pyproject_optional_extras()
    unknown = sorted(readme_extras - pyproject_extras)
    assert not unknown, (
        "README references optional extras not present in pyproject.toml:\n"
        + "\n".join(unknown)
    )
