"""Guardrails for retiring deprecated runtime contact-tools import paths."""

from __future__ import annotations

from pathlib import Path


def test_repo_has_no_deprecated_contact_tools_imports() -> None:
    root = Path(".")
    deprecated_import = ".".join(("thenvoi", "runtime", "contact_tools"))
    allowed_paths = {
        Path("src/thenvoi/compat/shims.py"),
        Path("src/thenvoi/runtime/contact_tools.py"),
    }

    offenders: list[str] = []
    for path in root.rglob("*.py"):
        rel_path = path.relative_to(root)
        if rel_path in allowed_paths:
            continue
        text = path.read_text(encoding="utf-8")
        if deprecated_import in text:
            offenders.append(str(rel_path))

    assert offenders == [], (
        f"Deprecated import path {deprecated_import!r} is still referenced in: "
        f"{offenders}"
    )
