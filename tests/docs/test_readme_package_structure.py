"""Validate README package structure references against real paths."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"

_TREE_ENTRY_RE = re.compile(
    r"^(?P<prefix>(?:│   |    )*)(?:├──|└──)\s+(?P<name>.+)$"
)


def _extract_package_structure_paths() -> list[Path]:
    readme = README_PATH.read_text(encoding="utf-8")
    marker = "## Package Structure"
    start = readme.index(marker)
    after_marker = readme[start:]

    first_fence = after_marker.index("```")
    second_fence = after_marker.index("```", first_fence + 3)
    block = after_marker[first_fence + 3 : second_fence]

    collected: list[Path] = []
    stack: dict[int, Path] = {}

    for raw_line in block.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped in {"src/thenvoi/", "examples/"}:
            root = Path(stripped.rstrip("/"))
            collected.append(root)
            stack = {0: root}
            continue

        match = _TREE_ENTRY_RE.match(line)
        if not match:
            continue

        if not stack:
            continue

        depth = (len(match.group("prefix")) // 4) + 1
        name = match.group("name").split("#", maxsplit=1)[0].strip()
        if not name:
            continue

        is_dir = name.endswith("/")
        entry_name = name.rstrip("/")
        parent = stack.get(depth - 1)
        if parent is None:
            continue

        path = parent / entry_name
        collected.append(path)

        if is_dir:
            stack[depth] = path
            for key in tuple(stack):
                if key > depth:
                    del stack[key]

    return collected


def test_readme_package_structure_paths_exist() -> None:
    paths = _extract_package_structure_paths()
    assert paths, "README package structure block produced no paths"

    missing = [str(path) for path in paths if not (REPO_ROOT / path).exists()]
    assert not missing, (
        "README package structure references missing paths:\n"
        + "\n".join(missing)
    )
