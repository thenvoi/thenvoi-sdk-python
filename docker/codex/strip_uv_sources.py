#!/usr/bin/env python3
"""Strip [tool.uv.sources] from pyproject.toml for container builds.

Host-specific source overrides (local wheels, SSH git URLs) don't work
inside Docker.  This script removes the section so ``uv sync`` resolves
dependencies from the published registry instead.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def strip_uv_sources(path: Path) -> None:
    content = path.read_text()
    updated = re.sub(
        r"\n\[tool\.uv\.sources\]\n(?:[^\[\n].*\n?)+",
        "\n",
        content,
        count=1,
    )
    if updated != content:
        path.write_text(updated)


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pyproject.toml")
    strip_uv_sources(target)
