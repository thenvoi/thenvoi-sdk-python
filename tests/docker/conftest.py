from __future__ import annotations

import sys
from pathlib import Path

# Make docker/shared/ importable for repo_init tests.
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent / "docker" / "shared")
)
