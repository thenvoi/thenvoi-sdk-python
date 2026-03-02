"""Integration helper public namespace."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ("check_and_format_participants",)


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module("thenvoi.integrations.base")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
