"""A2A protocol integration public namespace."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ("A2AAdapter", "A2AAuth", "A2ASessionState")

_EXPORT_MODULES: dict[str, str] = {
    "A2AAdapter": "thenvoi.integrations.a2a.adapter",
    "A2AAuth": "thenvoi.integrations.a2a.types",
    "A2ASessionState": "thenvoi.integrations.a2a.types",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
