"""Thenvoi platform namespace with lazy compatibility exports."""

from __future__ import annotations

from typing import Any

__all__ = ["ThenvoiLink", "PlatformEvent"]


def __getattr__(name: str) -> Any:
    if name == "ThenvoiLink":
        from .link import ThenvoiLink

        return ThenvoiLink
    if name == "PlatformEvent":
        from .event import PlatformEvent

        return PlatformEvent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
