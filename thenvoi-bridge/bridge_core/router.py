"""Deprecated compatibility shim for bridge mention routing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenvoi.compat.shims import make_registered_module_shim

if TYPE_CHECKING:
    from thenvoi.integrations.a2a_bridge.router import MentionRouter as MentionRouter

REMOVAL_VERSION, __all__, __getattr__ = make_registered_module_shim(
    __name__,
)
