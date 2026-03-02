"""Deprecated compatibility shim for bridge handler protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenvoi.compat.shims import make_registered_module_shim

if TYPE_CHECKING:
    from thenvoi.integrations.a2a_bridge.handler import BaseHandler as BaseHandler

REMOVAL_VERSION, __all__, __getattr__ = make_registered_module_shim(
    __name__,
)
