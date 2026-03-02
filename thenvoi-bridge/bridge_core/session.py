"""Deprecated compatibility shim for bridge session management."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenvoi.compat.shims import make_registered_module_shim

if TYPE_CHECKING:
    from thenvoi.integrations.a2a_bridge.session import (
        InMemorySessionStore as InMemorySessionStore,
    )
    from thenvoi.integrations.a2a_bridge.session import SessionData as SessionData
    from thenvoi.integrations.a2a_bridge.session import SessionStore as SessionStore

REMOVAL_VERSION, __all__, __getattr__ = make_registered_module_shim(
    __name__,
)
