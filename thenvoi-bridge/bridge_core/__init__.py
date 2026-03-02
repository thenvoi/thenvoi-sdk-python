"""Deprecated compatibility exports for ``bridge_core`` package.

Canonical implementations live in ``thenvoi.integrations.a2a_bridge``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenvoi.compat.shims import make_registered_module_shim

if TYPE_CHECKING:
    from thenvoi.integrations.a2a_bridge.bridge import BridgeConfig as BridgeConfig
    from thenvoi.integrations.a2a_bridge.bridge import (
        ParticipantRecord as ParticipantRecord,
    )
    from thenvoi.integrations.a2a_bridge.bridge import ReconnectConfig as ReconnectConfig
    from thenvoi.integrations.a2a_bridge.bridge import ThenvoiBridge as ThenvoiBridge
    from thenvoi.integrations.a2a_bridge.handler import BaseHandler as BaseHandler
    from thenvoi.integrations.a2a_bridge.health import HealthServer as HealthServer
    from thenvoi.integrations.a2a_bridge.router import MentionRouter as MentionRouter
    from thenvoi.integrations.a2a_bridge.session import (
        InMemorySessionStore as InMemorySessionStore,
    )
    from thenvoi.integrations.a2a_bridge.session import SessionData as SessionData
    from thenvoi.integrations.a2a_bridge.session import SessionStore as SessionStore

REMOVAL_VERSION, __all__, __getattr__ = make_registered_module_shim(
    __name__,
)
