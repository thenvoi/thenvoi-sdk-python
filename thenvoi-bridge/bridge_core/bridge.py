"""Deprecated compatibility shim for bridge runtime orchestrator.

Canonical import:
- ``from thenvoi.integrations.a2a_bridge.bridge import ThenvoiBridge``
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
    from thenvoi.integrations.a2a_bridge.bridge import main as main

REMOVAL_VERSION, __all__, __getattr__ = make_registered_module_shim(
    __name__,
)
