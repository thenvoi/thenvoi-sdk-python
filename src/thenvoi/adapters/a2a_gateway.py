"""Deprecated compatibility shim for A2A gateway adapter symbols.

Canonical imports:
- ``from thenvoi.adapters import A2AGatewayAdapter``
- ``from thenvoi.integrations.a2a.gateway import A2AGatewayAdapter``

This shim remains for backwards compatibility and is scheduled for removal in
``v1.0.0``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenvoi.compat.shims import make_registered_module_shim

if TYPE_CHECKING:
    from thenvoi.integrations.a2a.gateway.adapter import (
        A2AGatewayAdapter as A2AGatewayAdapter,
    )
    from thenvoi.integrations.a2a.gateway.server import GatewayServer as GatewayServer
    from thenvoi.integrations.a2a.gateway.types import (
        GatewaySessionState as GatewaySessionState,
    )
    from thenvoi.integrations.a2a.gateway.types import PendingA2ATask as PendingA2ATask

REMOVAL_VERSION, __all__, __getattr__ = make_registered_module_shim(
    __name__,
)
