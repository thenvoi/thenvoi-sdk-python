"""A2A Gateway integration public namespace."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = (
    "A2AGatewayAdapter",
    "GatewayServer",
    "PeerDirectory",
    "PeerRef",
    "GatewaySessionState",
    "GatewaySessionManager",
    "GatewayTaskCorrelator",
    "PendingA2ATask",
)

_EXPORT_MODULES: dict[str, str] = {
    "A2AGatewayAdapter": "thenvoi.integrations.a2a.gateway.adapter",
    "GatewayServer": "thenvoi.integrations.a2a.gateway.server",
    "PeerDirectory": "thenvoi.integrations.a2a.gateway.peer_directory",
    "PeerRef": "thenvoi.integrations.a2a.gateway.peer_directory",
    "GatewaySessionState": "thenvoi.integrations.a2a.gateway.types",
    "GatewaySessionManager": "thenvoi.integrations.a2a.gateway.session_manager",
    "GatewayTaskCorrelator": "thenvoi.integrations.a2a.gateway.task_correlator",
    "PendingA2ATask": "thenvoi.integrations.a2a.gateway.types",
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
