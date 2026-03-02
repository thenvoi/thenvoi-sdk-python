"""Thenvoi A2A bridge public namespace."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = (
    "BaseHandler",
    "HandlerResult",
    "BridgeConfig",
    "HealthServer",
    "InMemorySessionStore",
    "MentionRouter",
    "ParticipantRecord",
    "ReconnectConfig",
    "SessionData",
    "SessionStore",
    "ThenvoiBridge",
)

_EXPORT_MODULES: dict[str, str] = {
    "BaseHandler": "thenvoi.integrations.a2a_bridge.handler",
    "HandlerResult": "thenvoi.integrations.a2a_bridge.handler",
    "BridgeConfig": "thenvoi.integrations.a2a_bridge.bridge",
    "HealthServer": "thenvoi.integrations.a2a_bridge.health",
    "InMemorySessionStore": "thenvoi.integrations.a2a_bridge.session",
    "MentionRouter": "thenvoi.integrations.a2a_bridge.router",
    "ParticipantRecord": "thenvoi.integrations.a2a_bridge.bridge",
    "ReconnectConfig": "thenvoi.integrations.a2a_bridge.bridge",
    "SessionData": "thenvoi.integrations.a2a_bridge.session",
    "SessionStore": "thenvoi.integrations.a2a_bridge.session",
    "ThenvoiBridge": "thenvoi.integrations.a2a_bridge.bridge",
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
