"""Shared utilities for deprecated symbol-forwarding shims."""

from __future__ import annotations

import importlib
import warnings
from typing import Any, Mapping, NamedTuple


def _load_target(target: str) -> Any:
    """Load a forwarded symbol from ``module_path:attribute``."""
    module_path, _, attribute = target.partition(":")
    if not module_path or not attribute:
        raise ValueError(
            f"Invalid shim target '{target}'. Expected 'module_path:attribute'."
        )
    module = importlib.import_module(module_path)
    return getattr(module, attribute)


def make_deprecated_getattr(
    module_name: str,
    forward_map: Mapping[str, str],
    *,
    removal_version: str,
    guidance: str,
) -> Any:
    """Create a module-level ``__getattr__`` for deprecated forwarding shims."""

    def _getattr(name: str) -> Any:
        target = forward_map.get(name)
        if target is None:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

        warnings.warn(
            (
                f"{module_name}.{name} is deprecated and will be removed in "
                f"v{removal_version}. {guidance}"
            ),
            DeprecationWarning,
            stacklevel=3,
        )
        return _load_target(target)

    return _getattr


class ShimDefinition(NamedTuple):
    forward_map: Mapping[str, str]
    guidance: str
    removal_version: str = "1.0.0"


_REGISTERED_SHIMS: dict[str, ShimDefinition] = {
    "thenvoi.adapters.a2a": ShimDefinition(
        forward_map={
            "A2AAdapter": "thenvoi.integrations.a2a.adapter:A2AAdapter",
            "A2AAuth": "thenvoi.integrations.a2a.types:A2AAuth",
        },
        guidance="Import from thenvoi.adapters or thenvoi.integrations.a2a instead.",
    ),
    "thenvoi.adapters.a2a_gateway": ShimDefinition(
        forward_map={
            "A2AGatewayAdapter": "thenvoi.integrations.a2a.gateway.adapter:A2AGatewayAdapter",
            "GatewayServer": "thenvoi.integrations.a2a.gateway.server:GatewayServer",
            "GatewaySessionState": "thenvoi.integrations.a2a.gateway.types:GatewaySessionState",
            "PendingA2ATask": "thenvoi.integrations.a2a.gateway.types:PendingA2ATask",
        },
        guidance="Import from thenvoi.adapters or thenvoi.integrations.a2a.gateway instead.",
    ),
    "thenvoi.runtime.compat": ShimDefinition(
        forward_map={
            "ContactEventHandler": "thenvoi.runtime.contacts.contact_handler:ContactEventHandler",
            "ContactService": "thenvoi.runtime.contacts.service:ContactService",
            "HUB_ROOM_SYSTEM_PROMPT": "thenvoi.runtime.contacts.contact_handler:HUB_ROOM_SYSTEM_PROMPT",
            "MAX_DEDUP_CACHE_SIZE": "thenvoi.runtime.contacts.contact_handler:MAX_DEDUP_CACHE_SIZE",
        },
        guidance=(
            "Import from thenvoi.runtime.contacts.contact_handler or "
            "thenvoi.runtime.contacts.service instead."
        ),
    ),
    "thenvoi.runtime.compat.contact_handler": ShimDefinition(
        forward_map={
            "ContactEventHandler": "thenvoi.runtime.contacts.contact_handler:ContactEventHandler",
            "HUB_ROOM_SYSTEM_PROMPT": "thenvoi.runtime.contacts.contact_handler:HUB_ROOM_SYSTEM_PROMPT",
            "MAX_DEDUP_CACHE_SIZE": "thenvoi.runtime.contacts.contact_handler:MAX_DEDUP_CACHE_SIZE",
        },
        guidance="Import from thenvoi.runtime.contacts.contact_handler instead.",
    ),
    "thenvoi.runtime.compat.contact_service": ShimDefinition(
        forward_map={
            "ContactService": "thenvoi.runtime.contacts.service:ContactService",
        },
        guidance="Import from thenvoi.runtime.contacts.service instead.",
    ),
    "thenvoi.runtime.contact_tools": ShimDefinition(
        forward_map={
            "ContactTools": "thenvoi.runtime.contacts.contact_tools:ContactTools",
        },
        guidance="Import from thenvoi.runtime.contacts.contact_tools instead.",
    ),
    "thenvoi.runtime.custom_tools": ShimDefinition(
        forward_map={
            "CustomToolDef": "thenvoi.runtime.tooling.custom_tools:CustomToolDef",
            "get_custom_tool_name": "thenvoi.runtime.tooling.custom_tools:get_custom_tool_name",
            "custom_tool_to_openai_schema": "thenvoi.runtime.tooling.custom_tools:custom_tool_to_openai_schema",
            "custom_tool_to_anthropic_schema": "thenvoi.runtime.tooling.custom_tools:custom_tool_to_anthropic_schema",
            "custom_tools_to_schemas": "thenvoi.runtime.tooling.custom_tools:custom_tools_to_schemas",
            "find_custom_tool": "thenvoi.runtime.tooling.custom_tools:find_custom_tool",
            "execute_custom_tool": "thenvoi.runtime.tooling.custom_tools:execute_custom_tool",
        },
        guidance="Import from thenvoi.runtime.tooling.custom_tools instead.",
    ),
    "bridge_core": ShimDefinition(
        forward_map={
            "BaseHandler": "thenvoi.integrations.a2a_bridge.handler:BaseHandler",
            "BridgeConfig": "thenvoi.integrations.a2a_bridge.bridge:BridgeConfig",
            "HealthServer": "thenvoi.integrations.a2a_bridge.health:HealthServer",
            "InMemorySessionStore": "thenvoi.integrations.a2a_bridge.session:InMemorySessionStore",
            "MentionRouter": "thenvoi.integrations.a2a_bridge.router:MentionRouter",
            "ParticipantRecord": "thenvoi.integrations.a2a_bridge.bridge:ParticipantRecord",
            "ReconnectConfig": "thenvoi.integrations.a2a_bridge.bridge:ReconnectConfig",
            "SessionData": "thenvoi.integrations.a2a_bridge.session:SessionData",
            "SessionStore": "thenvoi.integrations.a2a_bridge.session:SessionStore",
            "ThenvoiBridge": "thenvoi.integrations.a2a_bridge.bridge:ThenvoiBridge",
        },
        guidance="Import from thenvoi.integrations.a2a_bridge instead.",
    ),
    "bridge_core.bridge": ShimDefinition(
        forward_map={
            "BridgeConfig": "thenvoi.integrations.a2a_bridge.bridge:BridgeConfig",
            "ParticipantRecord": "thenvoi.integrations.a2a_bridge.bridge:ParticipantRecord",
            "ReconnectConfig": "thenvoi.integrations.a2a_bridge.bridge:ReconnectConfig",
            "ThenvoiBridge": "thenvoi.integrations.a2a_bridge.bridge:ThenvoiBridge",
            "main": "thenvoi.integrations.a2a_bridge.bridge:main",
        },
        guidance="Import from thenvoi.integrations.a2a_bridge.bridge instead.",
    ),
    "bridge_core.handler": ShimDefinition(
        forward_map={
            "BaseHandler": "thenvoi.integrations.a2a_bridge.handler:BaseHandler",
        },
        guidance="Import from thenvoi.integrations.a2a_bridge.handler instead.",
    ),
    "bridge_core.health": ShimDefinition(
        forward_map={
            "HealthServer": "thenvoi.integrations.a2a_bridge.health:HealthServer",
        },
        guidance="Import from thenvoi.integrations.a2a_bridge.health instead.",
    ),
    "bridge_core.router": ShimDefinition(
        forward_map={
            "MentionRouter": "thenvoi.integrations.a2a_bridge.router:MentionRouter",
        },
        guidance="Import from thenvoi.integrations.a2a_bridge.router instead.",
    ),
    "bridge_core.session": ShimDefinition(
        forward_map={
            "InMemorySessionStore": "thenvoi.integrations.a2a_bridge.session:InMemorySessionStore",
            "SessionData": "thenvoi.integrations.a2a_bridge.session:SessionData",
            "SessionStore": "thenvoi.integrations.a2a_bridge.session:SessionStore",
        },
        guidance="Import from thenvoi.integrations.a2a_bridge.session instead.",
    ),
}


def make_registered_module_shim(module_name: str) -> tuple[str, tuple[str, ...], Any]:
    """Build ``REMOVAL_VERSION``, ``__all__``, and ``__getattr__`` from registry."""
    definition = _REGISTERED_SHIMS.get(module_name)
    if definition is None:
        raise KeyError(f"No registered shim definition for module {module_name!r}")
    removal_version = definition.removal_version
    exports = (*definition.forward_map.keys(), "REMOVAL_VERSION")
    module_getattr = make_deprecated_getattr(
        module_name,
        definition.forward_map,
        removal_version=removal_version,
        guidance=definition.guidance,
    )
    return removal_version, exports, module_getattr


__all__ = ("make_deprecated_getattr", "make_registered_module_shim")
