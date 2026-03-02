"""Core protocols and types for composition-based architecture."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "AgentInput",
    "AgentToolsProtocol",
    "ControlPlaneAdapter",
    "BoundaryError",
    "BoundaryResult",
    "ContactToolsProtocol",
    "FrameworkAdapter",
    "HistoryConverter",
    "HistoryProvider",
    "MemoryToolsProtocol",
    "MessagingDispatchToolsProtocol",
    "MessagingToolsProtocol",
    "PlatformMessage",
    "Preprocessor",
    "SimpleAdapter",
    "AnthropicSchemaToolsProtocol",
    "ToolDispatchProtocol",
    "ToolSchemaProviderProtocol",
    "ParticipantToolsProtocol",
    "create_adapter_from_config",
]

_EXPORT_MODULES: dict[str, str] = {
    "AgentToolsProtocol": "thenvoi.core.protocols",
    "ControlPlaneAdapter": "thenvoi.core.control_plane_adapter",
    "BoundaryError": "thenvoi.core.seams",
    "BoundaryResult": "thenvoi.core.seams",
    "ContactToolsProtocol": "thenvoi.core.protocols",
    "FrameworkAdapter": "thenvoi.core.protocols",
    "HistoryConverter": "thenvoi.core.protocols",
    "MemoryToolsProtocol": "thenvoi.core.protocols",
    "MessagingDispatchToolsProtocol": "thenvoi.core.protocols",
    "MessagingToolsProtocol": "thenvoi.core.protocols",
    "ParticipantToolsProtocol": "thenvoi.core.protocols",
    "AnthropicSchemaToolsProtocol": "thenvoi.core.protocols",
    "Preprocessor": "thenvoi.core.protocols",
    "ToolDispatchProtocol": "thenvoi.core.protocols",
    "ToolSchemaProviderProtocol": "thenvoi.core.protocols",
    "SimpleAdapter": "thenvoi.core.simple_adapter",
    "create_adapter_from_config": "thenvoi.core.adapter_config",
    "AgentInput": "thenvoi.core.types",
    "HistoryProvider": "thenvoi.core.types",
    "PlatformMessage": "thenvoi.core.types",
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
