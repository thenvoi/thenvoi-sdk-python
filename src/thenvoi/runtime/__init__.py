"""Core runtime package surface.

This module intentionally exposes runtime-core execution and lifecycle APIs only.
Domain-specific helpers (contacts, prompts, formatters, tool schemas, trackers)
should be imported from their dedicated submodules.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from thenvoi.runtime.tools import AgentTools

__all__ = (
    "AgentConfig",
    "SessionConfig",
    "PlatformMessage",
    "ConversationContext",
    "MessageHandler",
    "RoomPresence",
    "Execution",
    "ExecutionContext",
    "ExecutionHandler",
    "AgentRuntime",
    "AgentTools",
    "GracefulShutdown",
    "run_with_graceful_shutdown",
)


_EXPORT_MODULES: dict[str, tuple[str, str]] = {
    "AgentConfig": ("thenvoi.runtime.types", "AgentConfig"),
    "SessionConfig": ("thenvoi.runtime.types", "SessionConfig"),
    "PlatformMessage": ("thenvoi.runtime.types", "PlatformMessage"),
    "ConversationContext": ("thenvoi.runtime.types", "ConversationContext"),
    "MessageHandler": ("thenvoi.runtime.types", "MessageHandler"),
    "RoomPresence": ("thenvoi.runtime.presence", "RoomPresence"),
    "Execution": ("thenvoi.runtime.execution", "Execution"),
    "ExecutionContext": ("thenvoi.runtime.execution", "ExecutionContext"),
    "ExecutionHandler": ("thenvoi.runtime.execution", "ExecutionHandler"),
    "AgentRuntime": ("thenvoi.runtime.runtime", "AgentRuntime"),
    "GracefulShutdown": ("thenvoi.runtime.shutdown", "GracefulShutdown"),
    "run_with_graceful_shutdown": (
        "thenvoi.runtime.shutdown",
        "run_with_graceful_shutdown",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _EXPORT_MODULES[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
