"""Contact runtime subsystem public namespace."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "ContactEventHandler",
    "ContactTools",
    "ContactService",
    "ContactEventSink",
    "ContactRuntimePort",
    "RuntimeContactEventSink",
    "CallbackContactEventSink",
    "HUB_ROOM_SYSTEM_PROMPT",
    "MAX_DEDUP_CACHE_SIZE",
]

_EXPORT_MODULES: dict[str, str] = {
    "ContactEventHandler": "thenvoi.runtime.contacts.contact_handler",
    "ContactTools": "thenvoi.runtime.contacts.contact_tools",
    "ContactService": "thenvoi.runtime.contacts.service",
    "ContactEventSink": "thenvoi.runtime.contacts.sink",
    "ContactRuntimePort": "thenvoi.runtime.contacts.sink",
    "RuntimeContactEventSink": "thenvoi.runtime.contacts.sink",
    "CallbackContactEventSink": "thenvoi.runtime.contacts.sink",
    "HUB_ROOM_SYSTEM_PROMPT": "thenvoi.runtime.contacts.contact_handler",
    "MAX_DEDUP_CACHE_SIZE": "thenvoi.runtime.contacts.contact_handler",
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
