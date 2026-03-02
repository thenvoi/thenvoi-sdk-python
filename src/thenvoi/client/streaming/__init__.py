"""Thenvoi WebSocket streaming SDK public namespace.

Exports are resolved lazily from ``thenvoi.client.streaming.client`` to keep a
stable import surface while avoiding import-time side effects.
"""

from __future__ import annotations

import importlib
from typing import Any

_CLIENT_MODULE = "thenvoi.client.streaming.client"

__all__ = [
    "WebSocketClient",
    "MessageCreatedPayload",
    "RoomAddedPayload",
    "RoomRemovedPayload",
    "RoomOwner",
    "ParticipantAddedPayload",
    "ParticipantRemovedPayload",
    "MessageMetadata",
    "Mention",
    "ContactRequestReceivedPayload",
    "ContactRequestUpdatedPayload",
    "ContactAddedPayload",
    "ContactRemovedPayload",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(_CLIENT_MODULE)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
