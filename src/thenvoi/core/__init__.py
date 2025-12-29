"""Core protocols and types for composition-based architecture."""

from thenvoi.core.protocols import (
    AgentToolsProtocol,
    FrameworkAdapter,
    HistoryConverter,
    Preprocessor,
)
from thenvoi.core.simple_adapter import SimpleAdapter
from thenvoi.core.types import AgentInput, HistoryProvider, PlatformMessage

__all__ = [
    "AgentInput",
    "AgentToolsProtocol",
    "FrameworkAdapter",
    "HistoryConverter",
    "HistoryProvider",
    "PlatformMessage",
    "Preprocessor",
    "SimpleAdapter",
]
