"""Thenvoi SDK public API.

User-facing root imports are kept narrow:
- Agent
- ThenvoiLink
- PlatformEvent
- PlatformMessage
- AgentConfig
- SessionConfig
- ConversationContext

Runtime internals remain available from thenvoi.runtime.
"""

from importlib.metadata import version as _get_version, PackageNotFoundError

# Composition layer (new pattern)
from .agent import Agent

# Platform layer
from .platform import ThenvoiLink, PlatformEvent

# Runtime layer
from .runtime import (
    AgentConfig,
    ConversationContext,
    PlatformMessage,
    SessionConfig,
)

__all__ = [
    # Composition
    "Agent",
    # Platform
    "ThenvoiLink",
    "PlatformEvent",
    # Runtime - Types
    "PlatformMessage",
    "AgentConfig",
    "SessionConfig",
    "ConversationContext",
]

try:
    __version__ = _get_version("thenvoi-sdk")
except PackageNotFoundError:
    __version__ = "0.1.0"  # Fallback for editable installs
