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
from .config import AgentConfig

# Platform layer
from .platform import ThenvoiLink, PlatformEvent

# Runtime layer
from .runtime.execution import ExecutionContext
from .runtime.runtime import AgentRuntime
from .runtime.types import ConversationContext, PlatformMessage, SessionConfig

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
    "AgentRuntime",
    "ExecutionContext",
]

try:
    __version__ = _get_version("thenvoi-sdk")
except PackageNotFoundError:
    __version__ = "0.1.0"  # Fallback for editable installs
