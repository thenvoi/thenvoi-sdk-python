"""Public Thenvoi SDK exports for platform, runtime, and composition APIs."""

from importlib.metadata import version as _get_version, PackageNotFoundError

# Composition layer (new pattern)
from .agent import Agent

# Platform layer
from .platform.event import PlatformEvent
from .platform.link import ThenvoiLink

# Runtime layer
from .runtime.execution import Execution, ExecutionContext, ExecutionHandler
from .runtime.presence import RoomPresence
from .runtime.runtime import AgentRuntime
from .runtime.types import (
    AgentConfig,
    ConversationContext,
    PlatformMessage,
    SessionConfig,
)
from .runtime.formatters import (
    build_participants_message,
    format_history_for_llm,
    format_message_for_llm,
)
from .runtime.participant_tracker import ParticipantTracker
from .runtime.prompts import render_system_prompt
from .runtime.retry_tracker import MessageRetryTracker
from .runtime.shutdown import GracefulShutdown, run_with_graceful_shutdown
from .runtime.tools import (
    ALL_TOOL_NAMES,
    BASE_TOOL_NAMES,
    CHAT_TOOL_NAMES,
    CONTACT_TOOL_NAMES,
    MCP_TOOL_PREFIX,
    MEMORY_TOOL_NAMES,
    TOOL_MODELS,
    AgentTools,
    mcp_tool_names,
)

# Establish stable import-graph edges for optional adapters without forcing
# hard runtime dependency requirements for users who don't install those extras.
try:
    from .adapters.codex.adapter import CodexAdapter as _CodexAdapterImportEdge
except ImportError:
    _CodexAdapterImportEdge = None

from .adapters.crewai import CrewAIAdapter as _CrewAIAdapterImportEdge
from .integrations.claude_sdk.session_manager import (
    ClaudeSessionManager as _ClaudeSessionManagerImportEdge,
)

_ADAPTER_IMPORT_EDGES = (
    _CodexAdapterImportEdge,
    _CrewAIAdapterImportEdge,
    _ClaudeSessionManagerImportEdge,
)

__all__ = [
    # Composition
    "Agent",
    # Platform
    "ThenvoiLink",
    "PlatformEvent",
    # Runtime - Core
    "AgentRuntime",
    "RoomPresence",
    "Execution",
    "ExecutionContext",
    "ExecutionHandler",
    "AgentTools",
    # Runtime - Types
    "PlatformMessage",
    "AgentConfig",
    "SessionConfig",
    "ConversationContext",
    # Runtime - Prompts
    "render_system_prompt",
    # Runtime - Tools
    "TOOL_MODELS",
    "ALL_TOOL_NAMES",
    "BASE_TOOL_NAMES",
    "CHAT_TOOL_NAMES",
    "CONTACT_TOOL_NAMES",
    "MEMORY_TOOL_NAMES",
    "MCP_TOOL_PREFIX",
    "mcp_tool_names",
    # Runtime - Formatters
    "format_message_for_llm",
    "format_history_for_llm",
    "build_participants_message",
    # Runtime - Trackers
    "ParticipantTracker",
    "MessageRetryTracker",
    # Runtime - Shutdown
    "GracefulShutdown",
    "run_with_graceful_shutdown",
]

try:
    __version__ = _get_version("thenvoi-sdk")
except PackageNotFoundError:
    __version__ = "0.1.0"  # Fallback for editable installs
