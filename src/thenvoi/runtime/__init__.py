"""Thenvoi Runtime Layer - Agent execution and room management."""

from .execution import Execution, ExecutionContext, ExecutionHandler
from .formatters import (
    build_participants_message,
    format_history_for_llm,
    format_message_for_llm,
)
from .participant_tracker import ParticipantTracker
from .presence import RoomPresence
from .prompts import BASE_INSTRUCTIONS, TEMPLATES, render_system_prompt
from .retry_tracker import MessageRetryTracker
from .runtime import AgentRuntime
from .shutdown import GracefulShutdown, run_with_graceful_shutdown
from .tools import (
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
from .types import ConversationContext, MessageHandler, PlatformMessage, SessionConfig

__all__ = [
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
    "TOOL_MODELS",
    "ALL_TOOL_NAMES",
    "BASE_TOOL_NAMES",
    "CHAT_TOOL_NAMES",
    "CONTACT_TOOL_NAMES",
    "MEMORY_TOOL_NAMES",
    "MCP_TOOL_PREFIX",
    "mcp_tool_names",
    "format_message_for_llm",
    "format_history_for_llm",
    "build_participants_message",
    "render_system_prompt",
    "BASE_INSTRUCTIONS",
    "TEMPLATES",
    "ParticipantTracker",
    "MessageRetryTracker",
    "GracefulShutdown",
    "run_with_graceful_shutdown",
]
