"""
Thenvoi Runtime Layer - Agent execution and room management.

Components:
    RoomPresence: Cross-room lifecycle management
    Execution: Per-room execution protocol
    ExecutionContext: Default execution implementation (context accumulation)
    AgentRuntime: Convenience wrapper combining presence + execution
    AgentTools: Tool interface for LLM platform interaction

Utilities:
    formatters: Pure functions for message formatting
    prompts: System prompt rendering
    ParticipantTracker: Participant tracking with change detection
    MessageRetryTracker: Message retry tracking
"""

# Types
from .types import (
    AgentConfig,
    ConversationContext,
    MessageHandler,
    PlatformMessage,
    SessionConfig,
)

# Core runtime components
from .presence import RoomPresence
from .execution import Execution, ExecutionContext, ExecutionHandler
from .runtime import AgentRuntime

# Tools
from .tools import AgentTools, TOOL_MODELS

# Utilities
from .formatters import (
    format_message_for_llm,
    format_history_for_llm,
    build_participants_message,
)
from .prompts import render_system_prompt, BASE_INSTRUCTIONS, TEMPLATES
from .participant_tracker import ParticipantTracker
from .retry_tracker import MessageRetryTracker

__all__ = [
    # Types
    "AgentConfig",
    "SessionConfig",
    "PlatformMessage",
    "ConversationContext",
    "MessageHandler",
    # Core components
    "RoomPresence",
    "Execution",
    "ExecutionContext",
    "ExecutionHandler",
    "AgentRuntime",
    # Tools
    "AgentTools",
    "TOOL_MODELS",
    # Formatters
    "format_message_for_llm",
    "format_history_for_llm",
    "build_participants_message",
    # Prompts
    "render_system_prompt",
    "BASE_INSTRUCTIONS",
    "TEMPLATES",
    # Trackers
    "ParticipantTracker",
    "MessageRetryTracker",
]
