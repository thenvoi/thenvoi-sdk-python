"""
Thenvoi SDK - Connect AI agents to the Thenvoi platform.

Platform Layer:
    ThenvoiLink: WebSocket + REST transport
    PlatformEvent: Typed events from the platform

Runtime Layer:
    AgentRuntime: Convenience wrapper (RoomPresence + Execution)
    RoomPresence: Cross-room lifecycle management
    ExecutionContext: Per-room context accumulation
    AgentTools: Platform tools bound to a room (send_message, add_participant, etc.)
    PlatformMessage: Message data structure

Configuration:
    AgentConfig: Agent-level configuration
    SessionConfig: Per-session configuration

Example (SDK-heavy pattern):
    from thenvoi import ThenvoiLink, AgentRuntime, ExecutionContext, AgentTools
    from thenvoi.platform import PlatformEvent

    async def handle_event(ctx: ExecutionContext, event: PlatformEvent):
        tools = AgentTools.from_context(ctx)
        # Your LLM logic here
        await tools.send_message("Hello!", mentions=["User"])

    link = ThenvoiLink(agent_id="...", api_key="...", ws_url="...", rest_url="...")
    runtime = AgentRuntime(link, agent_id="...", on_execute=handle_event)
    await runtime.run()

Example (Framework-light pattern):
    from thenvoi import ThenvoiLink, RoomPresence

    link = ThenvoiLink(agent_id="...", api_key="...", ws_url="...", rest_url="...")
    presence = RoomPresence(link)
    presence.on_room_joined = my_join_handler
    presence.on_room_event = my_event_handler
    await presence.start()
    await link.run_forever()
"""

# Composition layer (new pattern)
from .agent import Agent

# Platform layer
from .platform import ThenvoiLink, PlatformEvent

# Runtime layer
from .runtime import (
    AgentRuntime,
    RoomPresence,
    Execution,
    ExecutionContext,
    ExecutionHandler,
    AgentTools,
    PlatformMessage,
    AgentConfig,
    SessionConfig,
    ConversationContext,
    render_system_prompt,
    TOOL_MODELS,
    # Formatters
    format_message_for_llm,
    format_history_for_llm,
    build_participants_message,
    # Trackers
    ParticipantTracker,
    MessageRetryTracker,
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
    # Runtime - Formatters
    "format_message_for_llm",
    "format_history_for_llm",
    "build_participants_message",
    # Runtime - Trackers
    "ParticipantTracker",
    "MessageRetryTracker",
]

__version__ = "0.0.1"
