"""
Thenvoi SDK - Connect AI agents to the Thenvoi platform.

Core Components:
    ThenvoiAgent: Main coordinator that handles WebSocket, REST, and sessions
    AgentTools: Platform tools bound to a room (send_message, add_participant, etc.)
    PlatformMessage: Message received from the platform
    AgentSession: Per-room session state management

Configuration:
    AgentConfig: Agent-level configuration
    SessionConfig: Per-session configuration

Example:
    from thenvoi import ThenvoiAgent, PlatformMessage, AgentTools

    async def handle_message(msg: PlatformMessage, tools: AgentTools):
        # Your LLM logic here
        await tools.send_message("Hello!", mentions=["User"])

    agent = ThenvoiAgent(agent_id="...", api_key="...")
    await agent.start(on_message=handle_message)
    await agent.run()
"""

from .core import (
    ThenvoiAgent,
    AgentSession,
    AgentTools,
    PlatformMessage,
    AgentConfig,
    SessionConfig,
    ConversationContext,
    render_system_prompt,
    TOOL_MODELS,
)

__all__ = [
    # Core
    "ThenvoiAgent",
    "AgentSession",
    "AgentTools",
    "PlatformMessage",
    # Config
    "AgentConfig",
    "SessionConfig",
    # Context
    "ConversationContext",
    # Prompts
    "render_system_prompt",
    # Tool definitions
    "TOOL_MODELS",
]

__version__ = "0.0.1"
