"""
Core infrastructure for Thenvoi platform integration.

This module provides the fundamental building blocks for agent frameworks:

Main Classes:
    - ThenvoiAgent: Coordinator for platform integration
    - AgentSession: Per-room message processing
    - AgentTools: Tools available to LLM (send_message, etc.)

Data Types:
    - PlatformMessage: Normalized message format
    - ConversationContext: Hydrated room context
    - AgentConfig, SessionConfig: Configuration dataclasses

Prompt System:
    - render_system_prompt: Render system prompt with agent identity + custom section + base instructions
    - BASE_INSTRUCTIONS: Environment and tools documentation

KEY DESIGN:
    SDK does NOT send messages directly.
    All communication is via AgentTools used by the LLM.

Example:
    from thenvoi.agent.core import ThenvoiAgent, PlatformMessage, AgentTools

    agent = ThenvoiAgent(agent_id="...", api_key="...")

    async def handler(msg: PlatformMessage, tools: AgentTools):
        # LLM uses tools to respond
        await tools.send_message("Hello!")

    await agent.start(on_message=handler)
"""

from .types import (
    AgentConfig,
    AgentTools,
    ConversationContext,
    MessageHandler,
    PlatformMessage,
    SessionConfig,
)
from .agent import ThenvoiAgent
from .session import AgentSession
from .prompts import (
    BASE_INSTRUCTIONS,
    TEMPLATES,
    render_system_prompt,
)
from .tool_definitions import TOOL_MODELS

__all__ = [
    # Core classes
    "ThenvoiAgent",
    "AgentSession",
    "AgentTools",
    # Configuration
    "AgentConfig",
    "SessionConfig",
    # Data types
    "PlatformMessage",
    "ConversationContext",
    "MessageHandler",
    # Prompts
    "BASE_INSTRUCTIONS",
    "TEMPLATES",
    "render_system_prompt",
    # Tool definitions
    "TOOL_MODELS",
]
