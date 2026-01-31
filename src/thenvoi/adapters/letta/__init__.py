"""
Letta adapter for Thenvoi platform.

Connects Letta agents (with persistent memory) to Thenvoi's
multi-room collaboration environment.

Example:
    from thenvoi import Agent
    from thenvoi.adapters.letta import LettaAdapter, LettaConfig, LettaMode

    adapter = LettaAdapter(
        config=LettaConfig(
            api_key="sk-let-...",
            mode=LettaMode.PER_ROOM,
        ),
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id="...",
        api_key="...",
        session_config=SessionConfig(enable_context_hydration=False),
    )

    await agent.run()
"""

from .adapter import LettaAdapter
from .converters import LettaPassthroughConverter
from .exceptions import (
    LettaAdapterError,
    LettaAgentNotFoundError,
    LettaConfigurationError,
    LettaConnectionError,
    LettaMemoryError,
    LettaTimeoutError,
    LettaToolExecutionError,
)
from .memory import MemoryBlocks, MemoryManager
from .modes import LettaConfig, LettaMode
from .prompts import (
    build_consolidation_prompt,
    build_room_entry_context,
    get_system_prompt,
)
from .state import LettaAdapterState, RoomState, StateStore
from .tools import (
    CustomToolBuilder,
    get_letta_tool_ids,
    register_thenvoi_tools,
)

__all__ = [
    # Main adapter
    "LettaAdapter",
    # Configuration
    "LettaMode",
    "LettaConfig",
    # Converter
    "LettaPassthroughConverter",
    # Memory management
    "MemoryManager",
    "MemoryBlocks",
    # Tools
    "CustomToolBuilder",
    "get_letta_tool_ids",
    "register_thenvoi_tools",
    # Prompts
    "get_system_prompt",
    "build_room_entry_context",
    "build_consolidation_prompt",
    # State management
    "RoomState",
    "LettaAdapterState",
    "StateStore",
    # Exceptions
    "LettaAdapterError",
    "LettaAgentNotFoundError",
    "LettaConnectionError",
    "LettaToolExecutionError",
    "LettaMemoryError",
    "LettaTimeoutError",
    "LettaConfigurationError",
]
