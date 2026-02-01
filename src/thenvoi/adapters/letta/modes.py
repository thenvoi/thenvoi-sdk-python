"""
Letta adapter configuration and mode definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LettaMode(Enum):
    """Operating mode for Letta adapter."""

    PER_ROOM = "per_room"
    """
    One Letta agent per Thenvoi room.

    - Clean conversation isolation
    - Each room has dedicated agent
    - Maps to Thenvoi AgentExecution concept
    - Use case: Business/data agents, sensitive contexts
    """

    SHARED = "shared"
    """
    One Letta agent across all rooms.

    - Single persistent identity
    - Room context via memory blocks
    - Agent sees all rooms (no isolation)
    - Use case: Personal assistants
    """


@dataclass
class LettaConfig:
    """
    Configuration for Letta adapter.

    Example:
        config = LettaConfig(
            api_key="sk-let-...",
            mode=LettaMode.PER_ROOM,
            model="gpt-4o",
        )
    """

    # Authentication (optional - falls back to LETTA_API_KEY env var)
    api_key: str | None = None
    """Letta API key. If None, uses LETTA_API_KEY env var (empty for self-hosted)."""

    # Mode selection
    mode: LettaMode = LettaMode.PER_ROOM
    """Operating mode: PER_ROOM or SHARED."""

    # Server configuration
    base_url: str = "https://api.letta.com"
    """Letta server URL. Change for self-hosted."""

    mcp_server_url: str | None = None
    """
    Thenvoi MCP server URL for tool execution.
    Example: 'http://localhost:8002/sse'
    If set, tools are executed via MCP instead of stubs.
    """

    # Agent configuration
    model: str = "openai/gpt-4o"
    """LLM model for agents. Format: 'provider/model-name'.
    Examples: 'openai/gpt-4o', 'letta/letta-free', 'anthropic/claude-3-5-sonnet'."""

    embedding_model: str = "openai/text-embedding-3-small"
    """Embedding model for memory retrieval. Format: 'provider/model-name'.
    Examples: 'openai/text-embedding-3-small', 'letta/letta-free'."""

    persona: str | None = None
    """
    Agent persona (personality description).
    Used in persona memory block.
    If None, uses agent_description from Thenvoi.
    """

    # Shared mode specific
    shared_agent_id: str | None = None
    """
    Pre-existing Letta agent ID for SHARED mode.
    If None, creates new agent on first use.
    """

    # Custom tools
    custom_tools: list[Any] = field(default_factory=list)
    """
    Custom tools to attach to agents.
    Can be:
    - Letta BaseTool instances
    - Pydantic models with run() method
    - Tool name strings (for built-in tools)
    """

    # Advanced
    api_timeout: int = 30
    """Timeout for Letta API calls in seconds. Default 30s."""

    tool_execution_timeout: int = 30
    """Timeout for tool execution in seconds."""

    max_reasoning_steps: int = 10
    """Maximum reasoning steps before forcing response."""

    # ─── Tool Configuration ───────────────────────────────────────────────────
    mcp_tools: list[str] | None = None
    """MCP tools to attach to agents. If None, uses default set:
    ['add_agent_chat_participant', 'create_agent_chat_event',
     'create_agent_chat_message', 'list_agent_chat_participants',
     'list_agent_peers']"""

    letta_base_tools: list[str] | None = None
    """Letta base tools to attach. If None, uses default set:
    ['memory', 'conversation_search', 'archival_memory_insert',
     'archival_memory_search']"""

    # ─── MCP Server Settings ──────────────────────────────────────────────────
    mcp_server_name: str = "thenvoi"
    """Name to register MCP server under in Letta."""

    mcp_server_type: str = "sse"
    """MCP transport type: 'sse' or 'http'."""

    # ─── Limits ───────────────────────────────────────────────────────────────
    summary_max_length: int = 150
    """Maximum character length for conversation summaries."""

    room_id_suffix_length: int = 8
    """Length of room ID suffix in per-room agent names."""

    # ─── Memory Block Defaults ────────────────────────────────────────────────
    initial_participants_text: str = "No participants yet. Updated when entering rooms."
    """Initial value for participants memory block."""

    initial_room_contexts_text: str = (
        "No room contexts yet. Update this as you interact in different rooms."
    )
    """Initial value for room_contexts memory block (SHARED mode only)."""
