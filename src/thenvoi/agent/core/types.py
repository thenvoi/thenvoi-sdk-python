"""
Core types for Thenvoi agent SDK.

This module defines the data structures and type aliases used across
the SDK for agent communication and platform interaction.

KEY DESIGN PRINCIPLE:
    SDK does NOT send messages directly.
    All communication is via AgentTools used by the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal

if TYPE_CHECKING:
    from .agent import ThenvoiAgent


@dataclass
class AgentConfig:
    """Configuration for ThenvoiAgent coordinator."""

    auto_subscribe_existing_rooms: bool = True


@dataclass
class SessionConfig:
    """Configuration for AgentSession."""

    enable_context_cache: bool = True
    context_cache_ttl_seconds: int = 300
    max_context_messages: int = 100


@dataclass
class PlatformMessage:
    """
    Message from platform (normalized for adapters).

    This is the message format passed to MessageHandlers.
    """

    id: str
    room_id: str
    content: str
    sender_id: str
    sender_type: str  # "User", "Agent", "System"
    sender_name: str | None
    message_type: str
    metadata: dict[str, Any]
    created_at: datetime

    def format_for_llm(self) -> str:
        """
        Format message with sender prefix for LLM consumption.

        Returns string in format: [SENDER_NAME]: message content
        """
        sender = self.sender_name or self.sender_type
        return f"[{sender}]: {self.content}"


@dataclass
class ConversationContext:
    """
    Hydrated context for a room.

    Contains conversation history and participant information
    for context-aware processing.
    """

    room_id: str
    messages: list[dict[str, Any]]
    participants: list[dict[str, Any]]
    hydrated_at: datetime


class AgentTools:
    """
    Tools available to the LLM for platform interaction.

    Bound to a specific room_id. Passed to MessageHandler.
    LLM uses these tools to communicate (SDK never sends directly).

    This class provides:
    - Tool methods (send_message, add_participant, etc.)
    - Schema converters for different LLM frameworks

    Example:
        async def my_handler(msg: PlatformMessage, tools: AgentTools):
            # LLM uses tools to respond
            await tools.send_message("Hello!")

            # Or get tool schemas for your LLM
            anthropic_tools = tools.to_anthropic_tools()
    """

    def __init__(self, room_id: str, coordinator: "ThenvoiAgent"):
        """
        Initialize AgentTools for a specific room.

        Args:
            room_id: The room this tools instance is bound to
            coordinator: The ThenvoiAgent coordinator (for internal API calls)
        """
        self.room_id = room_id
        self._coordinator = coordinator

    # --- Tools for LLM ---

    async def send_message(
        self, content: str, mentions: list[str] | list[dict[str, str]] | None = None
    ) -> dict[str, Any]:
        """
        Send a message to the current room.

        Args:
            content: Message content to send
            mentions: List of participant names (strings). SDK resolves names to IDs.

        Returns:
            Full API response dict with message details (id, content, sender, etc.)

        Raises:
            ValueError: If a mentioned name is not found in participants
        """
        resolved_mentions = self._resolve_mentions(mentions or [])
        return await self._coordinator._send_message_internal(
            self.room_id, content, resolved_mentions
        )

    def _resolve_mentions(
        self, mentions: list[str] | list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Resolve mention names to {id, name} dicts using session participants."""
        session = self._coordinator.active_sessions.get(self.room_id)
        participants = session.participants if session else []

        # Build name -> id lookup
        name_to_id = {p.get("name"): p.get("id") for p in participants}

        resolved = []
        for mention in mentions:
            if isinstance(mention, str):
                name = mention
            else:
                name = mention.get("name", "")
                if mention.get("id"):
                    resolved.append({"id": mention["id"], "name": name})
                    continue

            participant_id = name_to_id.get(name)
            if not participant_id:
                available = list(name_to_id.keys())
                raise ValueError(
                    f"Unknown participant '{name}'. Available: {available}"
                )

            resolved.append({"id": participant_id, "name": name})

        return resolved

    async def send_event(
        self,
        content: str,
        message_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Send an event to the current room.

        Events don't require mentions - use for tool_call, tool_result, error, thought, task.

        Args:
            content: Human-readable event content
            message_type: One of: tool_call, tool_result, thought, error, task
            metadata: Optional structured data for the event

        Returns:
            Full API response dict with event details
        """
        return await self._coordinator._send_event_internal(
            self.room_id, content, message_type, metadata
        )

    async def create_chatroom(self, name: str) -> str:
        """
        Create a new chat room.

        Args:
            name: Name for the new chat room

        Returns:
            Room ID of the created room
        """
        return await self._coordinator._create_chatroom_internal(name)

    async def add_participant(self, name: str, role: str = "member") -> dict[str, Any]:
        """
        Add a participant to the current room by name.

        Args:
            name: Name of the participant (agent or user) to add
            role: Role in room - "owner", "admin", or "member" (default)

        Returns:
            Dict with added participant info (id, name, role, status)
        """
        return await self._coordinator._add_participant_internal(
            self.room_id, name, role
        )

    async def remove_participant(self, name: str) -> dict[str, Any]:
        """
        Remove a participant from the current room by name.

        Args:
            name: Name of the participant to remove

        Returns:
            Dict with removed participant info (id, name, status)
        """
        return await self._coordinator._remove_participant_internal(self.room_id, name)

    async def lookup_peers(self, page: int = 1, page_size: int = 50) -> dict[str, Any]:
        """
        Find available peers (agents and users) on the platform.

        Automatically filters to peers NOT already in the current room.

        Args:
            page: Page number (default 1)
            page_size: Items per page (default 50, max 100)

        Returns:
            Dict with 'peers' list and 'metadata' (page, page_size, total_count, total_pages)
        """
        return await self._coordinator._lookup_peers_internal(
            page=page, page_size=page_size, not_in_chat=self.room_id
        )

    async def get_participants(self) -> list[dict[str, Any]]:
        """
        Get participants in the current room.

        Returns:
            List of participant information dictionaries
        """
        return await self._coordinator._get_participants_internal(self.room_id)

    async def get_context(self) -> ConversationContext:
        """
        Get conversation context for the current room.

        Returns:
            ConversationContext with messages and participants
        """
        return await self._coordinator.get_context(self.room_id)

    # --- Schema converters ---

    def to_anthropic_tools(self) -> list[dict[str, Any]]:
        """
        Convert to Anthropic tool schema format.

        Returns:
            List of tool definitions in Anthropic format
        """
        return [
            {
                "name": "send_message",
                "description": "Send a message to the chat room. Use this to respond to users or other agents.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The message content to send",
                        }
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "add_participant",
                "description": "Add a participant (agent or user) to the current chat room. Use lookup_peers first to find available agents by name.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the participant to add (must match a name from lookup_peers)",
                        },
                        "role": {
                            "type": "string",
                            "enum": ["owner", "admin", "member"],
                            "description": "Role for the participant in this room. Default is 'member'.",
                        },
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "remove_participant",
                "description": "Remove a participant from the current chat room by name.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the participant to remove",
                        }
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "lookup_peers",
                "description": "List available peers (agents and users) that can be added to this room. Automatically excludes peers already in the room. Returns paginated results with metadata.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "page": {
                            "type": "integer",
                            "description": "Page number (default 1)",
                        },
                        "page_size": {
                            "type": "integer",
                            "description": "Items per page (default 50, max 100)",
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": "get_participants",
                "description": "Get a list of all participants (users and agents) in the current chat room.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "create_chatroom",
                "description": "Create a new chat room for a specific task or conversation.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name for the new chat room",
                        }
                    },
                    "required": ["name"],
                },
            },
        ]

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """
        Convert to OpenAI tool schema format.

        Returns:
            List of tool definitions in OpenAI format
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "send_message",
                    "description": "Send a message to the chat room. Provide participant names in mentions array.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The message content to send.",
                            },
                            "mentions": {
                                "type": "array",
                                "description": "List of participant names to mention. At least one required.",
                                "items": {"type": "string"},
                                "minItems": 1,
                            },
                        },
                        "required": ["content", "mentions"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "add_participant",
                    "description": "Add a participant (agent or user) to the current chat room. Use lookup_peers first to find available agents by name.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Name of the participant to add (must match a name from lookup_peers)",
                            },
                            "role": {
                                "type": "string",
                                "enum": ["owner", "admin", "member"],
                                "description": "Role for the participant in this room. Default is 'member'.",
                            },
                        },
                        "required": ["name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "remove_participant",
                    "description": "Remove a participant from the current chat room by name.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Name of the participant to remove",
                            }
                        },
                        "required": ["name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "lookup_peers",
                    "description": "List available peers (agents and users) that can be added to this room. Automatically excludes peers already in the room. Returns paginated results with metadata (page, page_size, total_count, total_pages).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "page": {
                                "type": "integer",
                                "description": "Page number (default 1)",
                            },
                            "page_size": {
                                "type": "integer",
                                "description": "Items per page (default 50, max 100)",
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_participants",
                    "description": "Get a list of all participants in the current chat room.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_chatroom",
                    "description": "Create a new chat room for a specific task or conversation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Name for the new chat room",
                            }
                        },
                        "required": ["name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "send_event",
                    "description": "Send an event to the chat room. Use for thought, error, or task messages. Does NOT require mentions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Human-readable event content",
                            },
                            "message_type": {
                                "type": "string",
                                "enum": ["thought", "error", "task"],
                                "description": "Type of event",
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Optional structured data for the event",
                            },
                        },
                        "required": ["content", "message_type"],
                    },
                },
            },
        ]

    def to_langchain_tools(self) -> list[Any]:
        """
        Convert to LangChain tool format.

        Returns:
            List of LangChain StructuredTool instances

        Note:
            Requires langchain to be installed.
            Import is deferred to avoid hard dependency.
        """
        try:
            from langchain_core.tools import StructuredTool
        except ImportError:
            raise ImportError(
                "langchain_core is required for to_langchain_tools(). "
                "Install with: pip install langchain-core"
            )

        # Create wrapper functions that capture self
        # All wrappers catch exceptions and return error strings so LLM can see failures
        async def send_message_wrapper(
            content: str, mentions: list[str]
        ) -> dict[str, Any] | str:
            """Send a message to the chat room. Provide participant names in mentions."""
            try:
                return await self.send_message(content, mentions)
            except Exception as e:
                return f"Error sending message: {e}"

        async def add_participant_wrapper(
            name: str, role: str = "member"
        ) -> dict[str, Any] | str:
            """Add a participant (agent or user) to the chat room by name. Use lookup_peers first to find available agents."""
            try:
                return await self.add_participant(name, role)
            except Exception as e:
                return f"Error adding participant '{name}': {e}"

        async def remove_participant_wrapper(name: str) -> dict[str, Any] | str:
            """Remove a participant from the chat room by name."""
            try:
                return await self.remove_participant(name)
            except Exception as e:
                return f"Error removing participant '{name}': {e}"

        async def lookup_peers_wrapper(
            page: int = 1, page_size: int = 50
        ) -> dict[str, Any] | str:
            """List available peers (agents and users) on the platform. Returns paginated results with metadata."""
            try:
                return await self.lookup_peers(page, page_size)
            except Exception as e:
                return f"Error looking up peers: {e}"

        async def get_participants_wrapper() -> list[dict[str, Any]] | str:
            """Get participants in the chat room."""
            try:
                return await self.get_participants()
            except Exception as e:
                return f"Error getting participants: {e}"

        async def create_chatroom_wrapper(name: str) -> str:
            """Create a new chat room."""
            try:
                return await self.create_chatroom(name)
            except Exception as e:
                return f"Error creating chatroom '{name}': {e}"

        async def send_event_wrapper(
            content: str,
            message_type: Literal["thought", "error", "task"],
        ) -> dict[str, Any] | str:
            """Send an event to the chat room. No mentions required.

            message_type options:
            - 'thought': Share your reasoning or plan BEFORE taking actions. Explain what you're about to do and why.
            - 'error': Report an error or problem that occurred.
            - 'task': Report task progress or completion status.

            Always send a thought before complex actions to keep users informed.
            """
            try:
                return await self.send_event(content, message_type, None)
            except Exception as e:
                return f"Error sending event: {e}"

        return [
            StructuredTool.from_function(
                coroutine=send_message_wrapper,
                name="send_message",
                description="Send a message to the chat room. Provide participant names in mentions array.",
            ),
            StructuredTool.from_function(
                coroutine=add_participant_wrapper,
                name="add_participant",
                description="Add a participant (agent or user) to the chat room by name. Use lookup_peers first to find available agents. Provide name and optionally role (owner/admin/member, default: member).",
            ),
            StructuredTool.from_function(
                coroutine=remove_participant_wrapper,
                name="remove_participant",
                description="Remove a participant from the chat room by name.",
            ),
            StructuredTool.from_function(
                coroutine=lookup_peers_wrapper,
                name="lookup_peers",
                description="List available peers (agents and users) that can be added to this room. Automatically excludes peers already in the room. Supports pagination with page and page_size parameters. Returns dict with 'peers' list and 'metadata' (page, page_size, total_count, total_pages).",
            ),
            StructuredTool.from_function(
                coroutine=get_participants_wrapper,
                name="get_participants",
                description="Get a list of all participants in the current chat room.",
            ),
            StructuredTool.from_function(
                coroutine=create_chatroom_wrapper,
                name="create_chatroom",
                description="Create a new chat room for a specific task or conversation.",
            ),
            StructuredTool.from_function(
                coroutine=send_event_wrapper,
                name="send_event",
                description="Send an event. Use 'thought' to share reasoning BEFORE actions, 'error' for problems, 'task' for progress updates. Always send a thought before complex actions.",
            ),
        ]

    async def execute_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Execute a tool call by name.

        This is a convenience method for frameworks that need to
        dispatch tool calls programmatically.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool_name is unknown
        """
        tool_map = {
            "send_message": lambda: self.send_message(
                arguments["content"], arguments.get("mentions", [])
            ),
            "send_event": lambda: self.send_event(
                arguments["content"],
                arguments["message_type"],
                arguments.get("metadata"),
            ),
            "add_participant": lambda: self.add_participant(
                arguments["name"], arguments.get("role", "member")
            ),
            "remove_participant": lambda: self.remove_participant(arguments["name"]),
            "lookup_peers": lambda: self.lookup_peers(
                arguments.get("page", 1), arguments.get("page_size", 50)
            ),
            "get_participants": lambda: self.get_participants(),
            "create_chatroom": lambda: self.create_chatroom(arguments["name"]),
        }

        if tool_name not in tool_map:
            raise ValueError(f"Unknown tool: {tool_name}")

        return await tool_map[tool_name]()


# Callback type - receives AgentTools, NOT ThenvoiAgent
MessageHandler = Callable[[PlatformMessage, AgentTools], Awaitable[None]]
