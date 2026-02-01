"""
Letta adapter for Thenvoi platform.

Connects Letta agents (with persistent memory) to Thenvoi's
multi-room collaboration environment.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.core.simple_adapter import SimpleAdapter
from thenvoi.core.types import PlatformMessage

from .converters import LettaPassthroughConverter
from .exceptions import (
    LettaAdapterError,
    LettaAgentNotFoundError,
    LettaConnectionError,
    LettaTimeoutError,
)
from .memory import MemoryManager
from .modes import LettaConfig, LettaMode
from .prompts import build_consolidation_prompt, get_system_prompt
from .state import LettaAdapterState, RoomState, StateStore
from .tools import get_letta_tool_ids, register_thenvoi_tools

if TYPE_CHECKING:
    from letta_client import Letta

logger = logging.getLogger(__name__)


class LettaAdapter(SimpleAdapter[list[dict[str, Any]]]):
    """
    Connects Letta agents to the Thenvoi platform.

    Supports two operating modes:

    **PER_ROOM Mode** (default):
        - Creates one Letta agent per Thenvoi room
        - Each room has isolated conversation history
        - Maps to Thenvoi's AgentExecution concept
        - Best for: Business agents, data analysis, sensitive contexts

    **SHARED Mode**:
        - Uses one Letta agent across all rooms
        - Room context managed via memory blocks
        - Single persistent identity knows user across all rooms
        - Best for: Personal assistants

    Example (per-room mode):
        ```python
        adapter = LettaAdapter(
            config=LettaConfig(
                api_key="sk-let-...",
                mode=LettaMode.PER_ROOM,
                persona="You are a data analysis assistant.",
            ),
        )

        agent = Agent.create(
            adapter=adapter,
            agent_id="thenvoi-agent-id",
            api_key="thenvoi-api-key",
            session_config=SessionConfig(enable_context_hydration=False),
        )
        ```

    Example (shared mode):
        ```python
        adapter = LettaAdapter(
            config=LettaConfig(
                api_key="sk-let-...",
                mode=LettaMode.SHARED,
                persona="You are Alex's personal assistant.",
            ),
        )
        ```
    """

    def __init__(
        self,
        config: LettaConfig,
        state_storage_path: Path | str = "~/.thenvoi/letta_adapter_state.json",
    ):
        """
        Initialize Letta adapter.

        Args:
            config: Letta configuration (api_key, mode, model, etc.)
            state_storage_path: Path to persist room->agent mappings.
                              Supports ~ expansion. Default: ~/.thenvoi/letta_adapter_state.json
        """
        super().__init__(history_converter=LettaPassthroughConverter())
        self.config = config

        # State persistence
        self._state_store = StateStore(Path(state_storage_path).expanduser())

        # Letta client (initialized in on_started)
        self._client: Letta | None = None

        # Memory manager (initialized with client)
        self._memory_manager: MemoryManager | None = None

        # Agent metadata from Thenvoi (set in on_started)
        self._thenvoi_agent_name: str = ""
        self._thenvoi_agent_description: str = ""

        # Thenvoi tool IDs registered with Letta (set in on_started)
        self._thenvoi_tool_ids: dict[str, str] = {}

        # MCP server ID (set in on_started if MCP mode)
        self._mcp_server_id: str | None = None

    @property
    def state(self) -> LettaAdapterState:
        """Access persistent state."""
        return self._state_store.state

    @property
    def client(self) -> "Letta":
        """Get Letta client (raises if not initialized)."""
        if self._client is None:
            raise LettaAdapterError(
                "Letta client not initialized. Call on_started() first."
            )
        return self._client

    @property
    def memory_manager(self) -> MemoryManager:
        """Get memory manager (raises if not initialized)."""
        if self._memory_manager is None:
            raise LettaAdapterError(
                "Memory manager not initialized. Call on_started() first."
            )
        return self._memory_manager

    # ══════════════════════════════════════════════════════════════════════════
    # Lifecycle Methods
    # ══════════════════════════════════════════════════════════════════════════

    async def on_started(self, agent_name: str, agent_description: str) -> None:
        """
        Initialize Letta client and load/create agents.

        Called by Thenvoi runtime after platform connection is established.

        Args:
            agent_name: Name of this agent on Thenvoi platform
            agent_description: Description of this agent on Thenvoi platform
        """
        await super().on_started(agent_name, agent_description)

        self._thenvoi_agent_name = agent_name
        self._thenvoi_agent_description = agent_description

        # Initialize Letta client
        try:
            from letta_client import Letta

            self._client = Letta(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
            # Initialize memory manager
            self._memory_manager = MemoryManager(self._client)
            logger.info(f"Connected to Letta server at {self.config.base_url}")

            # Register tools with Letta
            if self.config.mcp_server_url:
                # MCP mode: Register MCP server (tools will be synced to agents after creation)
                self._mcp_server_id = self._register_mcp_server()
                self._thenvoi_tool_ids = {}
                if self._mcp_server_id:
                    logger.info(f"Registered MCP server: {self.config.mcp_server_url}")
                else:
                    logger.error("Failed to register MCP server")
            else:
                # Stub mode: Register stub tools (backwards compatibility, limited functionality)
                self._thenvoi_tool_ids = register_thenvoi_tools(self._client)
                self._mcp_server_id = None
                logger.warning(
                    "No MCP server configured. Tools registered as stubs - "
                    "they will execute server-side but with limited functionality. "
                    "Configure mcp_server_url for full tool support."
                )
        except ImportError:
            raise LettaAdapterError(
                "letta-client not installed. Run: uv add letta-client"
            )
        except Exception as e:
            raise LettaConnectionError(f"Failed to connect to Letta: {e}")

        # Load persisted state
        self._state_store.load()
        self.state.thenvoi_agent_id = agent_name
        self.state.mode = self.config.mode.value

        # For SHARED mode: ensure we have an agent
        if self.config.mode == LettaMode.SHARED:
            await self._ensure_shared_agent()

        logger.info(
            f"LettaAdapter started in {self.config.mode.value} mode "
            f"for Thenvoi agent '{agent_name}'"
        )

    async def _ensure_shared_agent(self) -> None:
        """Ensure shared agent exists for SHARED mode."""
        # Check config first (user-provided agent ID)
        if self.config.shared_agent_id:
            # Verify it exists
            try:
                await self._verify_agent_exists(self.config.shared_agent_id)
                self.state.shared_agent_id = self.config.shared_agent_id
                self._state_store.save()
                # Update agent config (system prompt, tools, etc.)
                await self._update_agent_config(
                    self.config.shared_agent_id, LettaMode.SHARED
                )
                logger.info(
                    f"Using configured shared agent: {self.config.shared_agent_id}"
                )
                return
            except LettaAgentNotFoundError:
                logger.warning(
                    f"Configured shared_agent_id '{self.config.shared_agent_id}' not found. "
                    f"Creating new agent."
                )

        # Check persisted state
        if self.state.shared_agent_id:
            try:
                await self._verify_agent_exists(self.state.shared_agent_id)
                # Update agent config (system prompt, tools, etc.)
                await self._update_agent_config(
                    self.state.shared_agent_id, LettaMode.SHARED
                )
                logger.info(
                    f"Using persisted shared agent: {self.state.shared_agent_id}"
                )
                return
            except LettaAgentNotFoundError:
                logger.warning("Persisted shared agent not found, creating new one")

        # Create new shared agent
        agent_id = await self._create_shared_agent()
        self.state.shared_agent_id = agent_id
        self._state_store.save()

    async def _verify_agent_exists(self, agent_id: str) -> None:
        """Verify a Letta agent exists."""
        try:
            self.client.agents.retrieve(agent_id=agent_id)
        except Exception as e:
            raise LettaAgentNotFoundError(agent_id) from e

    async def _update_agent_config(self, agent_id: str, mode: LettaMode) -> None:
        """
        Update an existing agent's configuration.

        Updates system prompt and tools to ensure they're current,
        even when reusing a persisted agent.

        Args:
            agent_id: The Letta agent ID to update
            mode: The operating mode (affects system prompt)
        """
        logger.info(f"Updating agent {agent_id} configuration...")

        # Build updated system prompt
        system_prompt = self._build_system_prompt(mode)

        # Get tool IDs (memory tools + any custom tools)
        tool_ids = get_letta_tool_ids(self.client, self._thenvoi_tool_ids)

        try:
            # Update agent via Letta API
            self.client.agents.update(
                agent_id=agent_id,
                system=system_prompt,
                tool_ids=tool_ids,
            )
            logger.info(f"Updated system prompt and tools for agent {agent_id}")

            # Sync MCP tools (these are attached separately)
            self._sync_mcp_tools_to_agent(agent_id)

        except Exception as e:
            logger.warning(f"Failed to update agent config: {e}")
            # Don't fail - agent still exists, just with old config

    def _register_mcp_server(self) -> str | None:
        """
        Register Thenvoi MCP server with Letta and return its ID.

        The MCP server provides Thenvoi platform tools (send_message, etc.)
        via the MCP protocol. Letta will call these tools directly through
        the MCP server, which executes them against the Thenvoi API.

        Returns:
            MCP server ID to use with refresh(), or None if registration failed.
        """
        if not self.config.mcp_server_url:
            return None

        mcp_server = None
        try:
            mcp_server = self.client.mcp_servers.create(
                server_name="thenvoi",
                config={
                    "mcp_server_type": "sse",
                    "server_url": self.config.mcp_server_url,
                },
            )
            logger.info(f"Registered MCP server: {mcp_server.id}")
        except Exception as e:
            # MCP server may already be registered, try to find it
            logger.warning(f"Failed to register MCP server (may already exist): {e}")
            try:
                servers = self.client.mcp_servers.list()
                for server in servers:
                    if server.server_name == "thenvoi":
                        mcp_server = server
                        logger.info(f"Found existing MCP server: {mcp_server.id}")
                        break
            except Exception as list_err:
                logger.error(f"Failed to list MCP servers: {list_err}")
                return None

        if not mcp_server:
            logger.error("Could not register or find MCP server")
            return None

        return mcp_server.id

    def _get_mcp_tool_ids(self) -> list[str]:
        """
        Get agent tool IDs from the registered MCP server.

        Only returns agent tools (prefixed with 'agent_' or containing '_agent_'),
        filtering out user tools.

        Returns:
            List of agent tool IDs from the MCP server.
        """
        if not self._mcp_server_id:
            return []

        # Agent tools that should be attached to Letta agents
        AGENT_TOOL_NAMES = {
            "create_agent_chat_message",
            "create_agent_chat_event",
            "add_agent_chat_participant",
            "remove_agent_chat_participant",
            "list_agent_chat_participants",
            "list_agent_chats",
            "list_agent_peers",
            "get_agent_chat",
            "get_agent_chat_context",
            "get_agent_me",
            "mark_agent_message_processed",
            "mark_agent_message_processing",
            "mark_agent_message_failed",
            "create_agent_chat",
        }

        try:
            # Get tools from MCP server via client.mcp_servers.tools.list()
            tools = self.client.mcp_servers.tools.list(
                mcp_server_id=self._mcp_server_id
            )

            # Filter to only include agent tools
            agent_tools = [t for t in tools if t.name in AGENT_TOOL_NAMES]
            tool_ids = [tool.id for tool in agent_tools]

            logger.info(
                f"Retrieved {len(tool_ids)} agent tools from MCP server "
                f"(filtered from {len(list(tools))} total)"
            )

            if not tool_ids:
                logger.warning(
                    "No agent tools found in MCP server. "
                    "Make sure the MCP server is using an agent API key (thnv_a_...) "
                    "or has agent tools registered."
                )

            return tool_ids
        except Exception as e:
            logger.error(f"Failed to get MCP tools: {e}")
            return []

    def _sync_mcp_tools_to_agent(self, agent_id: str) -> None:
        """
        Sync MCP server tools to an agent by updating its tool_ids.

        This must be called after agent creation to attach MCP tools.
        """
        if not self._mcp_server_id:
            return

        tool_ids = self._get_mcp_tool_ids()
        if not tool_ids:
            logger.warning("No MCP tools to sync to agent")
            return

        try:
            self.client.agents.update(
                agent_id=agent_id,
                tool_ids=tool_ids,
            )
            logger.info(f"Synced {len(tool_ids)} MCP tools to agent {agent_id}")
        except Exception as e:
            logger.error(f"Failed to sync MCP tools to agent: {e}")

    async def _create_shared_agent(self) -> str:
        """Create the shared agent for SHARED mode."""
        logger.info("Creating shared Letta agent...")

        persona = self.config.persona or self._thenvoi_agent_description

        # Get tool IDs (Thenvoi stub tools + memory tools)
        tool_ids = get_letta_tool_ids(self.client, self._thenvoi_tool_ids)

        # Build system prompt with agent identity
        system_prompt = self._build_system_prompt(LettaMode.SHARED)

        agent = self.client.agents.create(
            name=self._thenvoi_agent_name,
            description=self._thenvoi_agent_description
            or f"Letta agent for {self._thenvoi_agent_name}",
            model=self.config.model,
            embedding=self.config.embedding_model,
            memory_blocks=[
                {
                    "label": "persona",
                    "value": persona,
                },
                {
                    "label": "participants",
                    "value": "No participants yet. Updated when entering rooms.",
                },
                {
                    "label": "room_contexts",
                    "value": "No room contexts yet. Update this as you interact in different rooms.",
                },
            ],
            system=system_prompt,
            tool_ids=tool_ids,
        )

        # Sync MCP tools to agent after creation
        self._sync_mcp_tools_to_agent(agent.id)

        logger.info(f"Created shared Letta agent: {agent.id}")
        return agent.id

    # ══════════════════════════════════════════════════════════════════════════
    # Message Handling
    # ══════════════════════════════════════════════════════════════════════════

    async def on_message(
        self,
        msg: PlatformMessage,
        tools: AgentToolsProtocol,
        history: list[dict[str, Any]],  # Ignored - Letta manages its own
        participants_msg: str | None,
        *,
        is_session_bootstrap: bool,
        room_id: str,
    ) -> None:
        """
        Handle incoming message from Thenvoi platform.

        Routes to appropriate handler based on mode.
        """
        logger.debug(
            f"Handling message {msg.id} in room {room_id} (mode={self.config.mode.value})"
        )

        if self.config.mode == LettaMode.PER_ROOM:
            await self._handle_per_room(
                msg, tools, participants_msg, is_session_bootstrap, room_id
            )
        else:
            await self._handle_shared(
                msg, tools, participants_msg, is_session_bootstrap, room_id
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Per-Room Mode Handler
    # ──────────────────────────────────────────────────────────────────────────

    async def _handle_per_room(
        self,
        msg: PlatformMessage,
        tools: AgentToolsProtocol,
        participants_msg: str | None,
        is_session_bootstrap: bool,
        room_id: str,
    ) -> None:
        """Handle message with dedicated agent per room."""
        participants = await tools.get_participants()
        # Filter out agent itself - cannot mention yourself
        participant_names = [
            p.get("name", "Unknown")
            for p in participants
            if p.get("name") != self._thenvoi_agent_name
        ]

        # Get or create Letta agent for this room
        agent_id = await self._get_or_create_room_agent(room_id, participants)
        room_state = self.state.get_or_create_room_state(room_id)

        # Check if this is a rejoin (agent was inactive)
        is_rejoin = not room_state.is_active and room_state.last_interaction is not None
        if is_rejoin:
            room_state.is_active = True
            logger.info(f"Agent rejoined room {room_id}")

        # Update participants in agent memory if changed
        if participants_msg:
            await self._update_agent_participants(agent_id, participants)

        # Build message
        message_parts = []

        # Always include chat_id so agent can use it in tool calls
        message_parts.append(f"[Chat ID: {room_id}]")

        # Rejoin context
        if is_rejoin and room_state.last_interaction:
            time_ago = self._format_time_ago(room_state.last_interaction)
            message_parts.append(
                f"[System: You have rejoined this room after {time_ago}. "
                f"Previous context: {room_state.summary or 'Check your memory'}]"
            )

        # Participant changes
        if participants_msg:
            message_parts.append(f"[Update: {participants_msg}]")

        # The actual message
        message_parts.append(f"[{msg.sender_name}]: {msg.content}")

        formatted_msg = "\n".join(message_parts)

        # Send to Letta and process response
        await self._send_and_process(
            agent_id,
            formatted_msg,
            tools,
            room_state,
            participant_names,
            reply_to=msg.sender_name,
        )

    async def _get_or_create_room_agent(
        self,
        room_id: str,
        participants: list[dict[str, Any]],
    ) -> str:
        """Get existing Letta agent for room or create new one."""
        # Check persisted state
        existing_agent_id = self.state.get_room_agent(room_id)
        if existing_agent_id:
            try:
                await self._verify_agent_exists(existing_agent_id)
                # Update agent config (system prompt, tools, etc.)
                await self._update_agent_config(existing_agent_id, LettaMode.PER_ROOM)
                logger.debug(
                    f"Using existing agent {existing_agent_id} for room {room_id}"
                )
                return existing_agent_id
            except LettaAgentNotFoundError:
                logger.warning(
                    f"Persisted agent {existing_agent_id} not found, creating new"
                )

        # Create new agent for this room
        persona = self.config.persona or self._thenvoi_agent_description

        # Get tool IDs (Thenvoi stub tools + memory tools)
        tool_ids = get_letta_tool_ids(self.client, self._thenvoi_tool_ids)

        # Build system prompt with agent identity
        system_prompt = self._build_system_prompt(LettaMode.PER_ROOM)

        # Filter out agent itself from participants for mention list
        filtered_participants = self._filter_participants_for_mentions(participants)

        agent = self.client.agents.create(
            name=f"{self._thenvoi_agent_name}-{room_id[:8]}",
            description=self._thenvoi_agent_description
            or f"Letta agent for {self._thenvoi_agent_name}",
            model=self.config.model,
            embedding=self.config.embedding_model,
            memory_blocks=[
                {
                    "label": "persona",
                    "value": persona,
                },
                {
                    "label": "participants",
                    "value": self._format_participants(filtered_participants),
                },
            ],
            system=system_prompt,
            tool_ids=tool_ids,
        )

        # Sync MCP tools to agent after creation
        self._sync_mcp_tools_to_agent(agent.id)

        # Persist mapping
        self.state.set_room_agent(room_id, agent.id)
        self._state_store.save()

        logger.info(f"Created Letta agent {agent.id} for room {room_id}")
        return agent.id

    # ──────────────────────────────────────────────────────────────────────────
    # Shared Mode Handler
    # ──────────────────────────────────────────────────────────────────────────

    async def _handle_shared(
        self,
        msg: PlatformMessage,
        tools: AgentToolsProtocol,
        participants_msg: str | None,
        is_session_bootstrap: bool,
        room_id: str,
    ) -> None:
        """
        Handle message with shared agent across rooms using Conversations API.

        Each room maps to one Letta conversation, enabling thread-safe parallel
        room handling without race conditions. The shared agent's memory blocks
        are accessible from all conversations.
        """
        participants = await tools.get_participants()
        # Filter out agent itself - cannot mention yourself
        participant_names = [
            p.get("name", "Unknown")
            for p in participants
            if p.get("name") != self._thenvoi_agent_name
        ]
        room_state = self.state.get_or_create_room_state(room_id)
        agent_id = self.state.shared_agent_id

        if not agent_id:
            raise LettaAdapterError("Shared agent not initialized")

        # Get or create conversation for this room (thread-safe)
        conversation_id = await self._get_or_create_room_conversation(room_id, agent_id)

        # Update participants memory block if changed
        # (This updates the shared memory, visible from all conversations)
        # Filter out agent itself - it cannot mention itself
        if participants_msg or is_session_bootstrap:
            current_snapshot = room_state.participants_snapshot
            if set(participant_names) != set(current_snapshot):
                await self._update_agent_participants(agent_id, participants)
                room_state.participants_snapshot = participant_names

        # Build message (simpler than before - no room header needed,
        # each conversation is already isolated to one room)
        message_parts = []

        # Always include chat_id so agent can use it in tool calls
        message_parts.append(f"[Chat ID: {room_id}]")

        # Room re-entry context (only on bootstrap with prior history)
        if is_session_bootstrap and room_state.last_interaction:
            time_ago = self._format_time_ago(room_state.last_interaction)
            message_parts.append(
                f"[Context: You're back in this room after {time_ago}. "
                f"Previous topic: {room_state.summary or 'Check your memory'}]"
            )

        # Participant changes
        if participants_msg:
            message_parts.append(f"[Update: {participants_msg}]")

        # The actual message
        message_parts.append(f"[{msg.sender_name}]: {msg.content}")

        formatted_msg = "\n".join(message_parts)

        # Send to Letta via Conversations API and process response
        await self._send_and_process(
            agent_id,
            formatted_msg,
            tools,
            room_state,
            participant_names,
            conversation_id=conversation_id,
            reply_to=msg.sender_name,
        )

    async def _get_or_create_room_conversation(
        self,
        room_id: str,
        agent_id: str,
    ) -> str:
        """
        Get existing or create new Letta conversation for a room.

        Each Thenvoi room maps to exactly one Letta conversation.
        Conversations provide thread-safe parallel message handling.
        """
        # Check persisted state
        existing_conv_id = self.state.get_room_conversation(room_id)
        if existing_conv_id:
            # Verify conversation still exists
            try:
                await asyncio.to_thread(
                    self.client.conversations.retrieve,
                    conversation_id=existing_conv_id,
                )
                logger.debug(
                    f"Using existing conversation {existing_conv_id} for room {room_id}"
                )
                return existing_conv_id
            except Exception:
                logger.warning(
                    f"Persisted conversation {existing_conv_id} not found, creating new"
                )

        # Create new conversation for this room
        conversation = await asyncio.to_thread(
            self.client.conversations.create,
            agent_id=agent_id,
        )

        # Persist mapping
        self.state.set_room_conversation(room_id, conversation.id)
        self._state_store.save()

        logger.info(f"Created Letta conversation {conversation.id} for room {room_id}")
        return conversation.id

    # ══════════════════════════════════════════════════════════════════════════
    # Common Processing
    # ══════════════════════════════════════════════════════════════════════════

    async def _send_and_process(
        self,
        agent_id: str,
        message: str,
        tools: AgentToolsProtocol,
        room_state: RoomState,
        participant_names: list[str],
        conversation_id: str | None = None,
        reply_to: str | None = None,
    ) -> None:
        """
        Send message to Letta and process response.

        Args:
            agent_id: Letta agent ID
            message: Formatted message to send
            tools: Thenvoi tools protocol
            room_state: Room state for tracking
            participant_names: List of participant names
            conversation_id: If provided, uses Conversations API (SHARED mode).
                           If None, uses agents.messages.create (PER_ROOM mode).
            reply_to: Name of participant to mention in replies (usually the sender)
        """
        if conversation_id:
            logger.debug(
                f"Sending to Letta conversation {conversation_id}: {message[:100]}..."
            )
        else:
            logger.debug(f"Sending to Letta agent {agent_id}: {message[:100]}...")

        assistant_content: list[str] = []

        try:
            # Wrap Letta API call with timeout
            if conversation_id:
                # SHARED mode: use Conversations API (thread-safe)
                # Note: letta-client always returns a Stream for conversations.messages.create
                # so we need to consume it by iterating
                stream = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.client.conversations.messages.create,
                        conversation_id=conversation_id,
                        messages=[{"role": "user", "content": message}],
                    ),
                    timeout=self.config.api_timeout,
                )
                # Consume the stream to collect all response items
                response_items = list(stream)
            else:
                # PER_ROOM mode: use agents.messages.create directly
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.client.agents.messages.create,
                        agent_id=agent_id,
                        messages=[{"role": "user", "content": message}],
                        streaming=False,
                    ),
                    timeout=self.config.api_timeout,
                )
                response_items = response.messages

            # Process response messages
            for item in response_items:
                msg_type = getattr(item, "message_type", None) or getattr(
                    item, "type", "unknown"
                )

                if msg_type in ("assistant_message", "assistant"):
                    # Assistant text is internal thinking, NOT sent to participants.
                    # The agent uses create_agent_chat_message tool to send messages.
                    content = getattr(item, "content", "") or getattr(item, "text", "")
                    if content:
                        # Log as thought event for visibility
                        await tools.send_event(
                            content=content,
                            message_type="thought",
                        )
                        assistant_content.append(content)
                        logger.debug(f"Agent internal thought: {content[:100]}...")

                elif msg_type == "tool_call":
                    # Tool calls are executed server-side by Letta (via MCP or stubs)
                    # We just report them for visibility
                    tool_call = getattr(item, "tool_call", item)
                    tool_name = getattr(tool_call, "name", "unknown")
                    tool_args = getattr(tool_call, "arguments", {})

                    await tools.send_event(
                        content=f"Calling {tool_name}",
                        message_type="tool_call",
                        metadata={"tool": tool_name, "args": tool_args},
                    )

                elif msg_type == "tool_call_result":
                    # Tool results from server-side execution
                    tool_result = getattr(item, "tool_call_result", item)
                    tool_name = getattr(tool_result, "tool_name", "unknown")
                    result = getattr(tool_result, "result", "")

                    await tools.send_event(
                        content=str(result),
                        message_type="tool_result",
                        metadata={"tool": tool_name},
                    )

                elif msg_type == "reasoning_message":
                    content = getattr(item, "content", "")
                    if content:
                        await tools.send_event(
                            content=content,
                            message_type="thought",
                        )

        except asyncio.TimeoutError:
            logger.error(f"Letta API call timed out after {self.config.api_timeout}s")
            # Mentions are required - mention the sender or first participant
            if reply_to and reply_to != self._thenvoi_agent_name:
                mentions = [reply_to]
            elif participant_names:
                mentions = [participant_names[0]]
            else:
                logger.warning("No valid participants to mention for timeout message")
                raise LettaTimeoutError("send_message", self.config.api_timeout)

            await tools.send_message(
                "I'm sorry, but my response timed out. Please try again.",
                mentions=mentions,
            )
            raise LettaTimeoutError("send_message", self.config.api_timeout)

        finally:
            # Always update room state, even on timeout/error
            room_state.mark_interaction(
                message_id=str(hash(message)),
                participants=participant_names,
            )
            if assistant_content:
                room_state.summary = self._extract_summary(assistant_content)
            self._state_store.save()

    # ══════════════════════════════════════════════════════════════════════════
    # Cleanup
    # ══════════════════════════════════════════════════════════════════════════

    async def on_cleanup(self, room_id: str) -> None:
        """
        Handle agent leaving a room.

        Triggers memory consolidation and marks room as inactive.
        The Letta agent/conversation is kept (not deleted) for potential future rejoins.
        """
        logger.info(f"Agent leaving room {room_id}, triggering cleanup...")

        room_state = self.state.get_room_state(room_id)
        if not room_state:
            logger.warning(f"No state found for room {room_id}")
            return

        # Get the relevant agent ID and optional conversation ID
        if self.config.mode == LettaMode.PER_ROOM:
            agent_id = room_state.letta_agent_id
            conversation_id = None
        else:
            agent_id = self.state.shared_agent_id
            conversation_id = room_state.letta_conversation_id

        if agent_id:
            # Trigger memory consolidation
            consolidation_prompt = build_consolidation_prompt(room_id)
            try:
                if conversation_id:
                    # SHARED mode: send via conversation
                    # Note: letta-client always returns a Stream, consume it
                    stream = await asyncio.to_thread(
                        self.client.conversations.messages.create,
                        conversation_id=conversation_id,
                        messages=[{"role": "user", "content": consolidation_prompt}],
                    )
                    # Consume the stream (we don't need the response for consolidation)
                    list(stream)
                else:
                    # PER_ROOM mode: send directly to agent
                    await asyncio.to_thread(
                        self.client.agents.messages.create,
                        agent_id=agent_id,
                        messages=[{"role": "user", "content": consolidation_prompt}],
                        streaming=False,
                    )
                logger.info(f"Memory consolidation triggered for room {room_id}")
            except Exception as e:
                logger.warning(f"Memory consolidation failed: {e}")

            # In SHARED mode, also update room_contexts memory block
            if self.config.mode == LettaMode.SHARED and room_state.summary:
                await self.memory_manager.consolidate_room_memory(
                    agent_id=agent_id,
                    room_id=room_id,
                    summary=room_state.summary,
                )
                # Note: We don't clear current_room anymore since each room
                # has its own conversation - no need for context switching

        # Mark room as inactive (but keep state for potential rejoin)
        self.state.mark_room_inactive(room_id)
        self._state_store.save()

        logger.info(
            f"Room {room_id} marked inactive, agent preserved for future rejoin"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Utilities
    # ══════════════════════════════════════════════════════════════════════════

    def _build_system_prompt(self, mode: LettaMode) -> str:
        """
        Build system prompt with agent identity injected.

        Args:
            mode: Operating mode (PER_ROOM or SHARED)

        Returns:
            System prompt with agent name and identity
        """
        base_prompt = get_system_prompt(mode)

        # Inject agent identity at the beginning
        identity = f"""## Your Identity

You are **{self._thenvoi_agent_name}**.
{self._thenvoi_agent_description or ""}

When sending messages, you must @mention participants by their exact name.
Do NOT mention yourself ({self._thenvoi_agent_name}) - you cannot send messages to yourself.

---

"""
        return identity + base_prompt

    def _filter_participants_for_mentions(
        self, participants: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Filter out the agent itself from participants list.

        The agent should not appear in the list of mentionable participants
        because it cannot send messages to itself.

        Args:
            participants: Full list of room participants

        Returns:
            Filtered list without the agent itself
        """
        return [p for p in participants if p.get("name") != self._thenvoi_agent_name]

    async def _update_agent_participants(
        self,
        agent_id: str,
        participants: list[dict[str, Any]],
    ) -> None:
        """Update participants memory block in agent (excludes agent itself)."""
        filtered = self._filter_participants_for_mentions(participants)
        await self.memory_manager.update_participants(agent_id, filtered)

    def _format_participants(self, participants: list[dict[str, Any]]) -> str:
        """Format participants list for memory block."""
        lines = ["## Current Room Participants\n"]
        for p in participants:
            p_type = p.get("type", "Unknown")
            p_name = p.get("name", "Unknown")
            lines.append(f"- {p_name} ({p_type})")

        lines.append("\nTo mention a participant, use their EXACT name.")
        return "\n".join(lines)

    def _format_time_ago(self, dt: datetime) -> str:
        """Format datetime as human-readable time ago."""
        now = datetime.now(timezone.utc)

        # Ensure dt is timezone-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        diff = now - dt

        if diff.days > 30:
            months = diff.days // 30
            return f"{months} month{'s' if months > 1 else ''}"
        elif diff.days > 0:
            return f"{diff.days} day{'s' if diff.days > 1 else ''}"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''}"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''}"
        else:
            return "just now"

    def _extract_summary(self, content: list[str]) -> str:
        """Extract a brief summary from assistant response."""
        full_text = " ".join(content)

        # Simple heuristic: first 150 chars or first sentence
        if len(full_text) <= 150:
            return full_text.strip()

        # Try to find first sentence
        for end_char in (".", "!", "?"):
            idx = full_text.find(end_char)
            if 0 < idx < 150:
                return full_text[: idx + 1].strip()

        # Fall back to truncation
        return full_text[:147].strip() + "..."
