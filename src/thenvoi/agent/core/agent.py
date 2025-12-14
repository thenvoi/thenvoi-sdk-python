"""
ThenvoiAgent - Coordinator for Thenvoi platform integration.

Manages WebSocket connection, session lifecycle, and message routing.
Each agent gets one coordinator that spawns per-room sessions.

KEY DESIGN:
    - SDK does NOT send messages directly
    - All communication via AgentTools (used by LLM)
    - ThenvoiAgent provides internal methods for AgentTools
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Set

from thenvoi.client.rest import (
    AsyncRestClient,
    ChatMessageRequest,
    ChatMessageRequestMentionsItem,
    ChatEventRequest,
    ParticipantRequest,
)
from thenvoi.client.streaming import (
    WebSocketClient,
    MessageCreatedPayload,
    RoomAddedPayload,
    RoomRemovedPayload,
)

from .types import (
    AgentConfig,
    AgentTools,
    ConversationContext,
    MessageHandler,
    PlatformMessage,
    SessionConfig,
)
from .session import AgentSession

logger = logging.getLogger(__name__)


class ThenvoiAgent:
    """
    Coordinates room lifecycle and message routing.

    Owns shared connections, manages session lifecycle,
    routes messages to correct per-room session.

    NOTE: NO public send_message. All communication via AgentTools.

    Example:
        agent = ThenvoiAgent(
            agent_id="...",
            api_key="...",
        )

        async def handler(msg: PlatformMessage, tools: AgentTools):
            # Handler receives tools, not agent
            await tools.send_message("Hello!")

        await agent.start(on_message=handler)
    """

    def __init__(
        self,
        agent_id: str,
        api_key: str,
        ws_url: str = "wss://api.thenvoi.com/ws",
        rest_url: str = "https://api.thenvoi.com",
        config: AgentConfig | None = None,
        session_config: SessionConfig | None = None,
        on_session_cleanup: Callable[[str], Awaitable[None]] | None = None,
    ):
        """
        Initialize ThenvoiAgent coordinator.

        Args:
            agent_id: Agent ID from Thenvoi platform
            api_key: Agent-specific API key
            ws_url: WebSocket URL
            rest_url: REST API base URL
            config: Agent configuration
            session_config: Default config for new sessions
            on_session_cleanup: Optional async callback for session cleanup (receives room_id).
                               Used by adapters to clean up framework-specific state (e.g., checkpointer).
        """
        self.agent_id = agent_id
        self.api_key = api_key
        self.ws_url = ws_url
        self.rest_url = rest_url
        self.config = config or AgentConfig()
        self._session_config = session_config or SessionConfig()
        self._on_session_cleanup = on_session_cleanup

        # Create API client
        self._api_client = AsyncRestClient(
            api_key=self.api_key,
            base_url=self.rest_url,
        )

        # These are populated by start()
        self._name: str = ""
        self._description: str = ""
        self._ws_client: WebSocketClient | None = None
        self._sessions: dict[str, AgentSession] = {}
        self._on_message: MessageHandler | None = None
        self._is_running = False
        self._subscribed_rooms: Set[str] = set()

    @property
    def agent_name(self) -> str:
        """Agent name from platform."""
        return self._name

    @property
    def agent_description(self) -> str:
        """Agent description from platform."""
        return self._description

    @property
    def is_running(self) -> bool:
        """Check if agent is running."""
        return self._is_running

    @property
    def active_sessions(self) -> dict[str, AgentSession]:
        """Get active sessions by room_id."""
        return self._sessions.copy()

    async def start(self, on_message: MessageHandler) -> None:
        """
        Start the agent coordinator.

        1. Fetch agent metadata from platform
        2. Connect WebSocket
        3. Subscribe to room events (added/removed)
        4. Subscribe to existing rooms
        5. Create sessions for existing rooms

        Args:
            on_message: Async callback for handling messages.
                        Receives (PlatformMessage, AgentTools).
        """
        if self._is_running:
            logger.warning("Agent already running")
            return

        self._on_message = on_message

        # 1. Fetch agent metadata
        logger.info(f"Starting agent: {self.agent_id}")
        await self._fetch_agent_metadata()

        # 2. Connect WebSocket
        self._ws_client = WebSocketClient(
            self.ws_url,
            self.api_key,
            self.agent_id,
        )
        await self._ws_client.__aenter__()

        # 3. Subscribe to room events
        await self._ws_client.join_agent_rooms_channel(
            self.agent_id,
            on_room_added=self._on_room_added,
            on_room_removed=self._on_room_removed,
        )

        # 4. Subscribe to existing rooms (if configured)
        if self.config.auto_subscribe_existing_rooms:
            await self._subscribe_to_existing_rooms()

        self._is_running = True
        logger.info(f"Agent {self._name} started successfully")

    async def stop(self) -> None:
        """Stop all sessions and disconnect."""
        if not self._is_running:
            return

        logger.info(f"Stopping agent: {self._name}")
        self._is_running = False

        # Stop all sessions
        for room_id in list(self._sessions.keys()):
            await self._destroy_session(room_id)

        # Disconnect WebSocket
        if self._ws_client:
            await self._ws_client.__aexit__(None, None, None)
            self._ws_client = None

        logger.info("Agent stopped")

    async def run(self) -> None:
        """
        Run the agent until stopped or interrupted.

        This keeps the WebSocket connection alive and processes messages.
        """
        if not self._is_running or not self._ws_client:
            raise RuntimeError("Agent not started. Call start() first.")

        try:
            await self._ws_client.run_forever()
        except Exception as e:
            logger.error(f"Agent run error: {e}")
            raise

    # --- Internal methods (used by AgentTools) ---

    async def _send_message_internal(
        self, room_id: str, content: str, mentions: list[dict[str, str]]
    ) -> dict[str, Any]:
        """
        Internal: Send message via REST.

        Used by AgentTools.send_message().

        Args:
            room_id: Target room ID
            content: Message content
            mentions: List of mentions with 'id' and 'name' keys

        Returns:
            Full API response as dict (includes id, content, sender info, etc.)
        """
        logger.debug(f"Sending message to room {room_id}")

        # Convert dict mentions to ChatMessageRequestMentionsItem
        mention_items = [
            ChatMessageRequestMentionsItem(id=m["id"], name=m["name"]) for m in mentions
        ]

        response = await self._api_client.agent_api.create_agent_chat_message(
            chat_id=room_id,
            message=ChatMessageRequest(content=content, mentions=mention_items),
        )
        if not response.data:
            raise RuntimeError("Failed to send message - no response data")
        return response.data.model_dump()

    async def _send_event_internal(
        self,
        room_id: str,
        content: str,
        message_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Internal: Send event via REST.

        Used by AgentTools.send_event().
        Events don't require mentions - used for tool_call, tool_result, error, thought, task.

        Args:
            room_id: Target room ID
            content: Human-readable event content
            message_type: One of: tool_call, tool_result, thought, error, task
            metadata: Optional structured data for the event

        Returns:
            Full API response as dict
        """
        logger.debug(f"Sending {message_type} event to room {room_id}")

        response = await self._api_client.agent_api.create_agent_chat_event(
            chat_id=room_id,
            event=ChatEventRequest(
                content=content,
                message_type=message_type,
                metadata=metadata,
            ),
        )
        if not response.data:
            raise RuntimeError("Failed to send event - no response data")
        return response.data.model_dump()

    async def _create_chatroom_internal(self, name: str) -> str:
        """
        Internal: Create chatroom via REST.

        Used by AgentTools.create_chatroom().
        """
        logger.debug(f"Creating chatroom: {name}")
        # Note: This would need the actual API method
        # For now, placeholder
        raise NotImplementedError("create_chatroom not yet implemented in REST client")

    async def _add_participant_internal(
        self, room_id: str, name: str, role: str = "member"
    ) -> dict[str, Any]:
        """
        Internal: Add participant via REST by name.

        Looks up participant ID by name from peers, then adds to room.
        Used by AgentTools.add_participant().

        Args:
            room_id: Room to add participant to
            name: Participant name (agent or user)
            role: Role in room - "owner", "admin", or "member" (default)

        Returns:
            Dict with participant info that was added

        Raises:
            ValueError: If participant not found by name
        """
        logger.debug(
            f"Adding participant '{name}' with role '{role}' to room {room_id}"
        )

        # Look up participant ID by name (paginates through all peers)
        participant = await self._lookup_peer_by_name(name)
        if not participant:
            raise ValueError(
                f"Participant '{name}' not found. Use lookup_peers to find available peers."
            )

        participant_id = participant["id"]
        logger.debug(f"Resolved '{name}' to ID: {participant_id}")

        await self._api_client.agent_api.add_agent_chat_participant(
            chat_id=room_id,
            participant=ParticipantRequest(participant_id=participant_id, role=role),
        )

        return {
            "id": participant_id,
            "name": name,
            "role": role,
            "status": "added",
        }

    async def _remove_participant_internal(
        self, room_id: str, name: str
    ) -> dict[str, Any]:
        """
        Internal: Remove participant via REST by name.

        Looks up participant ID by name from room participants, then removes.
        Used by AgentTools.remove_participant().

        Args:
            room_id: Room to remove participant from
            name: Participant name to remove

        Returns:
            Dict with removed participant info

        Raises:
            ValueError: If participant not found in room
        """
        logger.debug(f"Removing participant '{name}' from room {room_id}")

        # Look up participant ID by name from current room participants
        participants = await self._get_participants_internal(room_id)
        participant = None
        for p in participants:
            if p.get("name", "").lower() == name.lower():
                participant = p
                break

        if not participant:
            raise ValueError(f"Participant '{name}' not found in this room.")

        participant_id = participant["id"]
        logger.debug(f"Resolved '{name}' to ID: {participant_id}")

        await self._api_client.agent_api.remove_agent_chat_participant(
            room_id,
            participant_id,
        )

        return {
            "id": participant_id,
            "name": name,
            "status": "removed",
        }

    async def _lookup_peers_internal(
        self,
        page: int = 1,
        page_size: int = 50,
        not_in_chat: str | None = None,
    ) -> dict[str, Any]:
        """
        Internal: Lookup available peers via REST with pagination.

        Used by AgentTools.lookup_peers().

        Args:
            page: Page number (default 1)
            page_size: Items per page (default 50)
            not_in_chat: Optional chat ID to exclude peers already in that chat

        Returns:
            Dict with 'peers' list and 'metadata' (page, page_size, total_count, total_pages)
        """
        logger.debug(f"Looking up peers: page={page}, page_size={page_size}")
        response = await self._api_client.agent_api.list_agent_peers(
            page=page,
            page_size=page_size,
            not_in_chat=not_in_chat,
        )

        peers = []
        if response.data:
            peers = [
                {
                    "id": peer.id,
                    "name": peer.name,
                    "type": getattr(peer, "type", "Agent"),
                    "description": peer.description,
                }
                for peer in response.data
            ]

        metadata = {
            "page": response.metadata.page if response.metadata else page,
            "page_size": response.metadata.page_size
            if response.metadata
            else page_size,
            "total_count": response.metadata.total_count
            if response.metadata
            else len(peers),
            "total_pages": response.metadata.total_pages if response.metadata else 1,
        }

        return {"peers": peers, "metadata": metadata}

    async def _lookup_peer_by_name(self, name: str) -> dict[str, Any] | None:
        """
        Internal: Find a peer by name, paginating through all results.

        Args:
            name: Name to search for (case-insensitive)

        Returns:
            Peer dict if found, None otherwise
        """
        page = 1
        while True:
            result = await self._lookup_peers_internal(page=page, page_size=100)
            for peer in result["peers"]:
                if peer.get("name", "").lower() == name.lower():
                    return peer

            # Check if more pages
            metadata = result["metadata"]
            if page >= metadata.get("total_pages", 1):
                break
            page += 1

        return None

    async def _get_participants_internal(self, room_id: str) -> list[dict[str, Any]]:
        """
        Internal: Get participants via REST.

        Used by AgentTools.get_participants().
        """
        logger.debug(f"Getting participants for room {room_id}")
        response = await self._api_client.agent_api.list_agent_chat_participants(
            chat_id=room_id,
        )
        if not response.data:
            return []

        return [
            {
                "id": p.id,
                "name": p.name,
                "type": p.type,
            }
            for p in response.data
        ]

    # --- Message lifecycle (SDK operations) ---

    async def _mark_processing(self, message_id: str, room_id: str) -> None:
        """Mark message as being processed."""
        logger.debug(f"Marking message {message_id} as processing")
        try:
            await self._api_client.agent_api.mark_agent_message_processing(
                chat_id=room_id,
                id=message_id,
            )
        except Exception as e:
            logger.warning(f"Failed to mark message {message_id} as processing: {e}")

    async def _mark_processed(self, message_id: str, room_id: str) -> None:
        """Mark message as successfully processed."""
        logger.debug(f"Marking message {message_id} as processed")
        try:
            await self._api_client.agent_api.mark_agent_message_processed(
                chat_id=room_id,
                id=message_id,
            )
        except Exception as e:
            logger.warning(f"Failed to mark message {message_id} as processed: {e}")

    async def _mark_failed(self, message_id: str, room_id: str, error: str) -> None:
        """Mark message as failed."""
        logger.warning(f"Marking message {message_id} as failed: {error}")
        try:
            await self._api_client.agent_api.mark_agent_message_failed(
                chat_id=room_id,
                id=message_id,
                error=error,
            )
        except Exception as e:
            logger.warning(f"Failed to mark message {message_id} as failed: {e}")

    async def _get_next_message(self, room_id: str) -> PlatformMessage | None:
        """
        Get next unprocessed message for room.

        Called during session backlog sync.

        Returns:
            PlatformMessage if there's an unprocessed message
            None if no unprocessed messages (204 No Content) or on error
        """
        from thenvoi_rest.core.api_error import ApiError

        logger.debug(f"Getting next message for room {room_id}")
        try:
            response = await self._api_client.agent_api.get_agent_next_message(
                chat_id=room_id,
            )
            if response.data is None:
                return None

            msg = response.data
            return PlatformMessage(
                id=msg.id,
                room_id=msg.chat_room_id or "",
                content=msg.content,
                sender_id=msg.sender_id,
                sender_type=msg.sender_type,
                sender_name=msg.sender_name or "",
                message_type=msg.message_type,
                metadata=msg.metadata or {},
                created_at=msg.inserted_at or datetime.now(timezone.utc),
            )
        except ApiError as e:
            # 204 No Content means no unprocessed messages - this is expected
            if e.status_code == 204:
                logger.debug(f"No unprocessed messages for room {room_id}")
                return None
            logger.warning(f"Failed to get next message for room {room_id}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to get next message for room {room_id}: {e}")
            return None

    # --- Context operations ---

    async def get_context(self, room_id: str) -> ConversationContext:
        """
        Get context for a room.

        Delegates to session if exists, otherwise fetches directly.
        """
        # If session exists, delegate to it (uses cache)
        if room_id in self._sessions:
            return await self._sessions[room_id].get_context()

        # Otherwise, fetch directly
        return await self._fetch_context(room_id)

    async def _fetch_context(self, room_id: str) -> ConversationContext:
        """Fetch context from API."""
        # Get context (messages + events visible to this agent)
        context_response = await self._api_client.agent_api.get_agent_chat_context(
            chat_id=room_id,
        )
        messages = []
        if context_response.data:
            for item in context_response.data:
                # Extract sender_name if available
                sender_name = getattr(item, "sender_name", None) or getattr(
                    item, "name", None
                )
                messages.append(
                    {
                        "id": item.id,
                        "content": getattr(item, "content", ""),
                        "sender_id": getattr(item, "sender_id", ""),
                        "sender_type": getattr(item, "sender_type", ""),
                        "sender_name": sender_name,
                        "message_type": getattr(item, "message_type", "text"),
                        "created_at": getattr(item, "inserted_at", None),
                    }
                )

        # Get participants
        participants = await self._get_participants_internal(room_id)

        return ConversationContext(
            room_id=room_id,
            messages=messages,
            participants=participants,
            hydrated_at=datetime.now(timezone.utc),
        )

    # --- Session management ---

    def _create_agent_tools(self, room_id: str) -> AgentTools:
        """Create AgentTools instance for a room."""
        return AgentTools(room_id=room_id, coordinator=self)

    async def _on_room_added(self, room_data: RoomAddedPayload) -> None:
        """Handle room_added event - create and start session."""
        room_id = room_data.id
        logger.info(
            f"[room_added] Agent added to room: {room_id} (title: {room_data.title})"
        )

        # Subscribe to room messages
        await self._subscribe_to_room(room_id)

        # Create session
        await self._create_session(room_id)

    async def _on_room_removed(self, room_data: RoomRemovedPayload) -> None:
        """Handle room_removed event - stop and cleanup session."""
        room_id = room_data.id
        logger.info(
            f"[room_removed] Agent removed from room: {room_id} (title: {room_data.title})"
        )

        # Unsubscribe from room
        await self._unsubscribe_from_room(room_id)

        # Destroy session
        await self._destroy_session(room_id)

    async def _on_message_created(self, msg_data: MessageCreatedPayload) -> None:
        """Handle message_created event - route to session."""
        room_id = msg_data.chat_room_id

        # Ignore own messages - this should never happen, backend should filter these
        if msg_data.sender_id == self.agent_id:
            logger.error(
                "Received own message via WebSocket - backend should filter these. "
                f"message_id={msg_data.id}, room_id={room_id}"
            )
            return

        # Convert to PlatformMessage
        msg = PlatformMessage(
            id=msg_data.id,
            room_id=room_id,
            content=msg_data.content,
            sender_id=msg_data.sender_id,
            sender_type=msg_data.sender_type,
            sender_name=None,  # Will be hydrated if needed
            message_type=msg_data.message_type,
            metadata={
                "mentions": [
                    {"id": m.id, "username": m.username}
                    for m in msg_data.metadata.mentions
                ],
                "status": msg_data.metadata.status,
            },
            created_at=datetime.fromisoformat(
                msg_data.inserted_at.replace("Z", "+00:00")
            ),
        )

        # Route to session
        if room_id in self._sessions:
            self._sessions[room_id].enqueue_message(msg)
        else:
            logger.warning(f"No session for room {room_id}, message dropped")

    async def _create_session(self, room_id: str) -> AgentSession:
        """Create and start a new session for a room."""
        if room_id in self._sessions:
            logger.debug(f"Session already exists for room {room_id}")
            return self._sessions[room_id]

        if not self._on_message:
            raise RuntimeError("No message handler set")

        session = AgentSession(
            room_id=room_id,
            api_client=self._api_client,
            on_message=self._on_message,
            coordinator=self,
            config=self._session_config,
        )

        self._sessions[room_id] = session
        await session.start()

        logger.debug(f"Created session for room {room_id}")
        return session

    async def _destroy_session(self, room_id: str) -> None:
        """Stop and cleanup session for a room."""
        if room_id not in self._sessions:
            return

        session = self._sessions.pop(room_id)
        await session.stop()

        # Call cleanup callback (for adapter to clean up checkpointer, etc.)
        if self._on_session_cleanup:
            try:
                await self._on_session_cleanup(room_id)
            except Exception as e:
                logger.warning(f"Session cleanup callback failed for {room_id}: {e}")

        logger.debug(f"Destroyed session for room {room_id}")

    async def _subscribe_to_room(self, room_id: str) -> None:
        """Subscribe to a room's message and participants channels."""
        if not self._ws_client:
            raise RuntimeError("WebSocket not connected")

        if room_id in self._subscribed_rooms:
            return

        # Subscribe to messages
        await self._ws_client.join_chat_room_channel(
            room_id,
            on_message_created=self._on_message_created,
        )

        # Subscribe to participant updates
        await self._ws_client.join_room_participants_channel(
            room_id,
            on_participant_added=lambda p: self._on_participant_added(room_id, p),
            on_participant_removed=lambda p: self._on_participant_removed(room_id, p),
        )

        self._subscribed_rooms.add(room_id)
        logger.debug(f"Subscribed to room {room_id}")

    async def _unsubscribe_from_room(self, room_id: str) -> None:
        """Unsubscribe from a room's message and participants channels."""
        if not self._ws_client:
            return

        if room_id not in self._subscribed_rooms:
            return

        # Mark as unsubscribed first to prevent duplicate calls
        self._subscribed_rooms.discard(room_id)

        try:
            await self._ws_client.leave_chat_room_channel(room_id)
        except Exception as e:
            logger.warning(f"Error unsubscribing from chat_room:{room_id}: {e}")

        try:
            await self._ws_client.leave_room_participants_channel(room_id)
        except Exception as e:
            logger.warning(f"Error unsubscribing from room_participants:{room_id}: {e}")

        logger.debug(f"Unsubscribed from room {room_id}")

    async def _on_participant_added(self, room_id: str, participant: dict) -> None:
        """Handle participant_added event - update session's participant list.

        NOTE: This is for OTHER participants, not the agent itself.
        When the agent is added to a room, we receive room_added instead.
        """
        participant_id = participant.get("id")

        # Ignore if it's the agent itself (handled by room_added)
        if participant_id == self.agent_id:
            return

        participant_name = participant.get("name", participant_id or "unknown")
        logger.info(f"[participant_added] {participant_name} added to room {room_id}")
        if room_id in self._sessions:
            self._sessions[room_id].add_participant(participant)

    async def _on_participant_removed(self, room_id: str, participant: dict) -> None:
        """Handle participant_removed event - update session's participant list.

        NOTE: This is for OTHER participants, not the agent itself.
        When the agent is removed from a room, we receive room_removed instead.
        """
        participant_id = participant.get("id")

        # Ignore if it's the agent itself (handled by room_removed)
        if participant_id == self.agent_id:
            return

        participant_name = participant.get("name", participant_id or "unknown")
        logger.info(
            f"[participant_removed] {participant_name} removed from room {room_id}"
        )
        if room_id in self._sessions:
            self._sessions[room_id].remove_participant(participant)

    async def _subscribe_to_existing_rooms(self) -> None:
        """Subscribe to all rooms where agent is a participant."""
        logger.debug("Subscribing to existing rooms")

        response = await self._api_client.agent_api.list_agent_chats()
        if not response.data:
            return

        for room in response.data:
            await self._subscribe_to_room(room.id)
            await self._create_session(room.id)

        logger.info(f"Subscribed to {len(response.data)} existing rooms")

    async def _fetch_agent_metadata(self) -> None:
        """Fetch agent metadata from platform."""
        response = await self._api_client.agent_api.get_agent_me()

        if not response.data:
            raise RuntimeError("Failed to fetch agent metadata")

        agent = response.data
        if not agent.description:
            raise ValueError(f"Agent {self.agent_id} has no description")

        self._name = agent.name
        self._description = agent.description

        logger.debug(f"Fetched metadata for agent: {self._name}")
