"""Session/context orchestration for A2A gateway room routing."""

from __future__ import annotations

import logging
from uuid import uuid4

from thenvoi.client.rest import (
    AsyncRestClient,
    ChatEventRequest,
    ChatRoomRequest,
    DEFAULT_REQUEST_OPTIONS,
    ParticipantRequest,
)
from thenvoi.integrations.a2a.gateway.types import GatewaySessionState

logger = logging.getLogger(__name__)


class GatewaySessionManager:
    """Own context-to-room routing and participant membership state."""

    def __init__(self, rest: AsyncRestClient) -> None:
        self._rest = rest
        self.context_to_room: dict[str, str] = {}
        self.room_participants: dict[str, set[str]] = {}

    async def get_or_create_room(
        self,
        context_id: str | None,
        target_peer_id: str,
    ) -> tuple[str, str]:
        """Get existing room for context or create one and join target peer."""
        if context_id is None or context_id not in self.context_to_room:
            response = await self._rest.agent_api_chats.create_agent_chat(
                chat=ChatRoomRequest(),
                request_options=DEFAULT_REQUEST_OPTIONS,
            )
            room_id = response.data.id

            await self._rest.agent_api_participants.add_agent_chat_participant(
                chat_id=room_id,
                participant=ParticipantRequest(
                    participant_id=target_peer_id,
                    role="member",
                ),
                request_options=DEFAULT_REQUEST_OPTIONS,
            )

            context_id = context_id or str(uuid4())
            self.context_to_room[context_id] = room_id
            self.room_participants[room_id] = {target_peer_id}
            logger.info(
                "Created new room %s for context %s with peer %s",
                room_id,
                context_id,
                target_peer_id,
            )
            return room_id, context_id

        room_id = self.context_to_room[context_id]
        if target_peer_id not in self.room_participants.get(room_id, set()):
            await self._rest.agent_api_participants.add_agent_chat_participant(
                chat_id=room_id,
                participant=ParticipantRequest(
                    participant_id=target_peer_id,
                    role="member",
                ),
                request_options=DEFAULT_REQUEST_OPTIONS,
            )
            self.room_participants.setdefault(room_id, set()).add(target_peer_id)
            logger.info(
                "Added peer %s to existing room %s (context=%s)",
                target_peer_id,
                room_id,
                context_id,
            )

        return room_id, context_id

    def rehydrate(self, history: GatewaySessionState) -> None:
        """Restore persisted gateway context mappings from history."""
        for context_id, room_id in history.context_to_room.items():
            if context_id not in self.context_to_room:
                self.context_to_room[context_id] = room_id
                logger.debug("Restored context mapping: %s → %s", context_id, room_id)

        for room_id, participants in history.room_participants.items():
            existing = self.room_participants.get(room_id, set())
            self.room_participants[room_id] = existing | participants

        logger.info(
            "Rehydrated gateway state: %d contexts, %d rooms",
            len(self.context_to_room),
            len(self.room_participants),
        )

    async def emit_context_event(self, room_id: str, context_id: str) -> None:
        """Persist context mapping in room event history for reconnect rehydration."""
        await self._rest.agent_api_events.create_agent_chat_event(
            chat_id=room_id,
            event=ChatEventRequest(
                content="A2A gateway context",
                message_type="task",
                metadata={
                    "gateway_context_id": context_id,
                    "gateway_room_id": room_id,
                },
            ),
        )
