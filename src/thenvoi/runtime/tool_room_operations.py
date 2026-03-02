"""Room-bound tool operations shared by AgentTools."""

from __future__ import annotations

import logging
from typing import Any

from thenvoi.client.rest import ChatRoomRequest, DEFAULT_REQUEST_OPTIONS

logger = logging.getLogger(__name__)


class RoomToolOperationsMixin:
    """Participant and messaging tool operations for room-bound tools."""

    room_id: str
    rest: Any
    _participants: list[dict[str, Any]]

    async def send_message(
        self, content: str, mentions: list[str] | list[dict[str, str]] | None = None
    ) -> dict[str, Any]:
        """Send a message to the current room."""
        from thenvoi.client.rest import (
            ChatMessageRequest,
            ChatMessageRequestMentionsItem,
        )

        resolved_mentions = self._resolve_mentions(mentions or [])
        logger.debug("Sending message to room %s", self.room_id)

        mention_items = [
            ChatMessageRequestMentionsItem(id=m["id"], handle=m["handle"])
            for m in resolved_mentions
        ]

        response = await self.rest.agent_api_messages.create_agent_chat_message(
            chat_id=self.room_id,
            message=ChatMessageRequest(content=content, mentions=mention_items),
            request_options=DEFAULT_REQUEST_OPTIONS,
        )
        if not response.data:
            raise RuntimeError("Failed to send message - no response data")
        return response.data.model_dump()

    async def send_event(
        self,
        content: str,
        message_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send an event to the current room."""
        from thenvoi.client.rest import ChatEventRequest

        logger.debug("Sending %s event to room %s", message_type, self.room_id)

        response = await self.rest.agent_api_events.create_agent_chat_event(
            chat_id=self.room_id,
            event=ChatEventRequest(
                content=content,
                message_type=message_type,
                metadata=metadata,
            ),
            request_options=DEFAULT_REQUEST_OPTIONS,
        )
        if not response.data:
            raise RuntimeError("Failed to send event - no response data")
        return response.data.model_dump()

    async def create_chatroom(self, task_id: str | None = None) -> str:
        """Create a new chat room."""
        logger.debug("Creating chatroom with task_id=%s", task_id)
        response = await self.rest.agent_api_chats.create_agent_chat(
            chat=ChatRoomRequest(task_id=task_id),
            request_options=DEFAULT_REQUEST_OPTIONS,
        )
        return response.data.id

    async def add_participant(self, name: str, role: str = "member") -> dict[str, Any]:
        """Add a participant to the current room by name."""
        from thenvoi.client.rest import ParticipantRequest

        logger.debug(
            "Adding participant '%s' with role '%s' to room %s",
            name,
            role,
            self.room_id,
        )

        current_participants = await self.get_participants()
        for participant in current_participants:
            if participant.get("name", "").lower() == name.lower():
                logger.debug("Participant '%s' is already in the room", name)
                return {
                    "id": participant["id"],
                    "name": participant["name"],
                    "role": role,
                    "status": "already_in_room",
                }

        peer = await self._lookup_peer_by_name(name)
        if not peer:
            raise ValueError(
                f"Participant '{name}' not found. Use thenvoi_lookup_peers to find available peers."
            )

        participant_id = peer["id"]
        logger.debug("Resolved '%s' to ID: %s", name, participant_id)

        await self.rest.agent_api_participants.add_agent_chat_participant(
            chat_id=self.room_id,
            participant=ParticipantRequest(participant_id=participant_id, role=role),
            request_options=DEFAULT_REQUEST_OPTIONS,
        )

        self._participants.append(
            {
                "id": participant_id,
                "name": name,
                "type": peer.get("type", "Agent"),
                "handle": peer.get("handle"),
            }
        )
        logger.debug(
            "Updated participant cache: added %s, total=%s",
            name,
            len(self._participants),
        )

        return {
            "id": participant_id,
            "name": name,
            "role": role,
            "status": "added",
        }

    async def remove_participant(self, name: str) -> dict[str, Any]:
        """Remove a participant from the current room by name."""
        logger.debug("Removing participant '%s' from room %s", name, self.room_id)

        participants = await self.get_participants()
        participant = None
        for room_participant in participants:
            if room_participant.get("name", "").lower() == name.lower():
                participant = room_participant
                break

        if not participant:
            raise ValueError(f"Participant '{name}' not found in this room.")

        participant_id = participant["id"]
        logger.debug("Resolved '%s' to ID: %s", name, participant_id)

        await self.rest.agent_api_participants.remove_agent_chat_participant(
            self.room_id,
            participant_id,
            request_options=DEFAULT_REQUEST_OPTIONS,
        )

        self._participants = [
            cached for cached in self._participants if cached.get("id") != participant_id
        ]
        logger.debug(
            "Updated participant cache: removed %s, total=%s",
            name,
            len(self._participants),
        )

        return {
            "id": participant_id,
            "name": name,
            "status": "removed",
        }

    async def lookup_peers(self, page: int = 1, page_size: int = 50) -> dict[str, Any]:
        """Find available peers (agents and users) on the platform."""
        logger.debug("Looking up peers: page=%s, page_size=%s", page, page_size)
        response = await self.rest.agent_api_peers.list_agent_peers(
            page=page,
            page_size=page_size,
            not_in_chat=self.room_id,
            request_options=DEFAULT_REQUEST_OPTIONS,
        )

        peers = []
        if response.data:
            peers = [self._format_peer_entry(peer) for peer in response.data]

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

    async def get_participants(self) -> list[dict[str, Any]]:
        """Get participants in the current room."""
        logger.debug("Getting participants for room %s", self.room_id)
        response = await self.rest.agent_api_participants.list_agent_chat_participants(
            chat_id=self.room_id,
            request_options=DEFAULT_REQUEST_OPTIONS,
        )
        if not response.data:
            return []

        return [
            {
                "id": participant.id,
                "name": participant.name,
                "type": participant.type,
                "handle": getattr(participant, "handle", None),
            }
            for participant in response.data
        ]

    @staticmethod
    def _format_peer_entry(peer: Any) -> dict[str, Any]:
        """Build stable lookup_peers payload with optional peer description."""
        formatted_peer: dict[str, Any] = {
            "id": peer.id,
            "name": peer.name,
            "type": getattr(peer, "type", "Agent"),
            "handle": getattr(peer, "handle", None),
        }

        description = getattr(peer, "description", None)
        if description:
            formatted_peer["description"] = description

        return formatted_peer

    def _resolve_mentions(
        self, mentions: list[str] | list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Resolve mention handles, names, or IDs against cached participants."""
        handle_to_participant = {
            (participant.get("handle") or "").lstrip("@"): participant
            for participant in self._participants
        }
        name_to_participant = {
            participant.get("name"): participant for participant in self._participants
        }
        id_to_participant = {
            participant.get("id"): participant for participant in self._participants
        }

        resolved: list[dict[str, str]] = []
        for mention in mentions:
            if isinstance(mention, str):
                identifier = mention.lstrip("@")
            else:
                if mention.get("id"):
                    resolved.append(
                        {"id": mention["id"], "handle": mention.get("handle", "")}
                    )
                    continue
                raw_identifier = mention.get("handle") or mention.get("name", "")
                identifier = raw_identifier.lstrip("@")

            participant = handle_to_participant.get(identifier)
            if not participant:
                participant = name_to_participant.get(identifier)
            if not participant:
                participant = id_to_participant.get(identifier)

            if not participant:
                available_handles = list(handle_to_participant.keys())
                raise ValueError(
                    f"Unknown participant '{identifier}'. "
                    f"Available handles: {available_handles}"
                )

            resolved.append(
                {
                    "id": participant["id"],
                    "handle": participant.get("handle", ""),
                }
            )

        return resolved

    async def _lookup_peer_by_name(self, name: str) -> dict[str, Any] | None:
        """Find a peer by name, paginating through all results."""
        page = 1
        while True:
            result = await self.lookup_peers(page=page, page_size=100)
            for peer in result["peers"]:
                if peer.get("name", "").lower() == name.lower():
                    return peer

            metadata = result["metadata"]
            if page >= metadata.get("total_pages", 1):
                break
            page += 1

        return None
