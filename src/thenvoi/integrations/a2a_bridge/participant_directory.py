"""Participant cache service for bridge message dispatch."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypedDict

from thenvoi.client.rest import DEFAULT_REQUEST_OPTIONS

if TYPE_CHECKING:
    from thenvoi.client.streaming import ParticipantAddedPayload, ParticipantRemovedPayload
    from thenvoi.platform.link import ThenvoiLink

logger = logging.getLogger(__name__)


class ParticipantRecord(TypedDict):
    """Typed dict for participant cache entries."""

    id: str
    name: str
    type: str


class ParticipantDirectory:
    """Cache-backed participant directory with REST fallback."""

    def __init__(
        self,
        link: "ThenvoiLink",
        cache: dict[str, list[ParticipantRecord]],
    ) -> None:
        self._link = link
        self._cache = cache

    def set_link(self, link: "ThenvoiLink") -> None:
        """Update the active link client used for REST lookups."""
        self._link = link

    def clear(self) -> None:
        self._cache.clear()

    def remove_room(self, room_id: str) -> None:
        self._cache.pop(room_id, None)

    async def preload_room(self, room_id: str) -> None:
        """Fetch and cache participants for a room."""
        try:
            self._cache[room_id] = await self.fetch_room(room_id)
        except Exception:
            logger.warning(
                "Failed to cache participants for room %s",
                room_id,
                exc_info=True,
            )

    async def get_for_room(self, room_id: str) -> list[ParticipantRecord]:
        """Return participants for a room with cache fallback."""
        if room_id in self._cache:
            return self._cache[room_id]

        try:
            participants = await self.fetch_room(room_id)
            self._cache[room_id] = participants
            return participants
        except Exception:
            logger.warning(
                "Failed to fetch participants for room %s",
                room_id,
                exc_info=True,
            )
            return []

    async def fetch_room(self, room_id: str) -> list[ParticipantRecord]:
        """Fetch participants from the platform."""
        response = (
            await self._link.rest.agent_api_participants.list_agent_chat_participants(
                chat_id=room_id,
                request_options=DEFAULT_REQUEST_OPTIONS,
            )
        )
        if not response.data:
            return []

        return [
            ParticipantRecord(id=participant.id, name=participant.name, type=participant.type)
            for participant in response.data
        ]

    def on_participant_added(
        self,
        room_id: str,
        payload: "ParticipantAddedPayload",
    ) -> None:
        """Update cache on participant-added events."""
        cached = self._cache.get(room_id)
        if cached is None:
            return
        if not any(participant["id"] == payload.id for participant in cached):
            cached.append(
                ParticipantRecord(
                    id=payload.id,
                    name=payload.name,
                    type=payload.type,
                )
            )

    def on_participant_removed(
        self,
        room_id: str,
        payload: "ParticipantRemovedPayload",
    ) -> None:
        """Update cache on participant-removed events."""
        cached = self._cache.get(room_id)
        if cached is None:
            return
        self._cache[room_id] = [
            participant for participant in cached if participant["id"] != payload.id
        ]

    @staticmethod
    def resolve_sender_name(
        participants: list[ParticipantRecord],
        sender_id: str,
    ) -> str | None:
        """Resolve sender display name from cached participants."""
        return next(
            (participant["name"] for participant in participants if participant["id"] == sender_id),
            None,
        )


__all__ = ["ParticipantDirectory", "ParticipantRecord"]
