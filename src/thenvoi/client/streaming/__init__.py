"""Thenvoi WebSocket streaming SDK.

This module provides WebSocket-based real-time communication with the Thenvoi platform.

Usage:
    from thenvoi.client.streaming import WebSocketClient
"""

from thenvoi.client.streaming.client import (
    KNOWN_DISCONNECT_REASONS,
    WebSocketClient,
    MessageCreatedPayload,
    RoomAddedPayload,
    RoomRemovedPayload,
    RoomDeletedPayload,
    ParticipantAddedPayload,
    ParticipantRemovedPayload,
    MessageMetadata,
    Mention,
    ContactRequestReceivedPayload,
    ContactRequestUpdatedPayload,
    ContactAddedPayload,
    ContactRemovedPayload,
    extract_disconnect_reason,
    humanize_disconnect_reason,
)

__all__ = [
    "KNOWN_DISCONNECT_REASONS",
    "WebSocketClient",
    "MessageCreatedPayload",
    "RoomAddedPayload",
    "RoomRemovedPayload",
    "RoomDeletedPayload",
    "ParticipantAddedPayload",
    "ParticipantRemovedPayload",
    "MessageMetadata",
    "Mention",
    "ContactRequestReceivedPayload",
    "ContactRequestUpdatedPayload",
    "ContactAddedPayload",
    "ContactRemovedPayload",
    "extract_disconnect_reason",
    "humanize_disconnect_reason",
]
