"""
Core infrastructure for Thenvoi platform integration.

This module contains the fundamental building blocks that ALL agent frameworks use:
- PlatformClient: Platform registration and WebSocket connection
- RoomManager: Room subscription and event handling
"""

from .platform_client import ThenvoiPlatformClient
from .room_manager import RoomManager

__all__ = [
    "ThenvoiPlatformClient",
    "RoomManager",
]
