"""
Thenvoi Platform Layer - Wire-level connection to Thenvoi platform.

Components:
    ThenvoiLink: WebSocket connection + event dispatch (REST via .rest)
    PlatformEvent: Single event type for all platform events
"""

from .event import PlatformEvent
from .link import ThenvoiLink

__all__ = [
    "ThenvoiLink",
    "PlatformEvent",
]
