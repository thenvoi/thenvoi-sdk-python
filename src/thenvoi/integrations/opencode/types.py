"""Typed session state for the OpenCode integration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class OpencodeSessionState:
    """Session metadata persisted in Thenvoi task events."""

    session_id: str | None = None
    room_id: str | None = None
    created_at: datetime | None = None
