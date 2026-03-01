"""Codex integration types."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class CodexSessionState:
    """Session state extracted from platform history for Codex rehydration."""

    thread_id: str | None = None
    room_id: str | None = None
    created_at: datetime | None = None

    def has_thread(self) -> bool:
        """Return True when a persisted codex thread_id is available."""
        return bool(self.thread_id)
