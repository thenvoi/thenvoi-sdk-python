"""OpenCode history converter."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from thenvoi.core.protocols import HistoryConverter
from thenvoi.integrations.opencode.types import OpencodeSessionState

logger = logging.getLogger(__name__)


class OpencodeHistoryConverter(HistoryConverter["OpencodeSessionState"]):
    """Extract the latest OpenCode session metadata from task events."""

    def set_agent_name(self, name: str) -> None:
        """No-op for metadata-only converter compatibility."""

    def convert(self, raw: list[dict[str, Any]]) -> OpencodeSessionState:
        logger.debug("OpencodeHistoryConverter: scanning %d messages", len(raw))

        for msg in reversed(raw):
            if msg.get("message_type") != "task":
                continue

            metadata = msg.get("metadata") or {}
            if not isinstance(metadata, dict):
                continue

            session_id = metadata.get("opencode_session_id")
            if not session_id:
                continue

            created_at = self._parse_iso_datetime(metadata.get("opencode_created_at"))
            return OpencodeSessionState(
                session_id=str(session_id),
                room_id=self._optional_str(metadata.get("opencode_room_id")),
                created_at=created_at,
            )

        return OpencodeSessionState()

    @staticmethod
    def _parse_iso_datetime(value: Any) -> datetime | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        if value is None:
            return None
        return str(value)
