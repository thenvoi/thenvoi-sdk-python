"""Kore.ai history converter (metadata-only)."""

from __future__ import annotations

import logging
from typing import Any

from thenvoi.core.protocols import HistoryConverter
from thenvoi.integrations.koreai.types import KoreAISessionState

logger = logging.getLogger(__name__)


class KoreAIHistoryConverter(HistoryConverter[KoreAISessionState]):
    """Extracts Kore.ai session state from platform history.

    Unlike converters for LLM-based adapters, this converter does not
    transform history into a chat format. It scans for the most recent
    task event containing Kore.ai session metadata so the adapter can
    restore session state after a process restart.

    Output: KoreAISessionState with koreai_identity and koreai_last_activity.
    """

    def convert(self, raw: list[dict[str, Any]]) -> KoreAISessionState:
        """Extract most recent Kore.ai session state from history.

        Scans history in reverse to find the latest task event with
        ``koreai_identity`` and ``koreai_last_activity`` in metadata.

        Args:
            raw: Raw platform history (list of message dicts).

        Returns:
            KoreAISessionState with extracted session data.
        """
        identity: str | None = None
        last_activity: float | None = None

        logger.debug("KoreAIHistoryConverter: scanning %d messages", len(raw))

        for msg in reversed(raw):
            msg_type = msg.get("message_type")
            if msg_type == "task":
                metadata = msg.get("metadata", {})
                if "koreai_identity" in metadata:
                    identity = metadata.get("koreai_identity")
                    raw_activity = metadata.get("koreai_last_activity")
                    if raw_activity is not None:
                        try:
                            last_activity = float(raw_activity)
                        except (ValueError, TypeError):
                            logger.warning(
                                "Invalid koreai_last_activity value: %r",
                                raw_activity,
                            )
                    break

        logger.debug(
            "KoreAIHistoryConverter result: identity=%s, last_activity=%s",
            identity,
            last_activity,
        )

        return KoreAISessionState(
            koreai_identity=identity,
            koreai_last_activity=last_activity,
        )
