"""Shared utilities and protocols for bridge message handlers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from bridge_core.handler import BaseHandler as Handler

if TYPE_CHECKING:
    from thenvoi.runtime.tools import AgentTools

logger = logging.getLogger(__name__)

__all__ = ["Handler", "resolve_sender"]


def resolve_sender(
    sender_id: str,
    tools: AgentTools,
) -> tuple[str | None, str | None]:
    """Resolve sender display name and handle from pre-cached participants.

    Uses the public ``tools.participants`` property (populated by the
    bridge), avoiding a redundant REST API call.

    Returns:
        Tuple of ``(display_name, handle)``.  Both are ``None`` when the
        sender is not found in the participant cache.  ``handle`` is also
        ``None`` when the participant was added via WebSocket event
        (which doesn't include handle).
    """
    participants: list[dict[str, Any]] = tools.participants
    for p in participants:
        if p.get("id") == sender_id:
            return p.get("name"), p.get("handle")
    logger.debug(
        "Sender %s not found in participant cache (%d entries)",
        sender_id,
        len(participants),
    )
    return None, None
