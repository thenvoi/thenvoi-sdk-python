"""State helpers for Claude SDK MCP tool handlers."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_execution(agent: Any, room_id: str) -> Any | None:
    runtime = getattr(agent, "runtime", None)
    executions = runtime.executions if runtime else {}
    return executions.get(room_id)


def get_tools(agent: Any, room_id: str) -> Any:
    """Get AgentTools for a room, with participants from execution context."""
    from thenvoi.runtime.tools import AgentTools

    execution = get_execution(agent, room_id)
    participants = execution.participants if execution else []
    return AgentTools(room_id, agent.link.rest, participants)


def get_participant_handles(agent: Any, room_id: str) -> list[str]:
    """Get list of participant handles in room."""
    execution = get_execution(agent, room_id)
    participants = execution.participants if execution else []
    return [participant.get("handle", "") for participant in participants if participant.get("handle")]


def parse_mention_handles(mentions: Any) -> list[str]:
    """Parse mention handles from either list or JSON-encoded list."""
    if not mentions:
        return []
    if isinstance(mentions, list):
        return [str(handle) for handle in mentions]
    if isinstance(mentions, str):
        try:
            parsed = json.loads(mentions)
        except json.JSONDecodeError as error:
            logger.warning("Failed to parse mentions payload: %s", error)
            return []
        if isinstance(parsed, list):
            return [str(handle) for handle in parsed]
    return []


def update_participants_cache_for_add(
    agent: Any,
    room_id: str,
    added_participant: dict[str, Any],
) -> None:
    """Update runtime participant cache to avoid mention races after add."""
    execution = get_execution(agent, room_id)
    if not execution:
        return

    execution.add_participant(
        {
            "id": added_participant["id"],
            "name": added_participant["name"],
            "type": "Agent",
        }
    )


def update_participants_cache_for_remove(
    agent: Any,
    room_id: str,
    removed_participant_id: str,
) -> None:
    """Update runtime participant cache to avoid mention races after remove."""
    execution = get_execution(agent, room_id)
    if execution:
        execution.remove_participant(removed_participant_id)
