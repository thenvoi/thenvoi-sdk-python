"""
Persistent state management for Letta adapter.

Stores:
- Room -> Letta Agent ID mappings (per-room mode)
- Shared agent ID (shared mode)
- Per-room state (last interaction, summaries, active flag)

Storage is file-based JSON by default, designed for easy migration to DB.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RoomState:
    """
    Per-room state tracking.

    Used in both modes to track room-specific metadata.
    In PER_ROOM mode, also stores the dedicated letta_agent_id.
    """

    room_id: str
    """Thenvoi room ID."""

    letta_agent_id: str | None = None
    """
    Letta agent ID for this room (PER_ROOM mode only).
    In SHARED mode, this is None - use LettaAdapterState.shared_agent_id.
    """

    letta_conversation_id: str | None = None
    """
    Letta conversation ID for this room (SHARED mode only).
    Each room maps to one conversation on the shared agent.
    In PER_ROOM mode, this is None - messages go directly to agent.
    """

    last_message_id: str | None = None
    """ID of last processed message in this room."""

    last_interaction: datetime | None = None
    """Timestamp of last interaction in this room."""

    summary: str | None = None
    """
    Brief summary of recent conversation topic.
    Updated after each message, consolidated on room exit.
    """

    participants_snapshot: list[str] = field(default_factory=list)
    """List of participant names at last interaction."""

    is_active: bool = True
    """
    Whether agent is currently active in this room.
    False when agent leaves room (but we keep state for potential rejoin).
    """

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When this room state was created."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "room_id": self.room_id,
            "letta_agent_id": self.letta_agent_id,
            "letta_conversation_id": self.letta_conversation_id,
            "last_message_id": self.last_message_id,
            "last_interaction": (
                self.last_interaction.isoformat() if self.last_interaction else None
            ),
            "summary": self.summary,
            "participants_snapshot": self.participants_snapshot,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoomState:
        """Deserialize from dictionary."""

        def parse_datetime(val: str | None) -> datetime | None:
            if val is None:
                return None
            return datetime.fromisoformat(val)

        return cls(
            room_id=data["room_id"],
            letta_agent_id=data.get("letta_agent_id"),
            letta_conversation_id=data.get("letta_conversation_id"),
            last_message_id=data.get("last_message_id"),
            last_interaction=parse_datetime(data.get("last_interaction")),
            summary=data.get("summary"),
            participants_snapshot=data.get("participants_snapshot", []),
            is_active=data.get("is_active", True),
            created_at=parse_datetime(data.get("created_at"))
            or datetime.now(timezone.utc),
        )

    def mark_interaction(self, message_id: str, participants: list[str]) -> None:
        """Update state after an interaction."""
        self.last_message_id = message_id
        self.last_interaction = datetime.now(timezone.utc)
        self.participants_snapshot = participants
        self.is_active = True


@dataclass
class LettaAdapterState:
    """
    Complete adapter state - persisted to disk/DB.

    Supports both modes:
    - PER_ROOM: each room_state has its own letta_agent_id
    - SHARED: shared_agent_id used for all rooms
    """

    # For SHARED mode
    shared_agent_id: str | None = None
    """Letta agent ID used across all rooms (SHARED mode only)."""

    # Per-room state (used in both modes)
    room_states: dict[str, RoomState] = field(default_factory=dict)
    """Room ID -> RoomState mapping."""

    # Metadata
    thenvoi_agent_id: str | None = None
    """Thenvoi agent ID this adapter is running as."""

    mode: str | None = None
    """Operating mode when state was created."""

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When this state was created."""

    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When this state was last updated."""

    # --- Room State Management ---

    def get_room_state(self, room_id: str) -> RoomState | None:
        """Get state for a room, or None if not tracked."""
        return self.room_states.get(room_id)

    def get_or_create_room_state(self, room_id: str) -> RoomState:
        """Get or create state for a room."""
        if room_id not in self.room_states:
            self.room_states[room_id] = RoomState(room_id=room_id)
            self.updated_at = datetime.now(timezone.utc)
        return self.room_states[room_id]

    def set_room_agent(self, room_id: str, letta_agent_id: str) -> None:
        """Associate a Letta agent with a room (PER_ROOM mode)."""
        room_state = self.get_or_create_room_state(room_id)
        room_state.letta_agent_id = letta_agent_id
        self.updated_at = datetime.now(timezone.utc)

    def get_room_agent(self, room_id: str) -> str | None:
        """Get Letta agent ID for a room (PER_ROOM mode)."""
        room_state = self.room_states.get(room_id)
        return room_state.letta_agent_id if room_state else None

    def set_room_conversation(self, room_id: str, conversation_id: str) -> None:
        """Associate a Letta conversation with a room (SHARED mode)."""
        room_state = self.get_or_create_room_state(room_id)
        room_state.letta_conversation_id = conversation_id
        self.updated_at = datetime.now(timezone.utc)

    def get_room_conversation(self, room_id: str) -> str | None:
        """Get Letta conversation ID for a room (SHARED mode)."""
        room_state = self.room_states.get(room_id)
        return room_state.letta_conversation_id if room_state else None

    def mark_room_inactive(self, room_id: str) -> None:
        """Mark room as inactive (agent left)."""
        if room_id in self.room_states:
            self.room_states[room_id].is_active = False
            self.updated_at = datetime.now(timezone.utc)

    def mark_room_active(self, room_id: str) -> None:
        """Mark room as active (agent rejoined)."""
        if room_id in self.room_states:
            self.room_states[room_id].is_active = True
            self.updated_at = datetime.now(timezone.utc)

    def get_active_rooms(self) -> list[str]:
        """Get list of active room IDs."""
        return [
            room_id for room_id, state in self.room_states.items() if state.is_active
        ]

    def get_inactive_rooms(self) -> list[str]:
        """Get list of inactive room IDs (agent left but state preserved)."""
        return [
            room_id
            for room_id, state in self.room_states.items()
            if not state.is_active
        ]

    # --- Serialization ---

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "shared_agent_id": self.shared_agent_id,
            "room_states": {
                room_id: state.to_dict() for room_id, state in self.room_states.items()
            },
            "thenvoi_agent_id": self.thenvoi_agent_id,
            "mode": self.mode,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LettaAdapterState:
        """Deserialize from dictionary."""
        room_states = {
            room_id: RoomState.from_dict(state_data)
            for room_id, state_data in data.get("room_states", {}).items()
        }

        def parse_datetime(val: str | None) -> datetime:
            if val is None:
                return datetime.now(timezone.utc)
            return datetime.fromisoformat(val)

        return cls(
            shared_agent_id=data.get("shared_agent_id"),
            room_states=room_states,
            thenvoi_agent_id=data.get("thenvoi_agent_id"),
            mode=data.get("mode"),
            created_at=parse_datetime(data.get("created_at")),
            updated_at=parse_datetime(data.get("updated_at")),
        )


class StateStore:
    """
    Persistent state storage.

    Currently file-based JSON. Interface designed for easy migration to DB backend.

    Usage:
        store = StateStore("~/.thenvoi/letta_state.json")
        state = store.load()
        state.set_room_agent("room-123", "agent-456")
        store.save()
    """

    def __init__(self, storage_path: Path | str):
        """
        Initialize state store.

        Args:
            storage_path: Path to JSON state file.
                         Supports ~ expansion.
        """
        self.storage_path = Path(storage_path).expanduser()
        self._state: LettaAdapterState | None = None
        self._dirty: bool = False

    def load(self) -> LettaAdapterState:
        """
        Load state from disk.

        Creates new state if file doesn't exist or is corrupted.
        """
        if self._state is not None:
            return self._state

        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text())
                self._state = LettaAdapterState.from_dict(data)
                logger.info(f"Loaded Letta adapter state from {self.storage_path}")
                logger.debug(f"  - Shared agent: {self._state.shared_agent_id}")
                logger.debug(f"  - Rooms tracked: {len(self._state.room_states)}")
            except json.JSONDecodeError as e:
                logger.warning(f"Corrupted state file, starting fresh: {e}")
                self._state = LettaAdapterState()
            except Exception as e:
                logger.warning(f"Failed to load state, starting fresh: {e}")
                self._state = LettaAdapterState()
        else:
            logger.info("No state file found, creating new state")
            self._state = LettaAdapterState()

        return self._state

    def save(self) -> None:
        """
        Persist state to disk.

        Uses atomic write (temp file + rename) to prevent corruption.
        """
        if self._state is None:
            return

        self._state.updated_at = datetime.now(timezone.utc)

        # Ensure directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write
        temp_path = self.storage_path.with_suffix(".tmp")
        try:
            temp_path.write_text(json.dumps(self._state.to_dict(), indent=2))
            temp_path.rename(self.storage_path)
            logger.debug(f"Saved Letta adapter state to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def mark_dirty(self) -> None:
        """Mark state as needing save."""
        self._dirty = True

    def save_if_dirty(self) -> None:
        """Save state if it has been modified."""
        if self._dirty:
            self.save()
            self._dirty = False

    @property
    def state(self) -> LettaAdapterState:
        """Get current state (loads if needed)."""
        if self._state is None:
            self.load()
        return self._state  # type: ignore[return-value]

    def reset(self) -> None:
        """Reset to empty state (for testing)."""
        self._state = LettaAdapterState()
        if self.storage_path.exists():
            self.storage_path.unlink()
