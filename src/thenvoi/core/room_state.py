"""Shared room-scoped state containers for adapter lifecycle consistency."""

from __future__ import annotations

from collections.abc import Callable, Iterator, MutableMapping, MutableSet
from typing import Generic, TypeVar

T = TypeVar("T")


class RoomStateStore(MutableMapping[str, T], Generic[T]):
    """Typed mapping for room-scoped adapter state."""

    def __init__(self) -> None:
        self._values: dict[str, T] = {}

    def __getitem__(self, key: str) -> T:
        return self._values[key]

    def __setitem__(self, key: str, value: T) -> None:
        self._values[key] = value

    def __delitem__(self, key: str) -> None:
        del self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def discard(self, room_id: str) -> None:
        """Remove room state if present."""
        self._values.pop(room_id, None)

    def ensure(self, room_id: str, factory: Callable[[], T]) -> T:
        """Return existing room state or create it via factory."""
        if room_id not in self._values:
            self._values[room_id] = factory()
        return self._values[room_id]


class RoomFlagStore(MutableSet[str]):
    """Set-like room marker store with explicit lifecycle helpers."""

    def __init__(self) -> None:
        self._rooms: set[str] = set()

    def __contains__(self, room_id: object) -> bool:
        return room_id in self._rooms

    def __iter__(self) -> Iterator[str]:
        return iter(self._rooms)

    def __len__(self) -> int:
        return len(self._rooms)

    def add(self, value: str) -> None:
        self._rooms.add(value)

    def discard(self, value: str) -> None:
        self._rooms.discard(value)

