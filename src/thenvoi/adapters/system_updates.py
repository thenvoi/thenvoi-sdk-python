"""Compatibility wrappers for canonical metadata-update helpers.

Prefer using `SimpleAdapter` metadata helpers directly.
"""

from __future__ import annotations

from collections.abc import Callable, MutableSequence
from typing import TypeVar

from thenvoi.core.simple_adapter import (
    append_metadata_updates,
    build_metadata_updates,
    prepend_metadata_updates_to_message,
)

T = TypeVar("T")


def build_system_update_messages(
    *,
    participants_msg: str | None,
    contacts_msg: str | None,
) -> list[str]:
    """Build canonical metadata updates via the shared core implementation."""
    return build_metadata_updates(
        participants_msg=participants_msg,
        contacts_msg=contacts_msg,
    )


def prepend_system_updates_to_message(
    user_message: str,
    *,
    participants_msg: str | None,
    contacts_msg: str | None,
) -> str:
    """Prepend canonical metadata updates via the shared core implementation."""
    return prepend_metadata_updates_to_message(
        user_message,
        participants_msg=participants_msg,
        contacts_msg=contacts_msg,
    )


def append_system_updates(
    target: MutableSequence[T],
    *,
    participants_msg: str | None,
    contacts_msg: str | None,
    make_entry: Callable[[str], T],
) -> int:
    """Append canonical metadata updates via the shared core implementation."""
    return append_metadata_updates(
        target,
        participants_msg=participants_msg,
        contacts_msg=contacts_msg,
        make_entry=make_entry,
    )


__all__ = [
    "append_system_updates",
    "build_system_update_messages",
    "prepend_system_updates_to_message",
]
