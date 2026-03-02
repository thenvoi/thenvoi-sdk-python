"""Tests for shared adapter system-update helpers."""

from __future__ import annotations

from thenvoi.adapters.system_updates import (
    append_system_updates,
    build_system_update_messages,
    prepend_system_updates_to_message,
)


def test_build_system_update_messages_orders_participants_then_contacts() -> None:
    updates = build_system_update_messages(
        participants_msg="Alice joined",
        contacts_msg="Contact approved",
    )

    assert updates == [
        "[System]: Alice joined",
        "[System]: Contact approved",
    ]


def test_prepend_system_updates_to_message_no_updates_is_identity() -> None:
    message = prepend_system_updates_to_message(
        "User message",
        participants_msg=None,
        contacts_msg=None,
    )
    assert message == "User message"


def test_prepend_system_updates_to_message_prepends_updates() -> None:
    message = prepend_system_updates_to_message(
        "User message",
        participants_msg="Alice joined",
        contacts_msg="Contact approved",
    )
    assert message == (
        "[System]: Alice joined\n\n[System]: Contact approved\n\nUser message"
    )


def test_append_system_updates_appends_entries_in_canonical_order() -> None:
    entries: list[dict[str, str]] = []
    count = append_system_updates(
        entries,
        participants_msg="Alice joined",
        contacts_msg="Contact approved",
        make_entry=lambda update: {"role": "user", "content": update},
    )

    assert count == 2
    assert entries == [
        {"role": "user", "content": "[System]: Alice joined"},
        {"role": "user", "content": "[System]: Contact approved"},
    ]
