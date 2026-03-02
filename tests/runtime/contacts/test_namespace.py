"""Namespace contract tests for ``thenvoi.runtime.contacts``."""

from __future__ import annotations

import pytest

import thenvoi.runtime.contacts as contacts_module
from thenvoi.runtime.contacts.contact_handler import (
    HUB_ROOM_SYSTEM_PROMPT,
    ContactEventHandler,
    MAX_DEDUP_CACHE_SIZE,
)
from thenvoi.runtime.contacts.contact_tools import ContactTools
from thenvoi.runtime.contacts.service import ContactService
from thenvoi.runtime.contacts.sink import (
    CallbackContactEventSink,
    ContactEventSink,
    ContactRuntimePort,
    RuntimeContactEventSink,
)


def test_contacts_namespace_exports_expected_symbols() -> None:
    assert contacts_module.ContactEventHandler is ContactEventHandler
    assert contacts_module.ContactTools is ContactTools
    assert contacts_module.ContactService is ContactService
    assert contacts_module.ContactEventSink is ContactEventSink
    assert contacts_module.ContactRuntimePort is ContactRuntimePort
    assert contacts_module.RuntimeContactEventSink is RuntimeContactEventSink
    assert contacts_module.CallbackContactEventSink is CallbackContactEventSink
    assert contacts_module.HUB_ROOM_SYSTEM_PROMPT == HUB_ROOM_SYSTEM_PROMPT
    assert contacts_module.MAX_DEDUP_CACHE_SIZE == MAX_DEDUP_CACHE_SIZE


def test_contacts_namespace_rejects_unknown_attributes() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(contacts_module, "DoesNotExist")


def test_contacts_namespace_dir_contains_public_symbols() -> None:
    public = set(contacts_module.__all__)
    available = set(dir(contacts_module))

    assert public.issubset(available)
