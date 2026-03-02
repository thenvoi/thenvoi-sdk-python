"""Deprecated compatibility shim for contact event handling.

Canonical import:
- ``from thenvoi.runtime.contacts.contact_handler import ContactEventHandler``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenvoi.compat.shims import make_registered_module_shim

if TYPE_CHECKING:
    from thenvoi.runtime.contacts.contact_handler import (
        HUB_ROOM_SYSTEM_PROMPT as HUB_ROOM_SYSTEM_PROMPT,
    )
    from thenvoi.runtime.contacts.contact_handler import (
        MAX_DEDUP_CACHE_SIZE as MAX_DEDUP_CACHE_SIZE,
    )
    from thenvoi.runtime.contacts.contact_handler import ContactEventHandler as ContactEventHandler

REMOVAL_VERSION, __all__, __getattr__ = make_registered_module_shim(
    __name__,
)
