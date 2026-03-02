"""Deprecated compatibility shim for contact service.

Canonical import:
- ``from thenvoi.runtime.contacts.service import ContactService``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenvoi.compat.shims import make_registered_module_shim

if TYPE_CHECKING:
    from thenvoi.runtime.contacts.service import ContactService as ContactService

REMOVAL_VERSION, __all__, __getattr__ = make_registered_module_shim(
    __name__,
)
