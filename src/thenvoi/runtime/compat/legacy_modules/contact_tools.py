"""Compatibility shim implementation for ``thenvoi.runtime.contact_tools``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenvoi.compat.shims import make_registered_module_shim

if TYPE_CHECKING:
    from thenvoi.runtime.contacts.contact_tools import ContactTools as ContactTools

REMOVAL_VERSION, __all__, __getattr__ = make_registered_module_shim(
    "thenvoi.runtime.contact_tools",
)
