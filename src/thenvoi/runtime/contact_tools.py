"""Deprecated compatibility import stub.

Canonical import:
- ``from thenvoi.runtime.contacts.contact_tools import ContactTools``
"""

from __future__ import annotations

from thenvoi.runtime.compat.legacy_modules import contact_tools as _legacy_contact_tools

REMOVAL_VERSION = _legacy_contact_tools.REMOVAL_VERSION
__all__ = _legacy_contact_tools.__all__
__getattr__ = _legacy_contact_tools.__getattr__
