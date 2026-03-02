"""Deprecated compatibility import stub.

Canonical import:
- ``from thenvoi.runtime.tooling.custom_tools import ...``
"""

from __future__ import annotations

from thenvoi.runtime.compat.legacy_modules import custom_tools as _legacy_custom_tools

REMOVAL_VERSION = _legacy_custom_tools.REMOVAL_VERSION
__all__ = _legacy_custom_tools.__all__
__getattr__ = _legacy_custom_tools.__getattr__
