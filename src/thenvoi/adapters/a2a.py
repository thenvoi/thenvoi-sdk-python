"""Deprecated compatibility shim for A2A adapter symbols.

Canonical imports:
- ``from thenvoi.adapters import A2AAdapter``
- ``from thenvoi.integrations.a2a import A2AAdapter``

This shim remains for backwards compatibility and is scheduled for removal in
``v1.0.0``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenvoi.compat.shims import make_registered_module_shim

if TYPE_CHECKING:
    from thenvoi.integrations.a2a.adapter import A2AAdapter as A2AAdapter
    from thenvoi.integrations.a2a.types import A2AAuth as A2AAuth

REMOVAL_VERSION, __all__, __getattr__ = make_registered_module_shim(
    __name__,
)
