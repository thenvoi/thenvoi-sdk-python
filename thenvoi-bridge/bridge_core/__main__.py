"""Compatibility entrypoint for ``python -m bridge_core``."""

from __future__ import annotations

from thenvoi.integrations.a2a_bridge.__main__ import _main

__all__ = ["_main"]

if __name__ == "__main__":
    _main()
