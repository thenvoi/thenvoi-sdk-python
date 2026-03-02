"""Compatibility entrypoint for demo orchestrator server.

Canonical entrypoint lives in
``thenvoi.integrations.a2a_gateway.orchestrator.__main__``.
"""

from __future__ import annotations

from thenvoi.integrations.a2a_gateway.orchestrator.__main__ import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
