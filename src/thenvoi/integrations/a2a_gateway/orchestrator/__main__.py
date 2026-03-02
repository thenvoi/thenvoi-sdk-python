"""Entry point for the demo orchestrator server."""

from __future__ import annotations

if __package__ in (None, ""):
    raise RuntimeError(
        "Run this entrypoint as a module:\n"
        "  uv run python -m thenvoi.integrations.a2a_gateway.orchestrator"
    )

from .cli import main


if __name__ == "__main__":
    main()
