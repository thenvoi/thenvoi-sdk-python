"""
History converter for Letta adapter.

Letta manages its own conversation history, so this is a passthrough converter.
We disable Thenvoi's context hydration since Letta doesn't need it.
"""

from __future__ import annotations

from typing import Any

from thenvoi.core.protocols import HistoryConverter


class LettaPassthroughConverter(HistoryConverter[list[dict[str, Any]]]):
    """
    Passthrough converter for Letta.

    Letta manages its own conversation history server-side.
    This converter simply returns the raw history unchanged.

    In practice, when using Letta adapter, you should set:
        session_config=SessionConfig(enable_context_hydration=False)

    This means the history will always be empty, and this converter
    is essentially a no-op.
    """

    def convert(self, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return history unchanged (passthrough)."""
        return raw
