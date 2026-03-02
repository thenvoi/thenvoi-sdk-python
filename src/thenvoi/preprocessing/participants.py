"""Participant update helpers used by preprocessing and integrations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thenvoi.runtime.execution import ExecutionContext


def check_and_format_participants(ctx: "ExecutionContext") -> str | None:
    """Return participant update text when room membership changed."""
    from thenvoi.runtime.formatters import build_participants_message

    if not ctx.participants_changed():
        return None

    msg = build_participants_message(ctx.participants)
    ctx.mark_participants_sent()
    return msg
