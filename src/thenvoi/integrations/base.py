"""
Base integration utilities shared across all frameworks.

This module provides common helpers that all framework integrations can use
for standardized behavior.

## Tool Event Pairing (for history reconstruction)

When reconstructing conversation history, integrations must pair tool calls with
their results. The matching strategy should be:

1. **Primary: Match by unique ID** - Use whatever unique identifier the framework
   provides (e.g., run_id, tool_call_id, correlation_id). This handles out-of-order
   results and back-to-back calls to the same tool.

2. **Fallback: LIFO by name** - When no unique ID is available, match the most
   recent pending tool call with the same name (stack-based, not queue).

3. **No match found** - Emit the tool result but do NOT fabricate a tool call.
   This keeps LLM state consistent even with incomplete history.

## Message Type Filtering

When the LLM responds via tools (e.g., thenvoi_send_message), skip assistant text messages
during history reconstruction. The actual LLM output is in the tool calls, not the
text. Including both would:
- Duplicate content in the conversation
- Teach the LLM to respond with text instead of using tools

Only reconstruct:
- Tool calls and results (paired into proper LLM message format)
- User messages (as user/human messages)

Skip:
- Assistant text (redundant with tool calls)
- Internal platform types (thought, error, task, etc.)

See each framework adapter for implementation details specific to that framework's
event format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenvoi.preprocessing.participants import (
    check_and_format_participants as _check_and_format_participants,
)

if TYPE_CHECKING:
    from thenvoi.runtime.execution import ExecutionContext

__all__ = ["check_and_format_participants"]


def check_and_format_participants(ctx: "ExecutionContext") -> str | None:
    """Compatibility wrapper for participant update formatting helper."""
    return _check_and_format_participants(ctx)
