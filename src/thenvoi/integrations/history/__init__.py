"""
History reconstruction utilities for converting platform history to framework formats.

The main entry point is `parse_platform_history()` which converts raw platform
history into normalized messages. Then use a framework adapter to convert to
the specific format needed.

Example:
    from thenvoi.integrations.history import parse_platform_history
    from thenvoi.integrations.history.adapters.anthropic import to_anthropic_messages

    # Parse platform history to normalized format
    normalized = parse_platform_history(raw_history)

    # Convert to Anthropic format
    messages = to_anthropic_messages(normalized)
"""

from .normalized import (
    NormalizedMessage,
    NormalizedSystemMessage,
    NormalizedToolExchange,
    NormalizedUserText,
)
from .parser import parse_platform_history

__all__ = [
    "parse_platform_history",
    "NormalizedMessage",
    "NormalizedUserText",
    "NormalizedToolExchange",
    "NormalizedSystemMessage",
]
