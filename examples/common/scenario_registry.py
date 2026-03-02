"""Deprecated shim for shared example scenario registry.

Use `thenvoi.example_support.scenarios` as the canonical import path.
"""

from __future__ import annotations

from thenvoi.example_support.scenarios import (
    BASIC_ASSISTANT_PROMPT,
    FrameworkName,
    basic_adapter_kwargs,
    basic_agent_key,
    basic_startup_message,
)

__all__ = [
    "BASIC_ASSISTANT_PROMPT",
    "FrameworkName",
    "basic_adapter_kwargs",
    "basic_agent_key",
    "basic_startup_message",
]
