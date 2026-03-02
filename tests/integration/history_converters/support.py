"""Shared provisioning and payload helpers for history converter integration tests."""

from __future__ import annotations

import json
import logging

from tests.support.integration.contracts.history import create_test_chat, no_clean

logger = logging.getLogger(__name__)


def _create_tool_call_content(
    tool_name: str,
    args: dict[str, object],
    tool_call_id: str,
) -> str:
    """Build tool_call JSON payload in the canonical adapter format."""
    return json.dumps(
        {
            "name": tool_name,
            "args": args,
            "tool_call_id": tool_call_id,
        }
    )


def _create_tool_result_content(
    tool_name: str,
    output: str,
    tool_call_id: str,
) -> str:
    """Build tool_result JSON payload in the canonical adapter format."""
    return json.dumps(
        {
            "name": tool_name,
            "output": output,
            "tool_call_id": tool_call_id,
        }
    )


__all__ = [
    "_create_tool_call_content",
    "_create_tool_result_content",
    "create_test_chat",
    "logger",
    "no_clean",
]
