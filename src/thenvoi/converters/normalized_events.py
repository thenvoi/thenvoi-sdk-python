"""Normalized intermediate history events for converter pipelines."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextHistoryEvent:
    """Canonical text-history event."""

    kind: Literal["text"]
    role: str
    content: str
    sender_name: str
    sender_type: str


@dataclass(frozen=True)
class ToolCallHistoryEvent:
    """Canonical tool-call history event."""

    kind: Literal["tool_call"]
    name: str
    args: dict[str, Any]
    tool_call_id: str


@dataclass(frozen=True)
class ToolResultHistoryEvent:
    """Canonical tool-result history event."""

    kind: Literal["tool_result"]
    name: str
    output: str
    tool_call_id: str
    is_error: bool


NormalizedHistoryEvent = (
    TextHistoryEvent | ToolCallHistoryEvent | ToolResultHistoryEvent
)


def normalize_history_events(raw: list[dict[str, Any]]) -> list[NormalizedHistoryEvent]:
    """Normalize raw platform history into canonical converter events."""
    events: list[NormalizedHistoryEvent] = []

    for hist in raw:
        message_type = str(hist.get("message_type", "text"))
        if message_type == "text":
            events.append(
                TextHistoryEvent(
                    kind="text",
                    role=str(hist.get("role", "user")),
                    content=str(hist.get("content", "")),
                    sender_name=str(hist.get("sender_name", "")),
                    sender_type=str(hist.get("type") or hist.get("sender_type", "User")),
                )
            )
            continue

        content = str(hist.get("content", ""))
        if message_type == "tool_call":
            parsed_call = _parse_tool_call_event(content)
            if parsed_call is not None:
                events.append(parsed_call)
            continue

        if message_type == "tool_result":
            parsed_result = _parse_tool_result_event(content)
            if parsed_result is not None:
                events.append(parsed_result)

    return events


def _parse_tool_call_event(content: str) -> ToolCallHistoryEvent | None:
    try:
        event = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Failed to parse tool_call: %s", repr(content[:100]))
        return None

    if not isinstance(event, dict):
        logger.warning("Failed to parse tool_call: %s", repr(content[:100]))
        return None

    tool_name = event.get("name")
    if not tool_name:
        logger.warning("Skipping tool_call with missing name: %s", repr(content[:100]))
        return None

    tool_call_id = event.get("tool_call_id") or event.get("run_id")
    if not tool_call_id:
        logger.warning(
            "Skipping tool_call with missing tool_call_id: %s",
            repr(content[:100]),
        )
        return None

    args = event.get("args")
    if args is None:
        data = event.get("data", {})
        if isinstance(data, dict):
            args = data.get("input", {})
    if not isinstance(args, dict):
        args = {}

    return ToolCallHistoryEvent(
        kind="tool_call",
        name=str(tool_name),
        args=args,
        tool_call_id=str(tool_call_id),
    )


def _parse_tool_result_event(content: str) -> ToolResultHistoryEvent | None:
    try:
        event = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Failed to parse tool_result: %s", repr(content[:100]))
        return None

    if not isinstance(event, dict):
        logger.warning("Failed to parse tool_result: %s", repr(content[:100]))
        return None

    tool_name = event.get("name")
    if not tool_name:
        logger.warning(
            "Skipping tool_result with missing name: %s",
            repr(content[:100]),
        )
        return None

    output = _extract_tool_result_output(event)
    tool_call_id = _extract_tool_result_call_id(event, output)
    if not tool_call_id:
        logger.warning(
            "Skipping tool_result with missing tool_call_id: %s",
            repr(content[:100]),
        )
        return None

    return ToolResultHistoryEvent(
        kind="tool_result",
        name=str(tool_name),
        output=output,
        tool_call_id=tool_call_id,
        is_error=bool(event.get("is_error", False)),
    )


def _extract_tool_result_output(event: dict[str, Any]) -> str:
    output: Any = event.get("output")
    if output is None:
        data = event.get("data", {})
        if isinstance(data, dict):
            output = data.get("output", "")
    return str(output if output is not None else "")


def _extract_tool_result_call_id(event: dict[str, Any], output: str) -> str:
    tool_call_id = event.get("tool_call_id")
    if tool_call_id:
        return str(tool_call_id)

    match = re.search(r"tool_call_id='([^']+)'", output)
    if match:
        return match.group(1)

    run_id = event.get("run_id")
    if run_id:
        return str(run_id)

    return ""
