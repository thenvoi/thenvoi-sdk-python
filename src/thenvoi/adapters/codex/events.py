"""Task-event shaping helpers for Codex adapter."""

from __future__ import annotations

from typing import Any


def task_event_id(params: dict[str, Any]) -> str | None:
    """Extract a stable task identifier from Codex event params."""
    for key in ("taskId", "task_id"):
        value = params.get(key)
        if isinstance(value, str) and value:
            return value

    task_value = params.get("task")
    if isinstance(task_value, dict):
        for key in ("taskId", "task_id", "id"):
            nested = task_value.get(key)
            if isinstance(nested, str) and nested:
                return nested
    return None


def task_event_title(params: dict[str, Any]) -> str | None:
    """Extract human-facing task title from Codex event params."""
    for key in ("title", "name"):
        value = params.get(key)
        if isinstance(value, str) and value:
            return value

    task_value = params.get("task")
    if isinstance(task_value, str) and task_value:
        return task_value
    if isinstance(task_value, dict):
        for key in ("title", "name", "description"):
            nested = task_value.get(key)
            if isinstance(nested, str) and nested:
                return nested
    return None


def task_event_summary(params: dict[str, Any]) -> str | None:
    """Extract concise task summary from Codex event params."""
    for key in ("summary", "result", "message", "description"):
        value = params.get(key)
        if isinstance(value, str) and value:
            return value

    task_value = params.get("task")
    if isinstance(task_value, dict):
        for key in ("summary", "result", "message", "description"):
            nested = task_value.get(key)
            if isinstance(nested, str) and nested:
                return nested
    return None


def build_task_event_content(
    *,
    task_id: str | None,
    task: str,
    status: str,
    summary: str | None = None,
) -> str:
    """Render task event payload content in a consistent multiline shape."""
    lines: list[str] = []
    if task_id:
        lines.append(f"UUID: {task_id}")
    lines.append(f"Task: {task}")
    lines.append(f"Status: {status}")
    if summary and summary != task:
        lines.append(f"Summary: {summary}")
    return "\n".join(lines)


__all__ = [
    "build_task_event_content",
    "task_event_id",
    "task_event_summary",
    "task_event_title",
]

