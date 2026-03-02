"""Approval-related contracts and helper utilities for Codex adapter."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

ApprovalMode = Literal["auto_accept", "auto_decline", "manual"]
ApprovalDecision = Literal["accept", "decline"]


@dataclass
class PendingApproval:
    """In-flight manual approval request state."""

    request_id: int | str
    method: str
    summary: str
    created_at: datetime
    future: asyncio.Future[str]


def approval_summary(method: str, params: dict[str, Any]) -> str:
    """Build a concise human-readable approval request summary."""
    if method == "item/commandExecution/requestApproval":
        command = params.get("command")
        if isinstance(command, str) and command:
            return f"command: {command}"
        return "command execution"
    if method == "item/fileChange/requestApproval":
        reason = params.get("reason")
        if isinstance(reason, str) and reason:
            return f"file changes: {reason}"
        return "file changes"
    return method


def approval_token(request_id: int | str, params: dict[str, Any]) -> str:
    """Return stable token used in manual approval commands."""
    for key in ("approvalId", "approval_id", "itemId", "callId"):
        value = params.get(key)
        if isinstance(value, str) and value:
            return value
    return f"req-{request_id}"


__all__ = [
    "ApprovalDecision",
    "ApprovalMode",
    "PendingApproval",
    "approval_summary",
    "approval_token",
]

