"""Codex integration types."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class CodexSessionState:
    """Session state extracted from platform history for Codex rehydration."""

    thread_id: str | None = None
    room_id: str | None = None
    created_at: datetime | None = None

    def has_thread(self) -> bool:
        """Return True when a persisted codex thread_id is available."""
        return bool(self.thread_id)


# ---------------------------------------------------------------------------
# Structured error types
# ---------------------------------------------------------------------------

# Mapping from Codex error type to (human description, suggested action).
CODEX_ERROR_REMEDIATION: dict[str, tuple[str, str]] = {
    "ContextWindowExceeded": (
        "Context window exceeded — the conversation is too long for the model.",
        "compact_context",
    ),
    "UsageLimitExceeded": (
        "Usage limit exceeded — you have hit your API quota.",
        "wait_or_upgrade",
    ),
    "HttpConnectionFailed": (
        "HTTP connection failed — could not reach the API.",
        "check_connectivity",
    ),
    "SandboxError": (
        "Sandbox error — a sandbox policy violation occurred.",
        "review_sandbox_policy",
    ),
    "Unauthorized": (
        "Unauthorized — authentication failed or expired.",
        "re_authenticate",
    ),
    "BadRequest": (
        "Bad request — the input format is invalid.",
        "check_input_format",
    ),
    "ResponseTooManyFailedAttempts": (
        "Too many failed attempts — the model could not produce a valid response.",
        "retry_different_approach",
    ),
}


def build_structured_error_metadata(
    error_obj: dict[str, Any],
    *,
    thread_id: str | None = None,
    turn_id: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Parse a Codex error dict and return (content, metadata) for a structured error event.

    The ``error_obj`` is typically the ``error`` field from a turn payload or an
    ``error`` notification.  It may contain a nested ``codexErrorInfo`` dict with
    a ``type`` field identifying the error class.
    """
    codex_info = error_obj.get("codexErrorInfo") or {}
    if not isinstance(codex_info, dict):
        codex_info = {}
    error_type = codex_info.get("type") or ""
    error_code = codex_info.get("code") or ""
    http_status = codex_info.get("httpStatus")
    is_retryable = bool(codex_info.get("retryable", False))
    additional = error_obj.get("additionalDetails")

    # Look up remediation
    remediation = CODEX_ERROR_REMEDIATION.get(str(error_type))
    if remediation:
        content, suggested_action = remediation
    else:
        raw_message = error_obj.get("message", "")
        content = (
            str(raw_message)
            if raw_message
            else f"Codex error: {error_type or 'unknown'}"
        )
        suggested_action = ""

    metadata: dict[str, Any] = {
        "codex_error_type": error_type or None,
        "codex_error_code": error_code or None,
        "codex_http_status": http_status,
        "codex_is_retryable": is_retryable,
        "codex_suggested_action": suggested_action or None,
    }
    if thread_id:
        metadata["codex_thread_id"] = thread_id
    if turn_id:
        metadata["codex_turn_id"] = turn_id
    if additional:
        metadata["codex_additional_details"] = additional

    return content, metadata


# ---------------------------------------------------------------------------
# Plan step tracking
# ---------------------------------------------------------------------------


@dataclass
class CodexPlanStep:
    """A single step in a Codex plan."""

    step: str
    status: str = "pending"  # pending | inProgress | completed


def parse_plan_steps(params: dict[str, Any]) -> list[CodexPlanStep]:
    """Extract plan steps from a turn/plan/updated event payload."""
    plan = params.get("plan") or params
    steps_raw = plan.get("steps") or []
    if not isinstance(steps_raw, list):
        return []
    steps: list[CodexPlanStep] = []
    for entry in steps_raw:
        if isinstance(entry, dict):
            text = (
                entry.get("text") or entry.get("step") or entry.get("description") or ""
            )
            status = entry.get("status") or "pending"
            if text:
                steps.append(CodexPlanStep(step=str(text), status=str(status)))
        elif isinstance(entry, str) and entry:
            steps.append(CodexPlanStep(step=entry))
    return steps


# ---------------------------------------------------------------------------
# Token usage tracking
# ---------------------------------------------------------------------------


@dataclass
class CodexTokenUsage:
    """Accumulated token usage for a Codex thread."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    def update(self, params: dict[str, Any]) -> None:
        """Update from a thread/tokenUsage/updated payload."""
        usage = params.get("usage") or params
        if not isinstance(usage, dict):
            return
        self.input_tokens = int(
            usage.get("inputTokens") or usage.get("input_tokens") or 0
        )
        self.output_tokens = int(
            usage.get("outputTokens") or usage.get("output_tokens") or 0
        )
        self.reasoning_tokens = int(
            usage.get("reasoningTokens") or usage.get("reasoning_tokens") or 0
        )
        self.total_tokens = int(
            usage.get("totalTokens")
            or usage.get("total_tokens")
            or (self.input_tokens + self.output_tokens + self.reasoning_tokens)
        )

    def to_metadata(self) -> dict[str, Any]:
        """Return metadata dict for a token usage event."""
        return {
            "codex_event_type": "token_usage",
            "codex_input_tokens": self.input_tokens,
            "codex_output_tokens": self.output_tokens,
            "codex_reasoning_tokens": self.reasoning_tokens,
            "codex_total_tokens": self.total_tokens,
        }

    def format_summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Token usage — input: {self.input_tokens:,}, "
            f"output: {self.output_tokens:,}, "
            f"reasoning: {self.reasoning_tokens:,}, "
            f"total: {self.total_tokens:,}"
        )


# ---------------------------------------------------------------------------
# Approval audit entry
# ---------------------------------------------------------------------------


@dataclass
class ApprovalAuditEntry:
    """Records an approval decision for audit purposes."""

    request_id: str | int
    method: str
    decision: str
    decided_by: str
    timestamp: str
    summary: str = ""
    session_level: bool = False
