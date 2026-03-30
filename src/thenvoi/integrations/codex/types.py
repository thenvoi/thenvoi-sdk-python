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
    """Cumulative token usage for a Codex thread.

    Each ``thread/tokenUsage/updated`` event carries cumulative totals,
    so :meth:`update` performs a full replacement (not additive accumulation).
    Per-turn deltas are computed by subtracting the previous snapshot.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    # Per-turn deltas (computed on each update)
    turn_input_tokens: int = 0
    turn_output_tokens: int = 0
    turn_reasoning_tokens: int = 0
    turn_total_tokens: int = 0

    def update(self, params: dict[str, Any]) -> None:
        """Replace counters from a ``thread/tokenUsage/updated`` payload.

        Codex emits **cumulative** totals per thread — each event supersedes
        the previous one — so a full replacement is correct here.
        Per-turn deltas are computed from the difference.
        """
        usage = params.get("usage") or params
        if not isinstance(usage, dict):
            return

        def _get(key_camel: str, key_snake: str) -> int:
            val = usage.get(key_camel)
            if val is None:
                val = usage.get(key_snake)
            if val is None:
                return 0
            try:
                return int(val)
            except (ValueError, TypeError):
                return 0

        prev_input = self.input_tokens
        prev_output = self.output_tokens
        prev_reasoning = self.reasoning_tokens
        prev_total = self.total_tokens

        self.input_tokens = _get("inputTokens", "input_tokens")
        self.output_tokens = _get("outputTokens", "output_tokens")
        self.reasoning_tokens = _get("reasoningTokens", "reasoning_tokens")
        total = usage.get("totalTokens")
        if total is None:
            total = usage.get("total_tokens")
        if total is not None:
            try:
                self.total_tokens = int(total)
            except (ValueError, TypeError):
                self.total_tokens = (
                    self.input_tokens + self.output_tokens + self.reasoning_tokens
                )
        else:
            self.total_tokens = (
                self.input_tokens + self.output_tokens + self.reasoning_tokens
            )

        # Compute per-turn deltas
        self.turn_input_tokens = max(0, self.input_tokens - prev_input)
        self.turn_output_tokens = max(0, self.output_tokens - prev_output)
        self.turn_reasoning_tokens = max(0, self.reasoning_tokens - prev_reasoning)
        self.turn_total_tokens = max(0, self.total_tokens - prev_total)

    def reset_turn_deltas(self) -> None:
        """Reset per-turn deltas (call at the start of a new turn)."""
        self.turn_input_tokens = 0
        self.turn_output_tokens = 0
        self.turn_reasoning_tokens = 0
        self.turn_total_tokens = 0

    def to_metadata(self) -> dict[str, Any]:
        """Return metadata dict for a token usage event."""
        meta: dict[str, Any] = {
            "codex_event_type": "token_usage",
            "codex_input_tokens": self.input_tokens,
            "codex_output_tokens": self.output_tokens,
            "codex_reasoning_tokens": self.reasoning_tokens,
            "codex_total_tokens": self.total_tokens,
        }
        if self.turn_total_tokens > 0:
            meta["codex_turn_input_tokens"] = self.turn_input_tokens
            meta["codex_turn_output_tokens"] = self.turn_output_tokens
            meta["codex_turn_reasoning_tokens"] = self.turn_reasoning_tokens
            meta["codex_turn_total_tokens"] = self.turn_total_tokens
        return meta

    def format_summary(self) -> str:
        """Human-readable summary."""
        summary = (
            f"Token usage — input: {self.input_tokens:,}, "
            f"output: {self.output_tokens:,}, "
            f"reasoning: {self.reasoning_tokens:,}, "
            f"total: {self.total_tokens:,}"
        )
        if self.turn_total_tokens > 0:
            summary += (
                f" (turn: +{self.turn_input_tokens:,} in, "
                f"+{self.turn_output_tokens:,} out, "
                f"+{self.turn_total_tokens:,} total)"
            )
        return summary


# ---------------------------------------------------------------------------
# Approval audit entry
# ---------------------------------------------------------------------------


@dataclass
class ApprovalAuditEntry:
    """Records an approval decision for audit purposes."""

    request_id: str
    method: str
    decision: str
    decided_by: str
    timestamp: str
    summary: str = ""
    session_level: bool = False
