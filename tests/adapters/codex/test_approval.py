from __future__ import annotations

from thenvoi.adapters.codex.approval import approval_summary, approval_token


def test_approval_summary_prefers_explicit_command_reason() -> None:
    assert (
        approval_summary(
            "item/commandExecution/requestApproval",
            {"command": "rm -rf /tmp/work"},
        )
        == "command: rm -rf /tmp/work"
    )
    assert approval_summary("item/fileChange/requestApproval", {"reason": "edit docs"}) == (
        "file changes: edit docs"
    )


def test_approval_summary_falls_back_to_generic_labels() -> None:
    assert approval_summary("item/commandExecution/requestApproval", {}) == "command execution"
    assert approval_summary("item/fileChange/requestApproval", {}) == "file changes"
    assert approval_summary("unknown/method", {}) == "unknown/method"


def test_approval_token_uses_first_supported_key_then_request_id() -> None:
    params = {"approvalId": "a1", "approval_id": "a2", "itemId": "a3"}
    assert approval_token("req-1", params) == "a1"
    assert approval_token(42, {"callId": "call-42"}) == "call-42"
    assert approval_token(99, {}) == "req-99"

