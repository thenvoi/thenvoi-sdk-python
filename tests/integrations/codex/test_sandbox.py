"""Tests for Codex sandbox configurator helpers."""

from __future__ import annotations

import logging

from thenvoi.integrations.codex.sandbox import CodexSandboxConfigurator


def test_normalize_sandbox_mode_accepts_aliases() -> None:
    assert CodexSandboxConfigurator.normalize_sandbox_mode("readOnly") == "read-only"
    assert (
        CodexSandboxConfigurator.normalize_sandbox_mode("danger_full_access")
        == "danger-full-access"
    )


def test_apply_thread_sandbox_uses_representable_policy_type() -> None:
    config = CodexSandboxConfigurator(logging.getLogger(__name__))
    params: dict[str, object] = {}
    config.apply_thread_sandbox(
        params=params,
        sandbox=None,
        sandbox_policy={"type": "workspaceWrite"},
    )
    assert params["sandbox"] == "workspace-write"


def test_apply_turn_sandbox_supports_external_sandbox_alias() -> None:
    config = CodexSandboxConfigurator(logging.getLogger(__name__))
    params: dict[str, object] = {}
    config.apply_turn_sandbox(
        params=params,
        sandbox="external-sandbox",
        sandbox_policy=None,
    )
    assert params["sandboxPolicy"] == {"type": "externalSandbox"}

