"""Codex sandbox policy normalization and request-parameter application."""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Any


class CodexSandboxConfigurator:
    """Normalize and apply Codex sandbox config across thread/turn requests."""

    SANDBOX_MODE_TO_POLICY_TYPE = MappingProxyType(
        {
            "read-only": "readOnly",
            "workspace-write": "workspaceWrite",
            "danger-full-access": "dangerFullAccess",
        }
    )

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def apply_thread_sandbox(
        self,
        *,
        params: dict[str, Any],
        sandbox: str | None,
        sandbox_policy: dict[str, Any] | None,
    ) -> None:
        """Apply sandbox to `thread/start` params (SandboxMode only)."""
        if sandbox_policy is not None:
            policy_type = sandbox_policy.get("type")
            if isinstance(policy_type, str):
                mode = self.normalize_sandbox_mode(policy_type)
                if mode is not None:
                    params["sandbox"] = mode
                    return
            self._logger.warning(
                "sandbox_policy type %s is not representable on thread/start; "
                "it will be applied at turn level instead",
                policy_type,
            )
            return

        if sandbox is None:
            return

        sandbox_mode = self.normalize_sandbox_mode(sandbox)
        if sandbox_mode is not None:
            params["sandbox"] = sandbox_mode
            return

        if self.canonical_sandbox_key(sandbox) == "external-sandbox":
            self._logger.debug(
                "external-sandbox will be applied at turn level, not on thread/start"
            )
            return

        self._logger.warning("Ignoring unsupported Codex sandbox value: %s", sandbox)

    def apply_turn_sandbox(
        self,
        *,
        params: dict[str, Any],
        sandbox: str | None,
        sandbox_policy: dict[str, Any] | None,
    ) -> None:
        """Apply sandbox to `turn/start` params (full SandboxPolicy)."""
        if sandbox_policy is not None:
            params["sandboxPolicy"] = self.normalize_sandbox_policy(sandbox_policy)
            return

        if sandbox is None:
            return

        sandbox_mode = self.normalize_sandbox_mode(sandbox)
        if sandbox_mode is not None:
            policy_type = self.SANDBOX_MODE_TO_POLICY_TYPE.get(sandbox_mode)
            if policy_type:
                params["sandboxPolicy"] = {"type": policy_type}
            return

        if self.canonical_sandbox_key(sandbox) == "external-sandbox":
            params["sandboxPolicy"] = {"type": "externalSandbox"}
            return

        self._logger.warning("Ignoring unsupported Codex sandbox value: %s", sandbox)

    @classmethod
    def normalize_sandbox_mode(cls, sandbox: str) -> str | None:
        """Normalize string aliases to `thread/start` SandboxMode values."""
        key = cls.canonical_sandbox_key(sandbox)
        if key in {"read-only", "workspace-write", "danger-full-access"}:
            return key
        return None

    @classmethod
    def normalize_sandbox_policy(cls, sandbox_policy: dict[str, Any]) -> dict[str, Any]:
        """Normalize sandbox policy type aliases to canonical `turn/start` types."""
        normalized = dict(sandbox_policy)
        policy_type = normalized.get("type")
        if not isinstance(policy_type, str):
            return normalized

        key = cls.canonical_sandbox_key(policy_type)
        normalized_type = cls.SANDBOX_MODE_TO_POLICY_TYPE.get(key)
        if normalized_type is None and key == "external-sandbox":
            normalized_type = "externalSandbox"

        if normalized_type is not None:
            normalized["type"] = normalized_type
        return normalized

    @staticmethod
    def canonical_sandbox_key(value: str) -> str:
        """Canonicalize variant sandbox strings into a stable internal key."""
        compact = value.strip().lower().replace("_", "-").replace(" ", "")
        aliases = {
            "readonly": "read-only",
            "read-only": "read-only",
            "workspacewrite": "workspace-write",
            "workspace-write": "workspace-write",
            "dangerfullaccess": "danger-full-access",
            "danger-full-access": "danger-full-access",
            "externalsandbox": "external-sandbox",
            "external-sandbox": "external-sandbox",
        }
        return aliases.get(compact, compact)

