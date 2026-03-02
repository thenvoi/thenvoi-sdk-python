"""Codex integration helpers public namespace."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = (
    "CodexJsonRpcError",
    "CodexStdioClient",
    "CodexWebSocketClient",
    "CodexSessionState",
    "OverloadRetryPolicy",
    "RpcEvent",
)

_EXPORT_MODULES: dict[str, str] = {
    "CodexJsonRpcError": "thenvoi.integrations.codex.rpc_base",
    "CodexStdioClient": "thenvoi.integrations.codex.stdio_client",
    "CodexWebSocketClient": "thenvoi.integrations.codex.websocket_client",
    "CodexSessionState": "thenvoi.integrations.codex.types",
    "OverloadRetryPolicy": "thenvoi.integrations.codex.rpc_base",
    "RpcEvent": "thenvoi.integrations.codex.rpc_base",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
