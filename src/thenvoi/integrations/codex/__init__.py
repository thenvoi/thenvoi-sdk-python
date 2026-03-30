"""Codex app-server integration helpers."""

from __future__ import annotations

from .rpc_base import (
    CodexJsonRpcError,
    OverloadRetryPolicy,
    RpcEvent,
)
from .stdio_client import CodexStdioClient
from .types import CodexSessionState
from .websocket_client import CodexWebSocketClient

__all__ = [
    "CodexJsonRpcError",
    "CodexSdkClient",
    "CodexStdioClient",
    "CodexWebSocketClient",
    "CodexSessionState",
    "OverloadRetryPolicy",
    "RpcEvent",
]


def __getattr__(name: str) -> object:
    if name == "CodexSdkClient":
        from .sdk_client import CodexSdkClient

        return CodexSdkClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
