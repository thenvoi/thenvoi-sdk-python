"""Async JSON-RPC client for Codex app-server over WebSocket."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from thenvoi.core.nonfatal import NonFatalErrorRecorder
from thenvoi.integrations.lifecycle import AsyncIntegrationLifecycle

from .rpc_base import BaseJsonRpcClient, OverloadRetryPolicy

logger = logging.getLogger(__name__)

_NONFATAL_WS_ERRORS: tuple[type[Exception], ...]


def _build_nonfatal_ws_errors() -> tuple[type[Exception], ...]:
    """Return concrete websocket/runtime exceptions considered non-fatal."""
    error_types: list[type[Exception]] = [
        OSError,
        RuntimeError,
        ConnectionError,
        TimeoutError,
        ValueError,
    ]
    try:
        from websockets.exceptions import ConnectionClosed, WebSocketException

        error_types.extend([ConnectionClosed, WebSocketException])
    except ImportError:
        pass
    return tuple(dict.fromkeys(error_types))


_NONFATAL_WS_ERRORS = _build_nonfatal_ws_errors()


class CodexWebSocketClient(NonFatalErrorRecorder, BaseJsonRpcClient):
    """Bidirectional JSON-RPC client with a single WebSocket read loop."""

    def __init__(
        self,
        *,
        ws_url: str = "ws://127.0.0.1:8765",
        retry_policy: OverloadRetryPolicy | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        connect_timeout_s: float = 10.0,
    ) -> None:
        super().__init__(retry_policy=retry_policy, sleep=sleep)
        self._transport_label = "websocket payload"
        self.ws_url = ws_url
        self._connect_timeout_s = connect_timeout_s
        self._init_nonfatal_errors()

        self._ws: Any = None
        self._reader_task: asyncio.Task[Any] | None = None
        self._lifecycle = AsyncIntegrationLifecycle(
            owner="Codex websocket client",
            logger=logger,
            on_task_error=self._on_lifecycle_task_error,
            fail_pending_operations=self._fail_pending,
        )

    async def connect(self) -> None:
        """Open WebSocket connection to codex app-server."""
        if self._connected:
            return

        try:
            from websockets.asyncio.client import connect
        except ImportError as exc:
            raise RuntimeError(
                "websockets package is required for CodexWebSocketClient"
            ) from exc

        # Codex app-server WS does not support permessage-deflate.
        self._ws = await connect(
            self.ws_url,
            compression=None,
            max_size=16 * 1024 * 1024,  # Codex can emit large JSON-RPC payloads.
            open_timeout=self._connect_timeout_s,
        )
        self._connected = True
        self._reader_task = self._lifecycle.spawn_task(
            "ws_reader",
            self._read_ws_loop(),
        )

    async def close(self) -> None:
        """Close read loop and websocket connection."""
        if not self._connected:
            return
        self._connected = False

        await self._lifecycle.shutdown(
            fail_pending_reason="Codex websocket client closed",
        )
        self._reader_task = None

        if self._ws is not None:
            try:
                await self._ws.close()
            except _NONFATAL_WS_ERRORS as error:
                self._record_nonfatal_error(
                    "websocket_close",
                    error,
                    log_level=logging.DEBUG,
                    ws_url=self.ws_url,
                )
            self._ws = None

    def _on_lifecycle_task_error(self, task_name: str, error: Exception) -> None:
        self._record_nonfatal_error(
            "reader_task_shutdown",
            error,
            ws_url=self.ws_url,
            task=task_name,
            log_level=logging.DEBUG,
        )

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("WebSocket not connected")
        await self._ws.send(json.dumps(payload, separators=(",", ":")))

    # ------------------------------------------------------------------
    # WebSocket-specific read loop
    # ------------------------------------------------------------------

    async def _read_ws_loop(self) -> None:
        if self._ws is None:
            return

        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    text = raw.decode("utf-8", errors="replace")
                else:
                    text = str(raw)
                await self._dispatch_rpc_message(text)
        except asyncio.CancelledError:
            raise
        except _NONFATAL_WS_ERRORS as error:
            self._record_nonfatal_error(
                "read_ws_loop",
                error,
                ws_url=self.ws_url,
            )
        finally:
            if self._connected:
                self._fail_pending("Codex websocket disconnected")
