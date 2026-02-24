"""Async JSON-RPC client for Codex app-server over WebSocket."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from .stdio_client import CodexJsonRpcError, OverloadRetryPolicy, RpcEvent

logger = logging.getLogger(__name__)

JsonRpcId = int | str


class CodexWebSocketClient:
    """Bidirectional JSON-RPC client with a single WebSocket read loop."""

    def __init__(
        self,
        *,
        ws_url: str = "ws://127.0.0.1:8765",
        retry_policy: OverloadRetryPolicy | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        connect_timeout_s: float = 10.0,
    ) -> None:
        self.ws_url = ws_url
        self.retry_policy = retry_policy or OverloadRetryPolicy()
        self._sleep = sleep
        self._connect_timeout_s = connect_timeout_s

        self._ws: Any = None
        self._reader_task: asyncio.Task[None] | None = None
        self._pending: dict[JsonRpcId, asyncio.Future[dict[str, Any]]] = {}
        self._events: asyncio.Queue[RpcEvent] = asyncio.Queue()
        self._request_id = 0
        self._connected = False

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
            open_timeout=self._connect_timeout_s,
        )
        self._connected = True
        self._reader_task = asyncio.create_task(self._read_ws_loop())

    async def initialize(
        self,
        *,
        client_name: str,
        client_title: str,
        client_version: str,
        experimental_api: bool = False,
        opt_out_notification_methods: list[str] | None = None,
    ) -> dict[str, Any]:
        """Perform initialize request followed by initialized notification."""
        capabilities: dict[str, Any] = {"experimentalApi": experimental_api}
        if opt_out_notification_methods:
            capabilities["optOutNotificationMethods"] = opt_out_notification_methods

        result = await self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": client_name,
                    "title": client_title,
                    "version": client_version,
                },
                "capabilities": capabilities,
            },
            retry_on_overload=False,
        )
        await self.notify("initialized", {})
        return result

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        retry_on_overload: bool = True,
    ) -> dict[str, Any]:
        """Send request and await result, with overload retry when enabled."""
        attempt = 0
        while True:
            attempt += 1
            try:
                return await self._request_once(method, params or {})
            except CodexJsonRpcError as err:
                if not retry_on_overload or not self._is_retryable_overload(err):
                    raise
                if attempt >= self.retry_policy.max_attempts:
                    raise
                delay_s = self.retry_policy.backoff_seconds(attempt)
                logger.warning(
                    "Codex overloaded for method=%s; retrying in %.2fs (attempt %s/%s)",
                    method,
                    delay_s,
                    attempt + 1,
                    self.retry_policy.max_attempts,
                )
                await self._sleep(delay_s)

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send JSON-RPC notification."""
        await self._send_json(
            {
                "method": method,
                "params": params or {},
            }
        )

    async def respond(self, request_id: JsonRpcId, result: dict[str, Any]) -> None:
        """Send JSON-RPC response to a server-initiated request."""
        await self._send_json({"id": request_id, "result": result})

    async def respond_error(
        self,
        request_id: JsonRpcId,
        *,
        code: int,
        message: str,
        data: Any | None = None,
    ) -> None:
        """Send JSON-RPC error response to a server-initiated request."""
        error: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        await self._send_json({"id": request_id, "error": error})

    async def recv_event(self, timeout_s: float | None = None) -> RpcEvent:
        """Receive next notification/request emitted by server."""
        if timeout_s is None:
            return await self._events.get()
        return await asyncio.wait_for(self._events.get(), timeout=timeout_s)

    async def close(self) -> None:
        """Close read loop and websocket connection."""
        if not self._connected:
            return
        self._connected = False

        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self._ws is not None:
            await self._ws.close()
            self._ws = None

        self._fail_pending("Codex websocket client closed")

    async def _request_once(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        request_id = self._next_request_id()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[request_id] = future
        try:
            await self._send_json(
                {
                    "method": method,
                    "id": request_id,
                    "params": params,
                }
            )
            response = await future
        finally:
            self._pending.pop(request_id, None)

        if "error" in response:
            error = response.get("error") or {}
            raise CodexJsonRpcError(
                code=int(error.get("code", -32000)),
                message=str(error.get("message", "Unknown JSON-RPC error")),
                data=error.get("data"),
            )
        return response.get("result") or {}

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("WebSocket not connected")
        await self._ws.send(json.dumps(payload, separators=(",", ":")))

    def _fail_pending(self, reason: str) -> None:
        """Fail all outstanding request futures and enqueue a disconnect event."""
        error = RuntimeError(reason)
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(error)
        self._pending.clear()
        self._events.put_nowait(
            RpcEvent(
                kind="notification",
                method="transport/closed",
                params={"reason": reason},
                id=None,
                raw={"method": "transport/closed", "params": {"reason": reason}},
            )
        )

    async def _read_ws_loop(self) -> None:
        if self._ws is None:
            return

        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    text = raw.decode("utf-8", errors="replace")
                else:
                    text = str(raw)
                await self._dispatch_raw_message(text)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Codex websocket read loop failed")
        finally:
            if self._connected:
                self._fail_pending("Codex websocket disconnected")

    async def _dispatch_raw_message(self, text: str) -> None:
        text = text.strip()
        if not text:
            return

        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Skipping non-JSON websocket payload: %s", text)
            return

        if not isinstance(message, dict):
            logger.warning("Skipping non-object JSON-RPC payload: %s", message)
            return

        msg_id = message.get("id")
        method = message.get("method")

        if method and msg_id is not None:
            await self._events.put(
                RpcEvent(
                    kind="request",
                    method=str(method),
                    params=message.get("params"),
                    id=msg_id,
                    raw=message,
                )
            )
            return

        if method:
            await self._events.put(
                RpcEvent(
                    kind="notification",
                    method=str(method),
                    params=message.get("params"),
                    id=None,
                    raw=message,
                )
            )
            return

        if msg_id is not None:
            future = self._pending.get(msg_id)
            if future is not None and not future.done():
                future.set_result(message)
            return

        logger.debug("Ignoring unknown JSON-RPC payload: %s", message)

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    @staticmethod
    def _is_retryable_overload(err: CodexJsonRpcError) -> bool:
        return err.code == -32001 and "overload" in err.message.lower()
