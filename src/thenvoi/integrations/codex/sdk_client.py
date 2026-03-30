"""Async JSON-RPC client for Codex app-server backed by the official SDK.

This module wraps the official ``codex-app-server`` Python SDK so that it
exposes the same ``_CodexClientProtocol`` interface used by the adapter's
event loop.  Server-initiated requests (tool calls, approvals) are bridged
from the SDK's synchronous callback model to the adapter's async world via
``asyncio.run_coroutine_threadsafe`` + ``concurrent.futures.Future``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from .rpc_base import CodexJsonRpcError, RpcEvent

logger = logging.getLogger(__name__)

# Type alias matching the adapter's async request handler signature.
# (RpcEvent) -> dict[str, Any]   (the JSON-RPC result to send back)
AsyncRequestHandler = Callable[..., Awaitable[dict[str, Any] | None]]

# Default timeout for blocking the SDK thread while waiting for async
# server-request resolution (seconds).
_SERVER_REQUEST_TIMEOUT_S = 300.0


class CodexSdkClient:
    """Wraps the official ``codex-app-server`` ``AppServerClient``.

    Implements the same interface as ``BaseJsonRpcClient`` so the adapter
    can use it interchangeably with ``CodexStdioClient`` /
    ``CodexWebSocketClient``.
    """

    def __init__(
        self,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        codex_bin: str | None = None,
        client_name: str = "thenvoi_codex_adapter",
        client_title: str = "Thenvoi Codex Adapter",
        client_version: str = "0.1.0",
        experimental_api: bool = True,
    ) -> None:
        try:
            from codex_app_server import AppServerClient, AppServerConfig  # type: ignore[missing-import]
        except ImportError as exc:
            raise ImportError(
                "codex-app-server is required for transport='sdk'. "
                "Install with: pip install thenvoi-sdk[codex]"
            ) from exc

        self._sdk_config = AppServerConfig(
            codex_bin=codex_bin,
            cwd=cwd,
            env=env,
            client_name=client_name,
            client_title=client_title,
            client_version=client_version,
            experimental_api=experimental_api,
        )
        self._sync_client: AppServerClient | None = None
        self._AppServerClient = AppServerClient

        # Async event loop reference (set on connect).
        self._loop: asyncio.AbstractEventLoop | None = None
        self._connected = False

        # Server-request bridge state (keyed by synthetic request id).
        self._next_synthetic_id = 0
        self._pending_server_responses: dict[
            int, concurrent.futures.Future[dict[str, Any]]
        ] = {}
        self._pending_lock = threading.Lock()

        # Guard: when True, server requests during request() auto-accept.
        self._in_request = False

        # Adapter registers its handler via set_request_handler().
        self._request_handler: AsyncRequestHandler | None = None

    # ------------------------------------------------------------------
    # Public: register the adapter's server-request handler
    # ------------------------------------------------------------------

    def set_request_handler(self, handler: AsyncRequestHandler) -> None:
        """Register the async callback that processes server requests.

        The adapter calls this once after building the client so that
        tool calls and approval requests are routed to the adapter.
        """
        self._request_handler = handler

    # ------------------------------------------------------------------
    # _CodexClientProtocol implementation
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        if self._connected:
            return
        self._loop = asyncio.get_running_loop()
        self._sync_client = self._AppServerClient(
            config=self._sdk_config,
            approval_handler=self._server_request_bridge,
        )
        await asyncio.to_thread(self._sync_client.start)
        self._connected = True

    async def initialize(
        self,
        *,
        client_name: str,
        client_title: str,
        client_version: str,
        experimental_api: bool = False,
        opt_out_notification_methods: list[str] | None = None,
    ) -> dict[str, Any]:
        if self._sync_client is None:
            raise RuntimeError("CodexSdkClient not connected")

        result = await asyncio.to_thread(self._sync_client.initialize)
        # Convert typed InitializeResponse to dict for compatibility.
        if hasattr(result, "model_dump"):
            return result.model_dump(by_alias=True, exclude_none=True)  # type: ignore[union-attr]
        return {}

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        retry_on_overload: bool = True,
    ) -> dict[str, Any]:
        if self._sync_client is None:
            raise RuntimeError("CodexSdkClient not connected")

        def _do() -> dict[str, Any]:
            assert self._sync_client is not None
            self._in_request = True
            try:
                if retry_on_overload:
                    from codex_app_server import retry_on_overload as _retry  # type: ignore[missing-import]

                    return _retry(  # type: ignore[return-value]
                        lambda: self._sync_client._request_raw(method, params),  # noqa: SLF001
                    )
                return self._sync_client._request_raw(method, params)  # noqa: SLF001
            except Exception as exc:
                raise _convert_sdk_error(exc) from exc
            finally:
                self._in_request = False

        return await asyncio.to_thread(_do)

    async def recv_event(self, timeout_s: float | None = None) -> RpcEvent:
        if self._sync_client is None:
            raise RuntimeError("CodexSdkClient not connected")

        coro = asyncio.to_thread(self._sync_client.next_notification)
        if timeout_s is not None:
            notification = await asyncio.wait_for(coro, timeout=timeout_s)
        else:
            notification = await coro
        return self._notification_to_rpc_event(notification)

    async def respond(self, request_id: int | str, result: dict[str, Any]) -> None:
        with self._pending_lock:
            future = self._pending_server_responses.pop(int(request_id), None)
        if future is not None and not future.done():
            future.set_result(result)

    async def respond_error(
        self,
        request_id: int | str,
        *,
        code: int,
        message: str,
        data: Any | None = None,
    ) -> None:
        error_result: dict[str, Any] = {
            "error": {"code": code, "message": message, "data": data},
        }
        with self._pending_lock:
            future = self._pending_server_responses.pop(int(request_id), None)
        if future is not None and not future.done():
            future.set_result(error_result)

    async def close(self) -> None:
        if not self._connected:
            return
        self._connected = False
        # Fail any pending server-request futures so threads unblock.
        with self._pending_lock:
            for future in self._pending_server_responses.values():
                if not future.done():
                    future.set_result({})
            self._pending_server_responses.clear()
        if self._sync_client is not None:
            try:
                await asyncio.to_thread(self._sync_client.close)
            except Exception:
                logger.debug("Exception during SDK client close", exc_info=True)
            self._sync_client = None

    # ------------------------------------------------------------------
    # Server-request bridge (sync callback → async handler)
    # ------------------------------------------------------------------

    def _server_request_bridge(
        self, method: str, params: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Called by the SDK on the thread-pool thread for server requests.

        If we are inside a ``request()`` call (``_in_request`` is True),
        auto-handle to avoid deadlock — the adapter hasn't entered its
        event loop yet so nobody would process the request.

        Otherwise, schedule the adapter's async handler on the event loop,
        block this thread until a result is ready, and return it to the SDK.
        """
        if self._in_request:
            return self._auto_handle_server_request(method, params)

        if self._loop is None or self._request_handler is None:
            return self._auto_handle_server_request(method, params)

        # Allocate a synthetic request id and a future for the response.
        req_id = self._allocate_synthetic_id()
        response_future: concurrent.futures.Future[dict[str, Any]] = (
            concurrent.futures.Future()
        )
        with self._pending_lock:
            self._pending_server_responses[req_id] = response_future

        # Build an RpcEvent that looks like the custom client would emit.
        rpc_event = RpcEvent(
            kind="request",
            method=method,
            params=params or {},
            id=req_id,
            raw={"method": method, "params": params or {}, "id": req_id},
        )

        # Schedule the adapter's handler on the event loop.
        handler = self._request_handler

        async def _run_handler() -> None:
            try:
                await handler(rpc_event)
            except Exception:
                logger.exception("SDK bridge: async request handler failed")
                # Unblock the waiting thread with an empty result.
                with self._pending_lock:
                    fut = self._pending_server_responses.pop(req_id, None)
                if fut is not None and not fut.done():
                    fut.set_result({})

        asyncio.run_coroutine_threadsafe(_run_handler(), self._loop)

        # Block the SDK thread until the adapter calls respond().
        try:
            return response_future.result(timeout=_SERVER_REQUEST_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            logger.warning(
                "SDK bridge: timed out waiting for server-request response "
                "(method=%s, id=%s)",
                method,
                req_id,
            )
            with self._pending_lock:
                self._pending_server_responses.pop(req_id, None)
            return {}

    @staticmethod
    def _auto_handle_server_request(
        method: str, params: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Fallback handler used during ``request()`` calls or when no async handler is set.

        Defaults to declining approval requests so that commands are never
        silently auto-approved when the adapter is not ready to handle them
        (e.g. during ``request()`` calls or before the handler is registered).
        """
        if method in {
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
        }:
            logger.warning(
                "SDK bridge: auto-declining approval request (method=%s) "
                "because no async handler is available",
                method,
            )
            return {"decision": "decline"}
        return {}

    def _allocate_synthetic_id(self) -> int:
        with self._pending_lock:
            self._next_synthetic_id += 1
            return self._next_synthetic_id

    # ------------------------------------------------------------------
    # Notification conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _notification_to_rpc_event(notification: Any) -> RpcEvent:
        """Convert an official SDK ``Notification`` to an ``RpcEvent``.

        The SDK returns typed Pydantic-model payloads.  We convert them
        back to a plain dict so the adapter's event loop can process them
        identically to events from the custom transport clients.
        """
        method: str = notification.method
        payload = notification.payload

        if hasattr(payload, "model_dump"):
            params: dict[str, Any] = payload.model_dump(by_alias=True)
        elif hasattr(payload, "params"):
            # UnknownNotification — raw dict
            params = dict(payload.params) if payload.params else {}
        else:
            params = {}

        return RpcEvent(
            kind="notification",
            method=method,
            params=params,
            id=None,
            raw={"method": method, "params": params},
        )


# ---------------------------------------------------------------------------
# Error conversion
# ---------------------------------------------------------------------------


def _convert_sdk_error(exc: Exception) -> Exception:
    """Convert SDK exception types to the adapter's ``CodexJsonRpcError``."""
    try:
        from codex_app_server import JsonRpcError as SdkJsonRpcError  # type: ignore[missing-import]
    except ImportError:
        return exc

    if isinstance(exc, SdkJsonRpcError):
        return CodexJsonRpcError(
            code=exc.code,
            message=exc.message,
            data=getattr(exc, "data", None),
        )
    return exc
