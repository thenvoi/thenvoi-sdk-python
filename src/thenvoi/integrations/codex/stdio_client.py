"""Async JSON-RPC client for Codex app-server over stdio."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import shutil
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

JsonRpcId = int | str


class CodexJsonRpcError(RuntimeError):
    """JSON-RPC error returned by codex app-server."""

    def __init__(
        self,
        *,
        code: int,
        message: str,
        data: Any | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"Codex JSON-RPC error {code}: {message}")


@dataclass(frozen=True)
class OverloadRetryPolicy:
    """Retry policy for retryable server overload errors (-32001)."""

    max_attempts: int = 3
    base_delay_s: float = 0.25
    max_delay_s: float = 2.0
    jitter_s: float = 0.1

    def backoff_seconds(self, attempt_number: int) -> float:
        """Compute exponential backoff delay for retry attempt N (1-based)."""
        base = min(self.base_delay_s * (2 ** (attempt_number - 1)), self.max_delay_s)
        if self.jitter_s <= 0:
            return base
        return max(0.0, base + random.uniform(-self.jitter_s, self.jitter_s))


@dataclass(frozen=True)
class RpcEvent:
    """Non-response JSON-RPC message from server."""

    kind: Literal["notification", "request"]
    method: str
    params: dict[str, Any] | list[Any] | None
    id: JsonRpcId | None
    raw: dict[str, Any]


class CodexStdioClient:
    """Bidirectional JSON-RPC client with a single stdout read loop."""

    def __init__(
        self,
        *,
        command: Sequence[str] | None = None,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        retry_policy: OverloadRetryPolicy | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.command = tuple(command) if command else self._default_codex_command()
        self.cwd = cwd
        if env is None:
            self.env = None
        else:
            merged_env = dict(os.environ)
            merged_env.update(env)
            self.env = merged_env
        self.retry_policy = retry_policy or OverloadRetryPolicy()
        self._sleep = sleep

        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._pending: dict[JsonRpcId, asyncio.Future[dict[str, Any]]] = {}
        self._events: asyncio.Queue[RpcEvent] = asyncio.Queue(maxsize=10000)
        self._request_id = 0
        self._connected = False

    @staticmethod
    def _default_codex_command() -> tuple[str, str, str, str]:
        """
        Resolve the default Codex CLI binary.

        Some installations expose `codex`, others `codex-cli`.
        """
        for binary in ("codex", "codex-cli"):
            resolved = shutil.which(binary)
            if resolved:
                return (resolved, "app-server", "--listen", "stdio://")
        return ("codex", "app-server", "--listen", "stdio://")

    async def connect(self) -> None:
        """Start the codex app-server process and reader tasks."""
        if self._connected:
            return

        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
                env=self.env,
                limit=16 * 1024 * 1024,  # 16 MB — Codex sends large JSON-RPC lines
            )
        except FileNotFoundError as exc:
            binary = self.command[0] if self.command else "codex"
            raise FileNotFoundError(
                f"Codex CLI binary not found: '{binary}'. "
                "Install Codex CLI or configure CodexAdapterConfig.codex_command."
            ) from exc
        self._connected = True
        self._reader_task = asyncio.create_task(self._read_stdout_loop())
        self._stderr_task = asyncio.create_task(self._read_stderr_loop())

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
        """Stop read loops and terminate process."""
        if not self._connected:
            return
        self._connected = False

        for task in (self._reader_task, self._stderr_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if self._proc and self._proc.stdin:
            self._proc.stdin.close()

        if self._proc:
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()

        self._fail_pending("Codex stdio client closed")

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
        if not self._proc or not self._proc.stdin:
            raise RuntimeError("Codex process not connected")
        line = json.dumps(payload, separators=(",", ":")) + "\n"
        self._proc.stdin.write(line.encode("utf-8"))
        await self._proc.stdin.drain()

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

    async def _read_stdout_loop(self) -> None:
        if not self._proc or not self._proc.stdout:
            return
        try:
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    self._fail_pending("Codex process exited unexpectedly")
                    return
                await self._dispatch_raw_line(line.decode("utf-8", errors="replace"))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Codex stdout loop failed")
            self._fail_pending("Codex stdout loop failed")

    async def _read_stderr_loop(self) -> None:
        if not self._proc or not self._proc.stderr:
            return
        try:
            while True:
                line = await self._proc.stderr.readline()
                if not line:
                    return
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    logger.debug("codex stderr: %s", text)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Codex stderr loop failed")

    async def _dispatch_raw_line(self, text: str) -> None:
        text = text.strip()
        if not text:
            return

        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Skipping non-JSON codex output line: %s", text)
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
