"""thenvoi-mcp helpers for the OpenCode adapter."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import socket
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)

_THENVOI_MCP_GIT_SOURCE = "git+https://github.com/thenvoi/thenvoi-mcp.git"


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class OpencodeMcpServerProtocol(Protocol):
    """Interface for thenvoi-mcp server management."""

    @property
    def server_name(self) -> str: ...

    @property
    def url(self) -> str | None: ...

    async def start(self) -> str | None: ...

    async def stop(self) -> None: ...


class ThenvoiMcpServer(OpencodeMcpServerProtocol):
    """Manage an external or subprocess-backed thenvoi-mcp SSE server."""

    def __init__(
        self,
        *,
        server_name: str = "thenvoi",
        server_url: str | None = None,
        command: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        host: str = "127.0.0.1",
        port: int | None = None,
        startup_timeout_s: float = 10.0,
    ) -> None:
        self._server_name = server_name
        self._configured_server_url = server_url.rstrip("/") if server_url else None
        self._server_url = self._configured_server_url
        self._command = tuple(command) if command else None
        self._host = host
        self._port = port
        self._startup_timeout_s = startup_timeout_s

        if env is None:
            self._env = None
        else:
            merged_env = dict(os.environ)
            merged_env.update(env)
            self._env = merged_env

        self._proc: asyncio.subprocess.Process | None = None
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._start_lock = asyncio.Lock()
        self._log_tail: deque[str] = deque(maxlen=20)

    @property
    def server_name(self) -> str:
        return self._server_name

    @property
    def url(self) -> str | None:
        return self._server_url

    async def start(self) -> str | None:
        """Return an existing SSE URL or start a local thenvoi-mcp process."""
        async with self._start_lock:
            if self._server_url is not None and self._proc is None:
                logger.info(
                    "Using existing thenvoi-mcp server %r at %s",
                    self._server_name,
                    self._server_url,
                )
                return self._server_url

            if self._proc is not None and self._server_url is not None:
                return self._server_url

            port = self._port or _find_free_port()
            command = self._command or self._default_command(port)
            target_url = f"http://{self._host}:{port}/sse"

            try:
                self._proc = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=self._env,
                )
            except FileNotFoundError as exc:
                binary = command[0] if command else "uvx"
                raise FileNotFoundError(
                    f"Unable to start thenvoi-mcp: '{binary}' was not found."
                ) from exc

            self._stdout_task = asyncio.create_task(self._read_log_stream("stdout"))
            self._stderr_task = asyncio.create_task(self._read_log_stream("stderr"))

            try:
                await self._wait_until_ready(target_url)
            except Exception:
                await self.stop()
                raise

            self._port = port
            self._server_url = target_url
            logger.info(
                "Started thenvoi-mcp server %r at %s",
                self._server_name,
                target_url,
            )
            return self._server_url

    async def stop(self) -> None:
        """Stop a subprocess-backed thenvoi-mcp server."""
        proc = self._proc
        self._proc = None
        self._server_url = self._configured_server_url

        if proc is None:
            return

        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()

        for task in (self._stdout_task, self._stderr_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        self._stdout_task = None
        self._stderr_task = None
        logger.info("Stopped thenvoi-mcp server %r", self._server_name)

    def _default_command(self, port: int) -> tuple[str, ...]:
        """Build the default `thenvoi-mcp` SSE command."""
        if shutil.which("uvx"):
            return (
                "uvx",
                "--from",
                _THENVOI_MCP_GIT_SOURCE,
                "thenvoi-mcp",
                "--transport",
                "sse",
                "--host",
                self._host,
                "--port",
                str(port),
            )

        if shutil.which("uv"):
            return (
                "uv",
                "tool",
                "run",
                "--from",
                _THENVOI_MCP_GIT_SOURCE,
                "thenvoi-mcp",
                "--transport",
                "sse",
                "--host",
                self._host,
                "--port",
                str(port),
            )

        raise FileNotFoundError(
            "Unable to start thenvoi-mcp: install uv/uvx or provide "
            "OpencodeAdapterConfig.mcp_server_command."
        )

    async def _read_log_stream(self, name: str) -> None:
        stream = None
        if self._proc is not None:
            stream = self._proc.stdout if name == "stdout" else self._proc.stderr
        if stream is None:
            return

        while True:
            line = await stream.readline()
            if not line:
                return
            text = line.decode("utf-8", errors="replace").rstrip()
            if not text:
                continue
            self._log_tail.append(f"{name}: {text}")
            logger.debug("thenvoi-mcp %s: %s", name, text)

    async def _wait_until_ready(self, target_url: str) -> None:
        deadline = asyncio.get_running_loop().time() + self._startup_timeout_s

        async with httpx.AsyncClient(timeout=0.5) as client:
            while asyncio.get_running_loop().time() < deadline:
                if self._proc is not None and self._proc.returncode is not None:
                    raise RuntimeError(self._startup_error_message())
                try:
                    async with client.stream("GET", target_url) as response:
                        response.raise_for_status()
                        return
                except httpx.HTTPError:
                    await asyncio.sleep(0.1)

        raise RuntimeError(self._startup_error_message())

    def _startup_error_message(self) -> str:
        if self._log_tail:
            details = "\n".join(self._log_tail)
            return f"thenvoi-mcp did not become ready.\nRecent logs:\n{details}"
        return "thenvoi-mcp did not become ready."
