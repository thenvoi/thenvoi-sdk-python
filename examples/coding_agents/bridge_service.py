#!/usr/bin/env python3
"""Bridge coordinator wrapper with a local health endpoint."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_THENVOI_REST_URL = "https://app.thenvoi.com"
DEFAULT_THENVOI_WS_URL = "wss://app.thenvoi.com/api/v1/socket/websocket"


class _HealthHandler(BaseHTTPRequestHandler):
    """Serve a simple readiness probe for Docker health checks."""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler interface
        if self.path != "/health":
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()
            return

        body = b'{"status":"ok"}\n'
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        logger.debug("health-server %s", format % args)


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} environment variable is required")
    return value


def _parse_health_port() -> int:
    raw = os.environ.get("BRIDGE_HEALTH_PORT", "18080")
    try:
        port = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"BRIDGE_HEALTH_PORT must be an integer; got '{raw}'"
        ) from exc

    if not (1 <= port <= 65535):
        raise ValueError(
            f"BRIDGE_HEALTH_PORT must be between 1 and 65535; got {port}"
        )
    return port


def _start_health_server(host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), _HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Bridge health endpoint listening on %s:%s/health", host, port)
    return server


async def _run_runner() -> None:
    import sys

    sys.path.insert(0, "/app/docker/claude_sdk")
    from runner import main as runner_main

    await runner_main()


def main() -> None:
    _require_env("AGENT_CONFIG")
    _require_env("AGENT_KEY")
    _require_env("ANTHROPIC_API_KEY")

    os.environ.setdefault("THENVOI_REST_URL", DEFAULT_THENVOI_REST_URL)
    os.environ.setdefault("THENVOI_WS_URL", DEFAULT_THENVOI_WS_URL)

    config_path = Path(os.environ["AGENT_CONFIG"])
    if not config_path.is_file():
        raise ValueError(f"AGENT_CONFIG does not exist: {config_path}")

    host = os.environ.get("BRIDGE_HEALTH_HOST", "0.0.0.0")
    port = _parse_health_port()
    server = _start_health_server(host, port)

    try:
        asyncio.run(_run_runner())
    finally:
        logger.info("Stopping bridge health endpoint")
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
