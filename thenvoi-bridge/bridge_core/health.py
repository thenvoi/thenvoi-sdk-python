"""Health check endpoint for the bridge."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from aiohttp import web

if TYPE_CHECKING:
    from thenvoi.platform.link import ThenvoiLink

    from .session import SessionStore

logger = logging.getLogger(__name__)


class HealthServer:
    """HTTP health check server.

    Exposes GET /health to report bridge connectivity and runtime status.
    """

    def __init__(
        self,
        link: ThenvoiLink,
        port: int = 8080,
        host: str = "0.0.0.0",
        session_store: SessionStore | None = None,
        handler_count: int = 0,
    ) -> None:
        """Initialize the health server.

        Args:
            link: ThenvoiLink to check connection status.
            port: Port to bind on. Defaults to 8080.
            host: Host to bind on. Defaults to "0.0.0.0".
            session_store: Optional session store for active session count.
            handler_count: Number of registered handlers.
        """
        self._link = link
        self._port = port
        self._host = host
        self._session_store = session_store
        self._handler_count = handler_count
        self._app = web.Application()
        self._app.router.add_get("/health", self._health_handler)
        self._runner: web.AppRunner | None = None

    async def start(self) -> None:
        """Start the health server."""
        if self._runner is not None:
            return
        runner = web.AppRunner(self._app)
        await runner.setup()
        site = web.TCPSite(runner, self._host, self._port)
        await site.start()
        self._runner = runner
        logger.info("Health server listening on port %s", self._port)

    async def stop(self) -> None:
        """Stop the health server."""
        if self._runner:
            try:
                await self._runner.cleanup()
            except Exception:
                logger.warning("Error during health server cleanup", exc_info=True)
            self._runner = None
            logger.info("Health server stopped")

    async def _health_handler(self, request: web.Request) -> web.Response:
        """Handle GET /health requests.

        Note: ``count()`` triggers lazy eviction of expired sessions,
        so the Docker healthcheck (default: every 30 s) doubles as the
        eviction driver.  See ``InMemorySessionStore`` for details.
        """
        connected = self._link.is_connected
        status = "healthy" if connected else "unhealthy"

        body: dict[str, Any] = {
            "status": status,
            "websocket_connected": connected,
            "handlers_registered": self._handler_count,
        }

        if self._handler_count == 0:
            body["warning"] = "no handlers registered"

        if self._session_store is not None:
            body["active_sessions"] = await self._session_store.count()

        return web.json_response(body, status=200 if connected else 503)
