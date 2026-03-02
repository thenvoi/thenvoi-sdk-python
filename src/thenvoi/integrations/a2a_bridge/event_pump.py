"""Shutdown-aware event pump for bridge link consumers."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable

logger = logging.getLogger(__name__)


class ShutdownAwareEventPump:
    """Consume events while racing I/O and handlers against shutdown."""

    def __init__(self, shutdown_event: asyncio.Event) -> None:
        self._shutdown_event = shutdown_event

    async def run(
        self,
        event_stream: AsyncIterator[object],
        *,
        handle_event: Callable[[object], Awaitable[None]],
        next_event: Callable[[AsyncIterator[object]], Awaitable[object]] = anext,
    ) -> None:
        """Consume events until shutdown is requested or stream closes."""
        shutdown_fut = asyncio.ensure_future(self._shutdown_event.wait())
        next_fut: asyncio.Future[object] | None = None
        handle_fut: asyncio.Future[None] | None = None
        try:
            while True:
                next_fut = asyncio.ensure_future(next_event(event_stream))
                done, _ = await asyncio.wait(
                    {shutdown_fut, next_fut},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if shutdown_fut in done:
                    next_fut.cancel()
                    break
                try:
                    event = next_fut.result()
                except StopAsyncIteration:
                    break
                except RuntimeError as error:
                    if isinstance(error.__cause__, StopAsyncIteration) or isinstance(
                        error.__context__, StopAsyncIteration
                    ):
                        break
                    raise
                next_fut = None

                handle_fut = asyncio.ensure_future(handle_event(event))
                done, _ = await asyncio.wait(
                    {shutdown_fut, handle_fut},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if shutdown_fut in done:
                    handle_fut.cancel()
                    result = (await asyncio.gather(handle_fut, return_exceptions=True))[0]
                    if isinstance(result, asyncio.CancelledError):
                        logger.debug("Event handler cancelled during shutdown")
                    elif isinstance(result, Exception):
                        logger.debug(
                            "Event handler raised during shutdown cancellation: %s",
                            result,
                            exc_info=True,
                        )
                    handle_fut = None
                    break

                try:
                    handle_fut.result()
                except Exception:
                    logger.warning(
                        "Unexpected error handling event %s",
                        type(event).__name__,
                        exc_info=True,
                    )
                handle_fut = None
        finally:
            if not shutdown_fut.done():
                shutdown_fut.cancel()
            if next_fut is not None and not next_fut.done():
                next_fut.cancel()
            if handle_fut is not None and not handle_fut.done():
                handle_fut.cancel()


__all__ = ["ShutdownAwareEventPump"]
