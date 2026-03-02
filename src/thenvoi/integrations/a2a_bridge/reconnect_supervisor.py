"""Reconnect backoff supervisor for bridge connections."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Protocol

from thenvoi.core.nonfatal import NonFatalErrorRecorder

logger = logging.getLogger(__name__)


class ReconnectConfigLike(Protocol):
    """Structural type for reconnect settings consumed by the supervisor."""

    initial_delay: float
    max_delay: float
    multiplier: float
    jitter: float
    max_retries: int


class ReconnectSupervisor(NonFatalErrorRecorder):
    """Run a connection loop with exponential backoff and jitter."""

    def __init__(self, config: ReconnectConfigLike) -> None:
        self._config = config
        self._init_nonfatal_errors()

    async def run(
        self,
        *,
        connect_once: Callable[[], Awaitable[None]],
        disconnect: Callable[[], Awaitable[None]],
        connected_event: asyncio.Event,
        shutdown_event: asyncio.Event,
    ) -> None:
        delay = self._config.initial_delay
        attempts = 0

        while not shutdown_event.is_set():
            connected_event.clear()
            try:
                await connect_once()
                break
            except Exception:
                if shutdown_event.is_set():
                    break

                if connected_event.is_set():
                    delay = self._config.initial_delay
                    attempts = 0

                attempts += 1
                if (
                    self._config.max_retries > 0
                    and attempts >= self._config.max_retries
                ):
                    logger.error(
                        "Max reconnect attempts (%d) reached, giving up",
                        self._config.max_retries,
                    )
                    break

                logger.warning(
                    "Connection lost, reconnecting in %.1fs",
                    delay,
                    exc_info=True,
                )

                try:
                    await disconnect()
                except Exception as error:
                    self._record_nonfatal_error("disconnect_cleanup", error)

                jitter = random.uniform(0, self._config.jitter)  # noqa: S311
                await asyncio.sleep(delay + jitter)
                delay = min(
                    delay * self._config.multiplier,
                    self._config.max_delay,
                )

        logger.info("Reconnect loop exited")


__all__ = ["ReconnectSupervisor"]
