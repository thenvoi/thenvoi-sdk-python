"""
Graceful shutdown utilities for Thenvoi agents.

Provides signal handling and shutdown coordination for clean termination.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from thenvoi.agent import Agent

logger = logging.getLogger(__name__)


class GracefulShutdown:
    """
    Manages graceful shutdown of a Thenvoi agent via signal handling.

    Catches SIGINT (Ctrl+C) and SIGTERM signals and triggers a graceful
    shutdown of the agent, allowing current message processing to complete
    within a configurable timeout.

    Note:
        Signal handlers using asyncio's add_signal_handler() only work on
        Unix systems (Linux, macOS). On Windows, use KeyboardInterrupt
        handling instead:

            try:
                await agent.run()
            except KeyboardInterrupt:
                await agent.stop(timeout=30.0)

    Example (recommended pattern):
        agent = Agent.create(...)

        async def main():
            shutdown = GracefulShutdown(agent, timeout=30.0)
            shutdown.register_signals()

            try:
                await agent.run()
            except asyncio.CancelledError:
                pass  # Normal shutdown

        asyncio.run(main())

    Example (context manager pattern):
        async def main():
            agent = Agent.create(...)

            async with GracefulShutdown(agent, timeout=30.0):
                await agent.run()

    Example (manual signal handling):
        agent = Agent.create(...)

        async def main():
            shutdown = GracefulShutdown(agent, timeout=30.0)

            # Register custom signal behavior
            def on_signal(signum):
                print(f"Received signal {signum}, shutting down...")

            shutdown.on_signal = on_signal
            shutdown.register_signals()

            await agent.run()

    Attributes:
        agent: The Agent instance to shut down.
        timeout: Seconds to wait for graceful shutdown.
        on_signal: Optional callback invoked when signal is received.
    """

    def __init__(
        self,
        agent: "Agent",
        timeout: float = 30.0,
        on_signal: Callable[[int], None] | None = None,
    ):
        """
        Initialize graceful shutdown handler.

        Args:
            agent: The Agent instance to manage.
            timeout: Seconds to wait for processing to complete before
                     forcing shutdown. Default is 30 seconds.
            on_signal: Optional callback invoked when a shutdown signal
                       is received. Receives the signal number.
        """
        self.agent = agent
        self.timeout = timeout
        self.on_signal = on_signal
        self._shutdown_event: asyncio.Event | None = None
        self._original_handlers: dict[int, Any] = {}
        self._registered = False
        self._shutdown_task: asyncio.Task[None] | None = None

    def register_signals(self) -> None:
        """
        Register signal handlers for SIGINT and SIGTERM.

        On Unix systems, also handles SIGHUP.

        Call this before starting the agent. Signals will trigger
        a graceful shutdown of the agent.
        """
        if self._registered:
            return

        loop = asyncio.get_running_loop()
        self._shutdown_event = asyncio.Event()

        # Register for SIGINT (Ctrl+C) and SIGTERM
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._original_handlers[sig] = signal.getsignal(sig)
            loop.add_signal_handler(sig, self._handle_signal, sig)

        # On Unix, also handle SIGHUP
        if hasattr(signal, "SIGHUP"):
            sig = signal.SIGHUP
            self._original_handlers[sig] = signal.getsignal(sig)
            loop.add_signal_handler(sig, self._handle_signal, sig)

        self._registered = True
        logger.debug("Graceful shutdown signal handlers registered")

    def unregister_signals(self) -> None:
        """
        Remove signal handlers and restore original handlers.

        Called automatically when using the context manager pattern.
        """
        if not self._registered:
            return

        loop = asyncio.get_running_loop()

        for sig in list(self._original_handlers.keys()):
            try:
                loop.remove_signal_handler(sig)
            except (ValueError, NotImplementedError):
                pass  # Signal handler was already removed or not supported

        self._original_handlers.clear()
        self._registered = False
        logger.debug("Graceful shutdown signal handlers unregistered")

    def _handle_signal(self, signum: int) -> None:
        """
        Handle received signal by initiating graceful shutdown.

        Args:
            signum: The signal number received.
        """
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")

        # Call user callback if provided
        if self.on_signal:
            try:
                self.on_signal(signum)
            except Exception as e:
                logger.warning(f"Error in on_signal callback: {e}")

        # Set shutdown event to unblock any waiters
        if self._shutdown_event:
            self._shutdown_event.set()

        # Schedule the shutdown coroutine (store reference to prevent GC)
        self._shutdown_task = asyncio.create_task(self._shutdown())

    async def _shutdown(self) -> None:
        """
        Perform graceful shutdown of the agent.

        Waits for current processing to complete within timeout,
        then stops the agent.
        """
        logger.info(f"Shutting down agent (timeout: {self.timeout}s)...")

        try:
            graceful = await self.agent.stop(timeout=self.timeout)
            if graceful:
                logger.info("Agent shut down gracefully")
            else:
                logger.warning("Agent shut down with processing interrupted")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)

    async def wait_for_shutdown(self) -> None:
        """
        Wait until a shutdown signal is received.

        Useful for custom run loops:
            shutdown = GracefulShutdown(agent)
            shutdown.register_signals()
            await agent.start()
            await shutdown.wait_for_shutdown()
        """
        if self._shutdown_event is None:
            self._shutdown_event = asyncio.Event()
        await self._shutdown_event.wait()

    # --- Async context manager ---

    async def __aenter__(self) -> "GracefulShutdown":
        """Enter async context - register signal handlers."""
        self.register_signals()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context - unregister signal handlers."""
        self.unregister_signals()


async def run_with_graceful_shutdown(
    agent: "Agent",
    timeout: float = 30.0,
    on_signal: Callable[[int], None] | None = None,
) -> None:
    """
    Run an agent with graceful shutdown signal handling.

    Convenience function that sets up signal handlers and runs the agent.
    On SIGINT/SIGTERM, waits for current processing to complete within
    the timeout before stopping.

    Args:
        agent: The Agent instance to run.
        timeout: Seconds to wait for graceful shutdown.
        on_signal: Optional callback when signal is received.

    Example:
        agent = Agent.create(...)
        await run_with_graceful_shutdown(agent, timeout=30.0)
    """
    shutdown = GracefulShutdown(agent, timeout=timeout, on_signal=on_signal)

    async with shutdown:
        try:
            await agent.run(shutdown_timeout=timeout)
        except asyncio.CancelledError:
            # Normal shutdown via signal
            pass
