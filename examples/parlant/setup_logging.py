"""Shared logging configuration for examples."""

import logging


def setup_logging(level=logging.INFO, debug=False):
    """Configure logging to show only thenvoi logs, hiding noisy dependencies.

    Args:
        level: Log level for thenvoi namespace (default INFO)
        debug: If True, enables DEBUG for all thenvoi components including WebSocket
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    actual_level = logging.DEBUG if debug else level
    logging.getLogger("thenvoi").setLevel(actual_level)
    # Also enable logging for the parlant adapter module
    logging.getLogger("thenvoi_parlant_agent").setLevel(actual_level)
