"""Shared logging helper for SDK example scripts."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable
from typing import Any, Callable

DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_example_logging(
    level: int = logging.INFO,
    *,
    root_level: int = logging.WARNING,
    extra_level_loggers: Iterable[str] = (),
    warning_loggers: Iterable[str] = (),
    debug_loggers: Iterable[str] = (),
    stream_stdout: bool = False,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str | None = DEFAULT_LOG_DATEFMT,
) -> None:
    """Configure logging for examples with consistent defaults and extensions."""
    basic_config: dict[str, Any] = {"level": root_level, "format": log_format}
    if date_format is not None:
        basic_config["datefmt"] = date_format
    if stream_stdout:
        basic_config["stream"] = sys.stdout

    logging.basicConfig(**basic_config)

    logging.getLogger("thenvoi").setLevel(level)
    for logger_name in extra_level_loggers:
        logging.getLogger(logger_name).setLevel(level)
    for logger_name in warning_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    for logger_name in debug_loggers:
        logging.getLogger(logger_name).setLevel(logging.DEBUG)


_PROFILE_OPTIONS: dict[str, dict[str, Any]] = {
    "a2a_bridge": {},
    "a2a_gateway": {
        "root_level": "level",
        "warning_loggers": ("httpcore", "httpx", "uvicorn"),
    },
    "anthropic": {"extra_level_loggers": ("thenvoi_anthropic_agent",)},
    "claude_sdk": {
        "extra_level_loggers": ("thenvoi_claude_sdk_agent", "session_manager")
    },
    "codex": {
        "root_level": "level",
        "stream_stdout": True,
        "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "date_format": None,
        "warning_loggers": ("websockets", "httpx"),
    },
    "crewai": {"extra_level_loggers": ("thenvoi_crewai_agent",)},
    "langgraph": {},
    "parlant": {"extra_level_loggers": ("thenvoi_parlant_agent",)},
    "pydantic_ai": {},
}


def setup_logging_profile(
    profile: str,
    level: int = logging.INFO,
    *,
    a2a_debug: bool = False,
) -> None:
    """Apply one of the standard example logging profiles."""
    if profile not in _PROFILE_OPTIONS:
        raise ValueError(f"Unknown logging profile: {profile}")

    options = dict(_PROFILE_OPTIONS[profile])
    if options.get("root_level") == "level":
        options["root_level"] = level
    if profile == "a2a_bridge" and a2a_debug:
        options["debug_loggers"] = ("thenvoi.integrations.a2a",)

    setup_example_logging(level, **options)


def build_logging_setup(profile: str) -> Callable[[int], None]:
    """Build a `setup_logging(level)` callable bound to a profile."""

    def _setup(level: int = logging.INFO) -> None:
        setup_logging_profile(profile, level)

    return _setup
