"""Thenvoi SDK exception hierarchy."""

from __future__ import annotations


class ThenvoiError(Exception):
    """Base for all SDK exceptions."""


class ThenvoiConfigError(ThenvoiError):
    """Configuration or setup problems. Actionable by developer."""


class ThenvoiConnectionError(ThenvoiError):
    """Transport failures (WebSocket, REST). Actionable by ops."""


class ThenvoiToolError(ThenvoiError):
    """Tool execution failures. Actionable by adapter/LLM."""
