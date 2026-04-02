"""Public exception types for Thenvoi SDK boundaries."""

from __future__ import annotations


class ThenvoiError(Exception):
    """Base exception for public Thenvoi SDK errors."""


class ThenvoiConfigError(ThenvoiError):
    """Raised when SDK configuration is missing, invalid, or inconsistent."""
