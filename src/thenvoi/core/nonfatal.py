"""Shared utilities for recording non-fatal errors."""

from __future__ import annotations

import logging
from typing import Any


class NonFatalErrorRecorder:
    """Mixin that records recoverable errors without interrupting execution."""

    _nonfatal_log_level: int = logging.WARNING

    def _init_nonfatal_errors(self) -> None:
        self._nonfatal_errors: list[dict[str, str]] = []

    @property
    def nonfatal_errors(self) -> list[dict[str, str]]:
        """Return a snapshot of non-fatal errors."""
        return list(self._nonfatal_errors)

    def _record_nonfatal_error(
        self,
        operation: str,
        error: Exception,
        *,
        log_level: int | None = None,
        **context: Any,
    ) -> None:
        details = {"operation": operation, "error": str(error)}
        details.update({k: str(v) for k, v in context.items()})
        self._nonfatal_errors.append(details)
        logger = logging.getLogger(type(self).__module__)
        logger.log(
            self._nonfatal_log_level if log_level is None else log_level,
            "Non-fatal %s error (context=%s): %s",
            operation,
            {k: str(v) for k, v in context.items()},
            error,
            exc_info=True,
        )
