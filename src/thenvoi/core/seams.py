"""Shared seam contract models for module handoffs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

ValueT = TypeVar("ValueT")


@dataclass(frozen=True)
class BoundaryError:
    """Typed error payload for cross-module seam boundaries."""

    code: str
    message: str
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class BoundaryResult(Generic[ValueT]):
    """Typed boundary result wrapper with either value or error."""

    value: ValueT | None = None
    error: BoundaryError | None = None

    @property
    def is_ok(self) -> bool:
        return self.error is None

    @classmethod
    def success(cls, value: ValueT) -> BoundaryResult[ValueT]:
        return cls(value=value, error=None)

    @classmethod
    def failure(
        cls,
        *,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> BoundaryResult[ValueT]:
        return cls(
            value=None,
            error=BoundaryError(code=code, message=message, details=details),
        )

    def unwrap(self, *, operation: str) -> ValueT:
        """Return value or raise RuntimeError with typed boundary context."""
        if self.error is not None:
            raise RuntimeError(
                f"{operation} failed ({self.error.code}): {self.error.message}"
            )
        if self.value is None:
            raise RuntimeError(f"{operation} failed (empty_result)")
        return self.value
