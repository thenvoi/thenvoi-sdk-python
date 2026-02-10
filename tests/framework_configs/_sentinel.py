"""Shared sentinel value for required config fields with no real default."""

from __future__ import annotations


class _MissingSentinel:
    """Sentinel indicating a required field was not provided."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<MISSING>"


MISSING = _MissingSentinel()
