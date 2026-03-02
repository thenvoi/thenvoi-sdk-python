"""Shared integration test skip markers."""

from __future__ import annotations

from typing import Any, Callable

import pytest

MarkerDecorator = Callable[[Any], Any]


def _compose_markers(*decorators: MarkerDecorator) -> MarkerDecorator:
    """Compose multiple pytest decorators into a single reusable marker."""

    def _decorator(target: Any) -> Any:
        wrapped = target
        for decorator in reversed(decorators):
            wrapped = decorator(wrapped)
        return wrapped

    return _decorator


requires_api = _compose_markers(
    pytest.mark.integration,
    pytest.mark.requires_api,
)

requires_multi_agent = _compose_markers(
    pytest.mark.integration,
    pytest.mark.requires_multi_agent,
)

requires_user_api = _compose_markers(
    pytest.mark.integration,
    pytest.mark.requires_user_api,
)

__all__ = [
    "requires_api",
    "requires_multi_agent",
    "requires_user_api",
]
