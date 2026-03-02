"""Compatibility tests for the ``thenvoi.platform`` namespace package."""

from __future__ import annotations

from thenvoi.platform import PlatformEvent, ThenvoiLink
from thenvoi.platform.event import PlatformEvent as PlatformEventFromEvent
from thenvoi.platform.link import ThenvoiLink as ThenvoiLinkFromLink


def test_namespace_exports_lazy_platform_symbols() -> None:
    assert PlatformEvent is PlatformEventFromEvent
    assert ThenvoiLink is ThenvoiLinkFromLink
