"""Tests for shared config defaults."""

from __future__ import annotations

from thenvoi.config.defaults import DEFAULT_REST_URL, DEFAULT_WS_URL


def test_default_platform_urls_use_expected_schemes() -> None:
    assert DEFAULT_REST_URL.startswith("https://")
    assert DEFAULT_WS_URL.startswith("wss://")


def test_default_platform_urls_target_thenvoi_domain() -> None:
    assert "app.thenvoi.com" in DEFAULT_REST_URL
    assert "app.thenvoi.com" in DEFAULT_WS_URL
