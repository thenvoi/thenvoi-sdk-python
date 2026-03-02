"""Tests for shared deprecation shim helpers."""

from __future__ import annotations

import math

import pytest

from thenvoi.compat.shims import make_deprecated_getattr


def test_make_deprecated_getattr_warns_and_forwards_symbol() -> None:
    """Forwarded symbol should resolve and emit one deprecation warning."""
    getattr_fn = make_deprecated_getattr(
        "legacy.module",
        {"PI": "math:pi"},
        removal_version="1.2.3",
        guidance="Use math.pi instead.",
    )

    with pytest.warns(DeprecationWarning, match=r"legacy\.module\.PI is deprecated"):
        value = getattr_fn("PI")

    assert value == math.pi


def test_make_deprecated_getattr_raises_attribute_error_for_unknown_symbol() -> None:
    """Unknown symbols should raise the same error shape as normal modules."""
    getattr_fn = make_deprecated_getattr(
        "legacy.module",
        {"PI": "math:pi"},
        removal_version="1.2.3",
        guidance="Use math.pi instead.",
    )

    with pytest.raises(AttributeError, match="module 'legacy\\.module'"):
        getattr_fn("UNKNOWN")
