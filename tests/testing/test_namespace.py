"""Compatibility tests for the ``thenvoi.testing`` namespace exports."""

from __future__ import annotations

import pytest

import thenvoi.testing as testing_namespace
from thenvoi.testing import FakeAgentTools
from thenvoi.testing.fake_tools import FakeAgentTools as FakeAgentToolsFromModule


def test_namespace_exports_fake_agent_tools() -> None:
    assert FakeAgentTools is FakeAgentToolsFromModule


def test_namespace_dir_includes_exported_symbol() -> None:
    assert "FakeAgentTools" in dir(testing_namespace)


def test_namespace_getattr_rejects_unknown_names() -> None:
    with pytest.raises(AttributeError):
        getattr(testing_namespace, "UnknownSymbol")
