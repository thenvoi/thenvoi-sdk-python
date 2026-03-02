"""Tests for thenvoi.adapters lazy namespace exports."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest


def test_adapters_namespace_lazily_resolves_known_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapters_module = importlib.import_module("thenvoi.adapters")

    anthropic_module = ModuleType("thenvoi.adapters.anthropic")

    class _FakeAnthropicAdapter:
        pass

    anthropic_module.AnthropicAdapter = _FakeAnthropicAdapter
    monkeypatch.setitem(sys.modules, "thenvoi.adapters.anthropic", anthropic_module)

    resolved = adapters_module.AnthropicAdapter

    assert resolved is _FakeAnthropicAdapter


def test_adapters_namespace_lazily_resolves_adapter_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapters_module = importlib.import_module("thenvoi.adapters")

    anthropic_module = ModuleType("thenvoi.adapters.anthropic")

    class _FakeAnthropicAdapterConfig:
        pass

    anthropic_module.AnthropicAdapterConfig = _FakeAnthropicAdapterConfig
    monkeypatch.setitem(sys.modules, "thenvoi.adapters.anthropic", anthropic_module)

    resolved = adapters_module.AnthropicAdapterConfig

    assert resolved is _FakeAnthropicAdapterConfig


def test_adapters_namespace_rejects_unknown_attribute() -> None:
    adapters_module = importlib.import_module("thenvoi.adapters")

    with pytest.raises(AttributeError):
        adapters_module.__getattr__("DoesNotExist")
