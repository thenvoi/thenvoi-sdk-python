"""Tests for thenvoi.converters lazy namespace exports."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest


def test_converters_namespace_resolves_langchain_exports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converters_module = importlib.import_module("thenvoi.converters")

    langchain_module = ModuleType("thenvoi.converters.langchain")

    class _FakeLangChainHistoryConverter:
        pass

    class _FakeLangChainMessages:
        pass

    langchain_module.LangChainHistoryConverter = _FakeLangChainHistoryConverter
    langchain_module.LangChainMessages = _FakeLangChainMessages
    monkeypatch.setitem(sys.modules, "thenvoi.converters.langchain", langchain_module)

    assert converters_module.LangChainHistoryConverter is _FakeLangChainHistoryConverter
    assert converters_module.LangChainMessages is _FakeLangChainMessages


def test_converters_namespace_resolves_singleton_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converters_module = importlib.import_module("thenvoi.converters")

    codex_module = ModuleType("thenvoi.converters.codex")

    class _FakeCodexHistoryConverter:
        pass

    codex_module.CodexHistoryConverter = _FakeCodexHistoryConverter
    monkeypatch.setitem(sys.modules, "thenvoi.converters.codex", codex_module)

    assert converters_module.CodexHistoryConverter is _FakeCodexHistoryConverter


def test_converters_namespace_rejects_unknown_attribute() -> None:
    converters_module = importlib.import_module("thenvoi.converters")

    with pytest.raises(AttributeError):
        converters_module.__getattr__("DoesNotExist")
