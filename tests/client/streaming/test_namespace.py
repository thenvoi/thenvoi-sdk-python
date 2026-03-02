"""Tests for thenvoi.client.streaming lazy namespace exports."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest


def test_streaming_namespace_resolves_and_caches_exports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streaming_module = importlib.import_module("thenvoi.client.streaming")
    streaming_module.__dict__.pop("WebSocketClient", None)

    client_module = ModuleType("thenvoi.client.streaming.client")

    class _FakeWebSocketClient:
        pass

    client_module.WebSocketClient = _FakeWebSocketClient
    monkeypatch.setitem(sys.modules, "thenvoi.client.streaming.client", client_module)

    resolved = streaming_module.WebSocketClient

    assert resolved is _FakeWebSocketClient
    assert streaming_module.__dict__["WebSocketClient"] is _FakeWebSocketClient


def test_streaming_namespace_rejects_unknown_attribute() -> None:
    streaming_module = importlib.import_module("thenvoi.client.streaming")

    with pytest.raises(AttributeError):
        streaming_module.__getattr__("DoesNotExist")


def test_streaming_namespace_dir_includes_public_exports() -> None:
    streaming_module = importlib.import_module("thenvoi.client.streaming")

    namespace_dir = streaming_module.__dir__()

    assert "WebSocketClient" in namespace_dir
    assert "ContactAddedPayload" in namespace_dir
