"""Import-surface tests for canonical a2a_bridge modules and bridge_core shims."""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest


@contextmanager
def _bridge_core_import_path() -> object:
    bridge_root = Path(__file__).resolve().parents[2] / "thenvoi-bridge"
    bridge_path = str(bridge_root)
    if bridge_path not in sys.path:
        sys.path.insert(0, bridge_path)
        try:
            yield None
        finally:
            sys.path.remove(bridge_path)
        return

    yield None


def test_canonical_a2a_bridge_import_surface() -> None:
    from thenvoi.integrations.a2a_bridge.handler import BaseHandler
    from thenvoi.integrations.a2a_bridge.session import (
        InMemorySessionStore,
        SessionData,
        SessionStore,
    )

    assert BaseHandler is not None
    assert InMemorySessionStore is not None
    assert SessionData is not None
    assert SessionStore is not None


def test_bridge_core_session_shim_forwards_to_canonical_symbols() -> None:
    with _bridge_core_import_path():
        module = importlib.import_module("bridge_core.session")

        with pytest.warns(DeprecationWarning, match="bridge_core.session"):
            shim_store = module.InMemorySessionStore
        with pytest.warns(DeprecationWarning, match="bridge_core.session"):
            shim_data = module.SessionData

    from thenvoi.integrations.a2a_bridge.session import (
        InMemorySessionStore,
        SessionData,
    )

    assert shim_store is InMemorySessionStore
    assert shim_data is SessionData


def test_bridge_core_handler_shim_forwards_to_canonical_symbols() -> None:
    with _bridge_core_import_path():
        module = importlib.import_module("bridge_core.handler")

        with pytest.warns(DeprecationWarning, match="bridge_core.handler"):
            shim_handler = module.BaseHandler

    from thenvoi.integrations.a2a_bridge.handler import BaseHandler

    assert shim_handler is BaseHandler
