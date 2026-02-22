"""Shared fixtures for E2E scenario tests."""

from __future__ import annotations

import pytest

from tests.e2e.adapters.conftest import ADAPTER_FACTORIES, AdapterFactory


@pytest.fixture(params=list(ADAPTER_FACTORIES.keys()))
def adapter_entry(
    request: pytest.FixtureRequest,
) -> tuple[str, AdapterFactory]:
    """Parametrized fixture yielding (name, factory) for each adapter."""
    name = request.param
    return name, ADAPTER_FACTORIES[name]
