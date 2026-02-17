"""Parametrized fixtures for framework conformance tests."""

from __future__ import annotations

import pytest

from tests.framework_configs.adapters import ADAPTER_CONFIGS
from tests.framework_configs.converters import CONVERTER_CONFIGS


@pytest.fixture(params=CONVERTER_CONFIGS, ids=lambda c: c.framework_id)
def converter_config(request):
    """Yield each ConverterConfig in turn."""
    return request.param


@pytest.fixture(params=ADAPTER_CONFIGS, ids=lambda c: c.framework_id)
def adapter_config(request):
    """Yield each AdapterConfig in turn."""
    return request.param


@pytest.fixture
def make_converter(converter_config):
    """Factory fixture: create a converter from the current config."""

    def _make(**kwargs):
        return converter_config.converter_factory(**kwargs)

    return _make


@pytest.fixture
def output(converter_config):
    """Output adapter for the current converter config."""
    return converter_config.output_adapter
