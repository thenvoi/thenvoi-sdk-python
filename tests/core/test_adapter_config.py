"""Tests for shared adapter config construction helper."""

from __future__ import annotations

import pytest

from thenvoi.adapters.anthropic import AnthropicAdapter, AnthropicAdapterConfig
from thenvoi.core.adapter_config import create_adapter_from_config


def test_create_adapter_from_config_dataclass() -> None:
    adapter = create_adapter_from_config(
        AnthropicAdapter,
        AnthropicAdapterConfig(
            model="claude-sonnet-4-5-20250929",
            custom_section="Custom",
            max_tokens=1024,
        ),
    )

    assert isinstance(adapter, AnthropicAdapter)
    assert adapter.model == "claude-sonnet-4-5-20250929"
    assert adapter.custom_section == "Custom"
    assert adapter.max_tokens == 1024


def test_create_adapter_from_config_requires_dataclass() -> None:
    with pytest.raises(TypeError, match="dataclass"):
        create_adapter_from_config(AnthropicAdapter, {"model": "claude"})
