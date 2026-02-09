"""Parameterized conformance tests for all framework adapters.

These tests verify the shared behavioral contract across all six framework
adapters. Framework-specific behavior (tool routing, stream handling, etc.)
remains in the per-framework test files under tests/adapters/.
"""

from __future__ import annotations

import pytest


class TestAdapterInitialization:
    """All adapters share common initialization patterns."""

    def test_default_initialization(self, adapter_config):
        """Adapter defaults match expected values."""
        adapter = adapter_config.adapter_factory()

        for attr_name, expected in adapter_config.default_values.items():
            actual = getattr(adapter, attr_name)
            assert actual == expected, (
                f"{adapter_config.display_name}.{attr_name}: "
                f"expected {expected!r}, got {actual!r}"
            )

    def test_custom_initialization(self, adapter_config):
        """Adapter accepts and stores custom kwargs."""
        if not adapter_config.custom_kwargs:
            pytest.skip(f"{adapter_config.display_name} has no custom kwargs to test")

        adapter = adapter_config.adapter_factory(**adapter_config.custom_kwargs)

        for attr_name, expected in adapter_config.custom_expected.items():
            actual = getattr(adapter, attr_name)
            assert actual == expected, (
                f"{adapter_config.display_name}.{attr_name}: "
                f"expected {expected!r}, got {actual!r}"
            )

    def test_defaults_to_empty_custom_tools(self, adapter_config):
        """Adapters with custom tools start with an empty list."""
        if not adapter_config.has_custom_tools_attr:
            pytest.skip(
                f"{adapter_config.display_name} does not support custom tools attribute"
            )

        adapter = adapter_config.adapter_factory()
        tools = getattr(adapter, adapter_config.custom_tools_attr)

        assert tools == []

    def test_has_history_converter(self, adapter_config):
        """Adapters have a history_converter attribute."""
        if not adapter_config.has_history_converter:
            pytest.skip(
                f"{adapter_config.display_name} does not expose history_converter"
            )

        adapter = adapter_config.adapter_factory()

        assert adapter.history_converter is not None


class TestAdapterCleanup:
    """All adapters handle cleanup safely."""

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_room_is_safe(self, adapter_config):
        """Cleaning up a room that was never used should not raise."""
        adapter = adapter_config.adapter_factory()

        # Should not raise
        await adapter.on_cleanup("nonexistent-room")
