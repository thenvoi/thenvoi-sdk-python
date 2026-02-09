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


class TestAdapterOnStarted:
    """All adapters set agent name and description after on_started."""

    @pytest.mark.asyncio
    async def test_after_on_started_sets_agent_name_and_description(
        self, adapter_config
    ):
        """After on_started(agent_name, agent_description), adapter has them set."""
        if getattr(adapter_config, "skip_on_started_conformance", False):
            pytest.skip(
                f"{adapter_config.display_name} on_started requires live client (tested in framework-specific tests)"
            )
        adapter = adapter_config.adapter_factory()
        await adapter.on_started(
            agent_name="TestBot",
            agent_description="A test bot for conformance.",
        )

        assert adapter.agent_name == "TestBot"
        assert adapter.agent_description == "A test bot for conformance."


class TestAdapterOnMessage:
    """All adapters expose an on_message method with the expected signature."""

    def test_on_message_is_callable(self, adapter_config):
        """Adapter has a callable on_message method."""
        adapter = adapter_config.adapter_factory()
        assert hasattr(adapter, "on_message")
        assert callable(adapter.on_message)

    def test_on_message_is_coroutine_function(self, adapter_config):
        """on_message must be an async method."""
        import inspect

        adapter = adapter_config.adapter_factory()
        assert inspect.iscoroutinefunction(adapter.on_message)


class TestAdapterCleanup:
    """All adapters handle cleanup safely."""

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_room_is_safe(self, adapter_config):
        """Cleaning up a room that was never used should not raise."""
        adapter = adapter_config.adapter_factory()

        # Should not raise
        await adapter.on_cleanup("nonexistent-room")

    @pytest.mark.asyncio
    async def test_cleanup_all_safe_when_supported(self, adapter_config):
        """If adapter has cleanup_all(), calling it should not raise."""
        adapter = adapter_config.adapter_factory()
        cleanup_all = getattr(adapter, "cleanup_all", None)
        if cleanup_all is None or not callable(cleanup_all):
            pytest.skip(f"{adapter_config.display_name} does not support cleanup_all")

        # Should not raise
        await adapter.cleanup_all()
