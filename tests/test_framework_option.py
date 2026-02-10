"""Tests for the --framework CLI option and framework registry completeness."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.conftest import pytest_collection_modifyitems
from tests.framework_configs.adapters import ADAPTER_CONFIGS
from tests.framework_configs.converters import CONVERTER_CONFIGS


class TestFrameworkOption:
    """Tests for pytest_collection_modifyitems --framework validation."""

    def test_invalid_framework_raises_usage_error(self):
        """Unknown --framework name raises pytest.UsageError with valid names."""
        config = MagicMock()
        config.getoption.return_value = "nonexistent_framework"

        with pytest.raises(
            pytest.UsageError, match="Unknown --framework='nonexistent_framework'"
        ):
            pytest_collection_modifyitems(config, items=[])

    def test_valid_framework_does_not_raise(self):
        """A known framework name does not raise."""
        config = MagicMock()
        config.getoption.return_value = "anthropic"

        # Should not raise; items list will just be filtered (emptied here)
        pytest_collection_modifyitems(config, items=[])


class TestFrameworkRegistryCompleteness:
    """Guard against accidental config omissions after test consolidation."""

    def test_all_adapter_frameworks_registered(self):
        """All 6 adapter frameworks must be present in ADAPTER_CONFIGS."""
        adapter_ids = {c.framework_id for c in ADAPTER_CONFIGS}
        expected = {
            "anthropic",
            "langgraph",
            "crewai",
            "claude_sdk",
            "pydantic_ai",
            "parlant",
        }
        assert adapter_ids == expected, (
            f"Missing adapter configs: {expected - adapter_ids}, "
            f"unexpected: {adapter_ids - expected}"
        )

    def test_all_converter_frameworks_registered(self):
        """All 6 converter frameworks must be present in CONVERTER_CONFIGS."""
        converter_ids = {c.framework_id for c in CONVERTER_CONFIGS}
        expected = {
            "anthropic",
            "langchain",
            "crewai",
            "claude_sdk",
            "pydantic_ai",
            "parlant",
        }
        assert converter_ids == expected, (
            f"Missing converter configs: {expected - converter_ids}, "
            f"unexpected: {converter_ids - expected}"
        )

    def test_adapter_and_converter_counts(self):
        """Exactly 6 adapters and 6 converters are registered."""
        assert len(ADAPTER_CONFIGS) == 6, (
            f"Expected 6 adapter configs, got {len(ADAPTER_CONFIGS)}"
        )
        assert len(CONVERTER_CONFIGS) == 6, (
            f"Expected 6 converter configs, got {len(CONVERTER_CONFIGS)}"
        )
