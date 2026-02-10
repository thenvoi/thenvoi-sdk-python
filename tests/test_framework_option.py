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
        """No duplicate configs: count matches the number of unique framework IDs."""
        expected_adapters = {c.framework_id for c in ADAPTER_CONFIGS}
        expected_converters = {c.framework_id for c in CONVERTER_CONFIGS}
        assert len(ADAPTER_CONFIGS) == len(expected_adapters), (
            f"Expected {len(expected_adapters)} adapter configs (one per framework), "
            f"got {len(ADAPTER_CONFIGS)} — duplicates?"
        )
        assert len(CONVERTER_CONFIGS) == len(expected_converters), (
            f"Expected {len(expected_converters)} converter configs (one per framework), "
            f"got {len(CONVERTER_CONFIGS)} — duplicates?"
        )
