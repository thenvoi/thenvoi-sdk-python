"""Tests for Thenvoi exception hierarchy."""

from __future__ import annotations

import pytest

from thenvoi.core.exceptions import (
    ThenvoiConfigError,
    ThenvoiConnectionError,
    ThenvoiError,
    ThenvoiToolError,
)


class TestExceptionHierarchy:
    def test_all_inherit_from_thenvoi_error(self) -> None:
        assert issubclass(ThenvoiConfigError, ThenvoiError)
        assert issubclass(ThenvoiConnectionError, ThenvoiError)
        assert issubclass(ThenvoiToolError, ThenvoiError)

    def test_thenvoi_error_inherits_from_exception(self) -> None:
        assert issubclass(ThenvoiError, Exception)

    def test_can_catch_with_base_class(self) -> None:
        with pytest.raises(ThenvoiError):
            raise ThenvoiConfigError("bad config")

    def test_message_preserved(self) -> None:
        err = ThenvoiToolError("send_message failed: 403")
        assert str(err) == "send_message failed: 403"

    def test_config_error_not_tool_error(self) -> None:
        assert not issubclass(ThenvoiConfigError, ThenvoiToolError)
        assert not issubclass(ThenvoiToolError, ThenvoiConfigError)


class TestConfigErrorWithSuggestion:
    """Tests for ThenvoiConfigError.with_suggestion()."""

    def test_suggests_close_match(self) -> None:
        err = ThenvoiConfigError.with_suggestion(
            "Unknown capability 'memry'.",
            "memry",
            ["memory", "contacts"],
        )
        assert "Did you mean 'memory'?" in str(err)

    def test_suggests_case_insensitive(self) -> None:
        err = ThenvoiConfigError.with_suggestion(
            "Unknown emit value 'EXEUCTION'.",
            "EXEUCTION",
            ["execution", "thoughts", "task_events"],
        )
        assert "Did you mean 'execution'?" in str(err)

    def test_no_suggestion_when_too_far(self) -> None:
        err = ThenvoiConfigError.with_suggestion(
            "Unknown capability 'completely_different'.",
            "completely_different",
            ["memory", "contacts"],
        )
        assert "Did you mean" not in str(err)
        assert "Unknown capability 'completely_different'." in str(err)

    def test_picks_closest_among_candidates(self) -> None:
        err = ThenvoiConfigError.with_suggestion(
            "Unknown param 'enabel_memory'.",
            "enabel_memory",
            ["enable_memory", "enable_contacts", "memory"],
        )
        assert "Did you mean 'enable_memory'?" in str(err)

    def test_max_distance_respected(self) -> None:
        # 'memo' -> 'memory' is distance 2
        err_default = ThenvoiConfigError.with_suggestion(
            "Bad name 'memo'.",
            "memo",
            ["memory"],
        )
        assert "Did you mean 'memory'?" in str(err_default)

        # With max_distance=1, 'memo' -> 'memory' is too far
        err_strict = ThenvoiConfigError.with_suggestion(
            "Bad name 'memo'.",
            "memo",
            ["memory"],
            max_distance=1,
        )
        assert "Did you mean" not in str(err_strict)

    def test_returns_thenvoi_config_error(self) -> None:
        err = ThenvoiConfigError.with_suggestion("msg", "x", ["y"], max_distance=5)
        assert isinstance(err, ThenvoiConfigError)
        assert isinstance(err, ThenvoiError)

    def test_empty_haystack_no_suggestion(self) -> None:
        err = ThenvoiConfigError.with_suggestion("Bad name.", "anything", [])
        assert "Did you mean" not in str(err)
