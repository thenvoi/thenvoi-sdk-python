"""Tests for Thenvoi exception hierarchy."""

from __future__ import annotations

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
        with __import__("pytest").raises(ThenvoiError):
            raise ThenvoiConfigError("bad config")

    def test_message_preserved(self) -> None:
        err = ThenvoiToolError("send_message failed: 403")
        assert str(err) == "send_message failed: 403"

    def test_config_error_not_tool_error(self) -> None:
        assert not issubclass(ThenvoiConfigError, ThenvoiToolError)
        assert not issubclass(ThenvoiToolError, ThenvoiConfigError)
