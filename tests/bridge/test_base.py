"""Tests for shared handler utilities in handlers._base."""

from __future__ import annotations

from handlers._base import Handler


# ---------------------------------------------------------------------------
# TestHandlerProtocol
# ---------------------------------------------------------------------------


class TestHandlerProtocol:
    def test_langchain_satisfies_protocol(self) -> None:
        from handlers.chain import LangChainHandler

        assert issubclass(LangChainHandler, Handler)
