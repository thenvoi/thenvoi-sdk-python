"""Tests for shared handler utilities in handlers._base."""

from __future__ import annotations

from handlers._base import Handler, resolve_sender

from .conftest import make_tools


# ---------------------------------------------------------------------------
# TestResolveSender
# ---------------------------------------------------------------------------


class TestResolveSender:
    def test_found_returns_name_and_handle(self) -> None:
        tools = make_tools(
            participants=[
                {"id": "user-1", "name": "Alice", "type": "User", "handle": "alice"},
                {"id": "user-2", "name": "Bob", "type": "User", "handle": "bob"},
            ]
        )
        name, handle = resolve_sender("user-1", tools)
        assert name == "Alice"
        assert handle == "alice"

    def test_found_without_handle(self) -> None:
        tools = make_tools(
            participants=[
                {"id": "user-1", "name": "Alice", "type": "User", "handle": None},
            ]
        )
        name, handle = resolve_sender("user-1", tools)
        assert name == "Alice"
        assert handle is None

    def test_not_found_returns_none(self) -> None:
        tools = make_tools(
            participants=[
                {"id": "user-2", "name": "Bob", "type": "User", "handle": "bob"}
            ]
        )
        name, handle = resolve_sender("user-1", tools)
        assert name is None
        assert handle is None

    def test_empty_participants_returns_none(self) -> None:
        tools = make_tools(participants=[])
        name, handle = resolve_sender("user-1", tools)
        assert name is None
        assert handle is None


# ---------------------------------------------------------------------------
# TestHandlerProtocol
# ---------------------------------------------------------------------------


class TestHandlerProtocol:
    def test_langchain_satisfies_protocol(self) -> None:
        from handlers.chain import LangChainHandler

        assert issubclass(LangChainHandler, Handler)
