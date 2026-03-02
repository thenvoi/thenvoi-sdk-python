"""Tests for examples/langgraph/standalone/sql_agent.py."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import examples.langgraph.standalone.sql_agent as sql_module


def test_create_sql_agent_builds_graph_and_conditional_flow(monkeypatch) -> None:
    mock_from_uri = MagicMock(return_value="db")
    mock_sql_database = SimpleNamespace(from_uri=mock_from_uri)
    llm_with_tools = SimpleNamespace(invoke=MagicMock(return_value="ai-response"))
    fake_llm = SimpleNamespace(bind_tools=MagicMock(return_value=llm_with_tools))
    mock_chat_openai = MagicMock(return_value=fake_llm)
    fake_toolkit = SimpleNamespace(get_tools=MagicMock(return_value=["tool-a"]))
    mock_toolkit_cls = MagicMock(return_value=fake_toolkit)
    mock_tool_node = MagicMock(return_value="tool-node")
    checkpointer = object()
    mock_checkpointer_cls = MagicMock(return_value=checkpointer)

    class _FakeStateGraph:
        def __init__(self, _state_type) -> None:
            self.nodes: dict[str, object] = {}
            self.conditional_function = None
            self.conditional_targets = None

        def add_node(self, name: str, fn: object) -> None:
            self.nodes[name] = fn

        def add_edge(self, _left: object, _right: object) -> None:
            return

        def add_conditional_edges(
            self,
            _node: str,
            func: object,
            targets: list[object],
        ) -> None:
            self.conditional_function = func
            self.conditional_targets = targets

        def compile(self, *, checkpointer: object) -> dict[str, object]:
            return {
                "nodes": self.nodes,
                "conditional_function": self.conditional_function,
                "conditional_targets": self.conditional_targets,
                "checkpointer": checkpointer,
            }

    monkeypatch.setattr(sql_module, "SQLDatabase", mock_sql_database)
    monkeypatch.setattr(sql_module, "ChatOpenAI", mock_chat_openai)
    monkeypatch.setattr(sql_module, "SQLDatabaseToolkit", mock_toolkit_cls)
    monkeypatch.setattr(sql_module, "ToolNode", mock_tool_node)
    monkeypatch.setattr(sql_module, "InMemorySaver", mock_checkpointer_cls)
    monkeypatch.setattr(sql_module, "StateGraph", _FakeStateGraph)

    graph = sql_module.create_sql_agent("custom.db")

    mock_from_uri.assert_called_once_with("sqlite:///custom.db")
    mock_chat_openai.assert_called_once_with(model="gpt-4o", temperature=0)
    mock_toolkit_cls.assert_called_once_with(db="db", llm=fake_llm)
    fake_llm.bind_tools.assert_called_once_with(["tool-a"])
    mock_tool_node.assert_called_once_with(["tool-a"])
    assert graph["checkpointer"] is checkpointer
    assert "agent" in graph["nodes"]
    assert "tools" in graph["nodes"]
    assert graph["conditional_targets"] == ["tools", sql_module.END]

    should_continue = graph["conditional_function"]
    assert should_continue is not None
    assert should_continue({"messages": [SimpleNamespace(tool_calls=[])]}) == sql_module.END
    assert should_continue({"messages": [SimpleNamespace(tool_calls=[{"id": "1"}])]} ) == "tools"

    agent_node = graph["nodes"]["agent"]
    result = agent_node({"messages": ["hello"]})
    assert result == {"messages": ["ai-response"]}
    llm_with_tools.invoke.assert_called_once_with(["hello"])


def test_download_chinook_db_returns_existing_path(monkeypatch) -> None:
    monkeypatch.setattr(sql_module.os.path, "exists", MagicMock(return_value=True))
    mock_urlretrieve = MagicMock()
    monkeypatch.setattr(sql_module.urllib.request, "urlretrieve", mock_urlretrieve)

    path = sql_module.download_chinook_db()

    assert path == "Chinook.db"
    mock_urlretrieve.assert_not_called()


def test_download_chinook_db_downloads_when_missing(monkeypatch) -> None:
    monkeypatch.setattr(sql_module.os.path, "exists", MagicMock(return_value=False))
    mock_urlretrieve = MagicMock(return_value=None)
    monkeypatch.setattr(sql_module.urllib.request, "urlretrieve", mock_urlretrieve)

    path = sql_module.download_chinook_db()

    assert path == "Chinook.db"
    mock_urlretrieve.assert_called_once()


def test_download_chinook_db_creates_fallback_on_download_error(monkeypatch) -> None:
    monkeypatch.setattr(sql_module.os.path, "exists", MagicMock(return_value=False))
    monkeypatch.setattr(
        sql_module.urllib.request,
        "urlretrieve",
        MagicMock(side_effect=RuntimeError("network error")),
    )
    cursor = SimpleNamespace(execute=MagicMock(), executemany=MagicMock())
    connection = SimpleNamespace(
        cursor=MagicMock(return_value=cursor),
        commit=MagicMock(),
        close=MagicMock(),
    )
    mock_connect = MagicMock(return_value=connection)
    monkeypatch.setattr(sql_module.sqlite3, "connect", mock_connect)

    path = sql_module.download_chinook_db()

    assert path == "Chinook.db"
    mock_connect.assert_called_once_with("Chinook.db")
    cursor.execute.assert_called_once()
    cursor.executemany.assert_called_once()
    connection.commit.assert_called_once()
    connection.close.assert_called_once()
