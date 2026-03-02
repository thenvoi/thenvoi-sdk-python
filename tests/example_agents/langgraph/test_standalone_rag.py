"""Tests for examples/langgraph/standalone/rag.py graph construction."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import examples.langgraph.standalone.rag as module


def test_create_rag_graph_builds_expected_workflow(monkeypatch) -> None:
    loaded_urls: list[str] = []
    split_input: list[Any] = []
    vectorstore_payload: dict[str, Any] = {}
    model_configs: list[dict[str, Any]] = []
    tool_nodes: list[list[Any]] = []
    compiled_kwargs: dict[str, Any] = {}
    graph_calls: dict[str, list[Any]] = {
        "add_node": [],
        "add_edge": [],
        "add_conditional_edges": [],
    }

    class _FakeLoader:
        def __init__(self, url: str) -> None:
            loaded_urls.append(url)

        def load(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(page_content="doc")]

    class _FakeSplitter:
        @classmethod
        def from_tiktoken_encoder(cls, chunk_size: int, chunk_overlap: int):
            assert chunk_size == 100
            assert chunk_overlap == 50
            return cls()

        def split_documents(self, docs: list[Any]) -> list[Any]:
            split_input.extend(docs)
            return docs

    class _FakeRetriever:
        def invoke(self, query: str) -> list[SimpleNamespace]:
            return [SimpleNamespace(page_content=f"context for {query}")]

    class _FakeVectorStore:
        @classmethod
        def from_documents(cls, documents: list[Any], embedding: Any):
            vectorstore_payload["documents"] = documents
            vectorstore_payload["embedding"] = embedding
            return cls()

        def as_retriever(self) -> _FakeRetriever:
            return _FakeRetriever()

    class _FakeEmbeddings:
        pass

    class _FakeChatOpenAI:
        def __init__(self, model: str, temperature: int) -> None:
            model_configs.append({"model": model, "temperature": temperature})

    class _FakeToolNode:
        def __init__(self, tools: list[Any]) -> None:
            tool_nodes.append(tools)

    class _FakeStateGraph:
        def __init__(self, _state_type: Any) -> None:
            pass

        def add_node(self, name: str, node: Any) -> None:
            graph_calls["add_node"].append(name)

        def add_edge(self, source: Any, destination: Any) -> None:
            graph_calls["add_edge"].append((source, destination))

        def add_conditional_edges(self, source: str, condition: Any, mapping: Any = None) -> None:
            graph_calls["add_conditional_edges"].append((source, mapping))

        def compile(self, **kwargs: Any) -> dict[str, Any]:
            compiled_kwargs.update(kwargs)
            return {"compiled": True}

    class _FakeSaver:
        pass

    monkeypatch.setattr(module, "WebBaseLoader", _FakeLoader)
    monkeypatch.setattr(module, "RecursiveCharacterTextSplitter", _FakeSplitter)
    monkeypatch.setattr(module, "InMemoryVectorStore", _FakeVectorStore)
    monkeypatch.setattr(module, "OpenAIEmbeddings", _FakeEmbeddings)
    monkeypatch.setattr(module, "ChatOpenAI", _FakeChatOpenAI)
    monkeypatch.setattr(module, "ToolNode", _FakeToolNode)
    monkeypatch.setattr(module, "StateGraph", _FakeStateGraph)
    monkeypatch.setattr(module, "InMemorySaver", _FakeSaver)
    monkeypatch.setattr(module, "tool", lambda fn: fn)

    result = module.create_rag_graph()

    assert result == {"compiled": True}
    assert len(loaded_urls) == 3
    assert len(split_input) == 3
    assert vectorstore_payload["documents"] == split_input
    assert isinstance(vectorstore_payload["embedding"], _FakeEmbeddings)
    assert model_configs == [
        {"model": "gpt-4o", "temperature": 0},
        {"model": "gpt-4o", "temperature": 0},
    ]
    assert "generate_query_or_respond" in graph_calls["add_node"]
    assert "retrieve" in graph_calls["add_node"]
    assert "rewrite_question" in graph_calls["add_node"]
    assert "generate_answer" in graph_calls["add_node"]
    assert graph_calls["add_edge"][0] == (module.START, "generate_query_or_respond")
    assert graph_calls["add_edge"][-1] == ("generate_answer", module.END)
    assert graph_calls["add_conditional_edges"][0][0] == "generate_query_or_respond"
    assert graph_calls["add_conditional_edges"][1][0] == "retrieve"
    assert len(tool_nodes) == 1
    assert len(tool_nodes[0]) == 1
    assert isinstance(compiled_kwargs["checkpointer"], _FakeSaver)
