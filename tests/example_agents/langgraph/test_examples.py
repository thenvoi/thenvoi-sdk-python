"""Tests for examples/langgraph/scenarios/*.py scripts."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _load_module(
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    *,
    preload_modules: dict[str, ModuleType] | None = None,
) -> ModuleType:
    if preload_modules:
        for module_name, module in preload_modules.items():
            monkeypatch.setitem(sys.modules, module_name, module)

    module_name = f"examples.langgraph.scenarios.{filename.removesuffix('.py')}"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.mark.asyncio
async def test_simple_agent_main_bootstraps_and_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch, "01_simple_agent.py")
    mock_chat_openai = MagicMock(return_value="llm")
    mock_checkpointer_cls = MagicMock(return_value="checkpointer")
    adapter_instance = object()
    mock_adapter_cls = MagicMock(return_value=adapter_instance)
    session = SimpleNamespace(agent=SimpleNamespace(run=AsyncMock()))
    mock_bootstrap = MagicMock(return_value=session)

    monkeypatch.setattr(module, "ChatOpenAI", mock_chat_openai)
    monkeypatch.setattr(module, "InMemorySaver", mock_checkpointer_cls)
    monkeypatch.setattr(module, "LangGraphAdapter", mock_adapter_cls)
    monkeypatch.setattr(module, "bootstrap_agent", mock_bootstrap)

    await module.main()

    mock_chat_openai.assert_called_once_with(model="gpt-4o")
    mock_checkpointer_cls.assert_called_once_with()
    mock_adapter_cls.assert_called_once_with(llm="llm", checkpointer="checkpointer")
    mock_bootstrap.assert_called_once_with(agent_key="simple_agent", adapter=adapter_instance)
    session.agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_custom_tools_main_creates_agent_with_additional_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch, "02_custom_tools.py")
    monkeypatch.setenv("THENVOI_WS_URL", "wss://ws.example")
    monkeypatch.setenv("THENVOI_REST_URL", "https://rest.example")

    monkeypatch.setattr(module, "load_dotenv", MagicMock())
    monkeypatch.setattr(module, "load_agent_config", MagicMock(return_value=("agent-1", "api-key")))
    monkeypatch.setattr(module, "ChatOpenAI", MagicMock(return_value="llm"))
    monkeypatch.setattr(module, "InMemorySaver", MagicMock(return_value="checkpointer"))
    adapter_instance = object()
    mock_adapter_cls = MagicMock(return_value=adapter_instance)
    runtime_agent = SimpleNamespace(run=AsyncMock())
    mock_create = MagicMock(return_value=runtime_agent)
    monkeypatch.setattr(module, "LangGraphAdapter", mock_adapter_cls)
    monkeypatch.setattr(module.Agent, "create", mock_create)

    await module.main()

    adapter_kwargs = mock_adapter_cls.call_args.kwargs
    assert adapter_kwargs["llm"] == "llm"
    assert adapter_kwargs["checkpointer"] == "checkpointer"
    assert len(adapter_kwargs["additional_tools"]) == 2
    assert "calculator" in adapter_kwargs["custom_section"].lower()
    runtime_agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_custom_personality_requires_ws_url(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch, "03_custom_personality.py")
    monkeypatch.delenv("THENVOI_WS_URL", raising=False)
    monkeypatch.setenv("THENVOI_REST_URL", "https://rest.example")
    monkeypatch.setattr(module, "load_dotenv", MagicMock())

    with pytest.raises(ValueError, match="THENVOI_WS_URL environment variable is required"):
        await module.main()


@pytest.mark.asyncio
async def test_calculator_as_tool_main_builds_tool_and_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calculator_stub = ModuleType("examples.langgraph.standalone.calculator")
    calculator_stub.create_calculator_graph = lambda: "unused"
    module = _load_module(
        monkeypatch,
        "04_calculator_as_tool.py",
        preload_modules={"examples.langgraph.standalone.calculator": calculator_stub},
    )
    monkeypatch.setenv("THENVOI_WS_URL", "wss://ws.example")
    monkeypatch.setenv("THENVOI_REST_URL", "https://rest.example")

    monkeypatch.setattr(module, "load_dotenv", MagicMock())
    monkeypatch.setattr(module, "load_agent_config", MagicMock(return_value=("calc-id", "calc-key")))
    monkeypatch.setattr(module, "create_calculator_graph", MagicMock(return_value="calc-graph"))
    monkeypatch.setattr(module, "graph_as_tool", MagicMock(return_value="calculator-tool"))
    monkeypatch.setattr(module, "ChatOpenAI", MagicMock(return_value="llm"))
    monkeypatch.setattr(module, "InMemorySaver", MagicMock(return_value="checkpointer"))
    adapter_instance = object()
    mock_adapter_cls = MagicMock(return_value=adapter_instance)
    runtime_agent = SimpleNamespace(run=AsyncMock())
    mock_create = MagicMock(return_value=runtime_agent)
    monkeypatch.setattr(module, "LangGraphAdapter", mock_adapter_cls)
    monkeypatch.setattr(module.Agent, "create", mock_create)

    await module.main()

    module.create_calculator_graph.assert_called_once()
    module.graph_as_tool.assert_called_once()
    assert mock_adapter_cls.call_args.kwargs["additional_tools"] == ["calculator-tool"]
    runtime_agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_rag_as_tool_main_builds_tool_and_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rag_stub = ModuleType("examples.langgraph.standalone.rag")
    rag_stub.create_rag_graph = lambda: "unused"
    module = _load_module(
        monkeypatch,
        "05_rag_as_tool.py",
        preload_modules={"examples.langgraph.standalone.rag": rag_stub},
    )
    monkeypatch.setenv("THENVOI_WS_URL", "wss://ws.example")
    monkeypatch.setenv("THENVOI_REST_URL", "https://rest.example")

    monkeypatch.setattr(module, "load_dotenv", MagicMock())
    monkeypatch.setattr(module, "load_agent_config", MagicMock(return_value=("rag-id", "rag-key")))
    monkeypatch.setattr(module, "create_rag_graph", MagicMock(return_value="rag-graph"))
    monkeypatch.setattr(module, "graph_as_tool", MagicMock(return_value="rag-tool"))
    monkeypatch.setattr(module, "ChatOpenAI", MagicMock(return_value="llm"))
    monkeypatch.setattr(module, "InMemorySaver", MagicMock(return_value="checkpointer"))
    adapter_instance = object()
    mock_adapter_cls = MagicMock(return_value=adapter_instance)
    runtime_agent = SimpleNamespace(run=AsyncMock())
    mock_create = MagicMock(return_value=runtime_agent)
    monkeypatch.setattr(module, "LangGraphAdapter", mock_adapter_cls)
    monkeypatch.setattr(module.Agent, "create", mock_create)

    await module.main()

    module.create_rag_graph.assert_called_once()
    module.graph_as_tool.assert_called_once()
    assert mock_adapter_cls.call_args.kwargs["additional_tools"] == ["rag-tool"]
    runtime_agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_delegate_to_sql_agent_main_builds_tool_and_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sql_stub = ModuleType("examples.langgraph.standalone.sql_agent")
    sql_stub.create_sql_agent = lambda _db_path: "unused"
    sql_stub.download_chinook_db = lambda: "unused.db"
    module = _load_module(
        monkeypatch,
        "06_delegate_to_sql_agent.py",
        preload_modules={"examples.langgraph.standalone.sql_agent": sql_stub},
    )
    monkeypatch.setenv("THENVOI_WS_URL", "wss://ws.example")
    monkeypatch.setenv("THENVOI_REST_URL", "https://rest.example")

    monkeypatch.setattr(module, "load_dotenv", MagicMock())
    monkeypatch.setattr(module, "load_agent_config", MagicMock(return_value=("sql-id", "sql-key")))
    monkeypatch.setattr(module, "download_chinook_db", MagicMock(return_value="Chinook.db"))
    monkeypatch.setattr(module, "create_sql_agent", MagicMock(return_value="sql-graph"))
    monkeypatch.setattr(module, "graph_as_tool", MagicMock(return_value="sql-tool"))
    monkeypatch.setattr(module, "ChatOpenAI", MagicMock(return_value="llm"))
    monkeypatch.setattr(module, "InMemorySaver", MagicMock(return_value="checkpointer"))
    adapter_instance = object()
    mock_adapter_cls = MagicMock(return_value=adapter_instance)
    runtime_agent = SimpleNamespace(run=AsyncMock())
    mock_create = MagicMock(return_value=runtime_agent)
    monkeypatch.setattr(module, "LangGraphAdapter", mock_adapter_cls)
    monkeypatch.setattr(module.Agent, "create", mock_create)

    await module.main()

    module.download_chinook_db.assert_called_once()
    module.create_sql_agent.assert_called_once_with("Chinook.db")
    module.graph_as_tool.assert_called_once()
    assert mock_adapter_cls.call_args.kwargs["additional_tools"] == ["sql-tool"]
    runtime_agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_tom_agent_uses_generated_prompt_and_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch, "07_tom_agent.py")
    monkeypatch.setattr(module, "ChatOpenAI", MagicMock(return_value="llm"))
    monkeypatch.setattr(module, "InMemorySaver", MagicMock(return_value="checkpointer"))
    monkeypatch.setattr(module, "generate_tom_prompt", MagicMock(return_value="tom-prompt"))
    adapter_instance = object()
    mock_adapter_cls = MagicMock(return_value=adapter_instance)
    session = SimpleNamespace(agent=SimpleNamespace(run=AsyncMock()))
    mock_bootstrap = MagicMock(return_value=session)
    monkeypatch.setattr(module, "LangGraphAdapter", mock_adapter_cls)
    monkeypatch.setattr(module, "bootstrap_agent", mock_bootstrap)

    await module.main()

    mock_adapter_cls.assert_called_once_with(
        llm="llm",
        checkpointer="checkpointer",
        custom_section="tom-prompt",
    )
    session.agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_jerry_agent_requires_rest_url(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch, "08_jerry_agent.py")
    monkeypatch.setenv("THENVOI_WS_URL", "wss://ws.example")
    monkeypatch.delenv("THENVOI_REST_URL", raising=False)
    monkeypatch.setattr(module, "load_dotenv", MagicMock())

    with pytest.raises(ValueError, match="THENVOI_REST_URL environment variable is required"):
        await module.main()


@pytest.mark.asyncio
async def test_jerry_agent_creates_adapter_and_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch, "08_jerry_agent.py")
    monkeypatch.setenv("THENVOI_WS_URL", "wss://ws.example")
    monkeypatch.setenv("THENVOI_REST_URL", "https://rest.example")

    monkeypatch.setattr(module, "load_dotenv", MagicMock())
    monkeypatch.setattr(module, "load_agent_config", MagicMock(return_value=("jerry-id", "jerry-key")))
    monkeypatch.setattr(module, "generate_jerry_prompt", MagicMock(return_value="jerry-prompt"))
    monkeypatch.setattr(module, "ChatOpenAI", MagicMock(return_value="llm"))
    monkeypatch.setattr(module, "InMemorySaver", MagicMock(return_value="checkpointer"))
    adapter_instance = object()
    mock_adapter_cls = MagicMock(return_value=adapter_instance)
    runtime_agent = SimpleNamespace(run=AsyncMock())
    mock_create = MagicMock(return_value=runtime_agent)
    monkeypatch.setattr(module, "LangGraphAdapter", mock_adapter_cls)
    monkeypatch.setattr(module.Agent, "create", mock_create)

    await module.main()

    mock_adapter_cls.assert_called_once_with(
        llm="llm",
        checkpointer="checkpointer",
        custom_section="jerry-prompt",
    )
    mock_create.assert_called_once_with(
        adapter=adapter_instance,
        agent_id="jerry-id",
        api_key="jerry-key",
        ws_url="wss://ws.example",
        rest_url="https://rest.example",
    )
    runtime_agent.run.assert_awaited_once()
