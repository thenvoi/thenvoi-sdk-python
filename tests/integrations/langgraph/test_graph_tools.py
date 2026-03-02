"""Tests for graph_as_tool error normalization."""

from __future__ import annotations

import pytest

from thenvoi.integrations.langgraph.graph_tools import graph_as_tool


class FailingGraph:
    """Graph double that always fails."""

    async def ainvoke(
        self,
        _input: dict[str, object],
        _config: dict[str, object] | None = None,
    ) -> dict[str, object]:
        raise RuntimeError("subgraph unavailable")


class SuccessfulGraph:
    """Graph double that returns a deterministic result."""

    async def ainvoke(
        self,
        _input: dict[str, object],
        _config: dict[str, object] | None = None,
    ) -> dict[str, int]:
        return {"result": 42}


@pytest.mark.asyncio
async def test_graph_as_tool_returns_normalized_error_message() -> None:
    """Subgraph exceptions should not bubble as raw framework errors."""
    wrapped = graph_as_tool(
        graph=FailingGraph(),
        name="calculator",
        description="Compute arithmetic results",
        input_schema={"operation": "Operation name"},
    )

    result = await wrapped.ainvoke({"operation": "divide"})

    assert result == "Error executing calculator: subgraph unavailable"


@pytest.mark.asyncio
async def test_graph_as_tool_normalizes_formatter_errors() -> None:
    """Formatter failures should follow the same tool error contract."""

    def failing_formatter(_state: dict[str, int]) -> str:
        raise ValueError("bad formatter")

    wrapped = graph_as_tool(
        graph=SuccessfulGraph(),
        name="calculator",
        description="Compute arithmetic results",
        input_schema={"operation": "Operation name"},
        result_formatter=failing_formatter,
    )

    result = await wrapped.ainvoke({"operation": "add"})

    assert result == "Error executing calculator: bad formatter"
