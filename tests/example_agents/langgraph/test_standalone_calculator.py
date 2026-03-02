"""Tests for examples/langgraph/standalone/calculator.py."""

from __future__ import annotations

import pytest

from examples.langgraph.standalone.calculator import calculate_node, create_calculator_graph


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "a", "b", "expected_result", "expected_error"),
    [
        ("add", 2.0, 3.0, 5.0, None),
        ("subtract", 10.0, 4.0, 6.0, None),
        ("multiply", 3.0, 4.0, 12.0, None),
        ("divide", 8.0, 2.0, 4.0, None),
        ("divide", 1.0, 0.0, None, "Cannot divide by zero"),
        ("invalid", 1.0, 2.0, None, "Unknown operation: invalid"),
    ],
)
async def test_calculate_node_handles_supported_and_error_cases(
    operation: str,
    a: float,
    b: float,
    expected_result: float | None,
    expected_error: str | None,
) -> None:
    result = await calculate_node(
        {
            "operation": operation,
            "a": a,
            "b": b,
            "result": None,
            "error": None,
        }
    )

    assert result["result"] == expected_result
    assert result["error"] == expected_error


@pytest.mark.asyncio
async def test_create_calculator_graph_compiles_and_invokes() -> None:
    graph = create_calculator_graph()

    state = {
        "operation": "add",
        "a": 7.0,
        "b": 5.0,
        "result": None,
        "error": None,
    }
    output = await graph.ainvoke(state, config={"configurable": {"thread_id": "test-1"}})

    assert output["result"] == 12.0
    assert output["error"] is None
