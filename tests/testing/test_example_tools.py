"""Tests for shared example tool helpers."""

from __future__ import annotations

import pytest

from thenvoi.testing.example_tools import (
    build_example_tool_registry,
    calculator,
    get_time,
    random_number,
    safe_math_eval,
)


async def _invoke_tool(tool_obj: object, args: dict[str, object]) -> dict[str, object]:
    """Invoke tool handlers regardless of whether decorators wrap callables."""
    if callable(tool_obj):
        return await tool_obj(args)  # type: ignore[misc]
    return await tool_obj.handler(args)  # type: ignore[union-attr]


def test_safe_math_eval_supports_basic_and_function_calls() -> None:
    assert safe_math_eval("2 + 3 * 4") == 14
    assert safe_math_eval("abs(-5)") == 5


def test_safe_math_eval_rejects_oversized_expression() -> None:
    expression = "1" * 1001
    with pytest.raises(ValueError, match="Expression too long"):
        safe_math_eval(expression)


def test_safe_math_eval_rejects_unsupported_function() -> None:
    with pytest.raises(ValueError, match="Unsupported function"):
        safe_math_eval("eval('1+1')")


@pytest.mark.asyncio
async def test_calculator_returns_value_payload() -> None:
    result = await _invoke_tool(calculator, {"expression": "10 / 2"})
    assert result == {"content": [{"type": "text", "text": "5.0"}]}


@pytest.mark.asyncio
async def test_calculator_returns_error_payload_for_bad_expression() -> None:
    result = await _invoke_tool(calculator, {"expression": "pow(10001, 2)"})
    assert result["is_error"] is True
    assert "Error:" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_get_time_returns_iso_text() -> None:
    result = await _invoke_tool(get_time, {})
    assert result["content"][0]["type"] == "text"
    assert "T" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_random_number_handles_range_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("thenvoi.testing.example_tools.randint", lambda _a, _b: 7)

    ok = await _invoke_tool(random_number, {"min": 1, "max": 10})
    assert ok == {"content": [{"type": "text", "text": "7"}]}

    error = await _invoke_tool(random_number, {"min": 10, "max": 1})
    assert error["is_error"] is True
    assert "min must be <=" in error["content"][0]["text"]


def test_build_example_tool_registry_returns_copy() -> None:
    registry = build_example_tool_registry()
    registry.pop("calculator")

    fresh_registry = build_example_tool_registry()
    assert "calculator" in fresh_registry
