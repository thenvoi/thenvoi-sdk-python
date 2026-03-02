"""Tests for shared adapter-level platform tool binding helpers."""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from pydantic import BaseModel

from thenvoi.adapters.platform_tool_bindings import (
    build_pydantic_tool_function,
    crewai_tool_bindings,
    platform_tool_names,
)


class _OverrideModel(BaseModel):
    content: str


def test_platform_tool_names_include_core_tools() -> None:
    names = platform_tool_names(include_memory_tools=False)
    assert "thenvoi_send_message" in names
    assert "thenvoi_send_event" in names


def test_crewai_tool_bindings_apply_schema_overrides() -> None:
    bindings = crewai_tool_bindings(
        include_memory_tools=False,
        overrides={"thenvoi_send_message": _OverrideModel},
    )
    by_name = {binding.name: binding for binding in bindings}
    assert by_name["thenvoi_send_message"].args_schema is _OverrideModel


@pytest.mark.asyncio
async def test_build_pydantic_tool_function_exposes_signature_and_invokes() -> None:
    calls: list[tuple[Any, str, dict[str, Any]]] = []

    async def _invoker(ctx: Any, tool_name: str, kwargs: dict[str, Any]) -> Any:
        calls.append((ctx, tool_name, kwargs))
        return {"ok": True}

    func = build_pydantic_tool_function(
        "thenvoi_send_event",
        context_annotation=object,
        invoker=_invoker,
    )
    signature = inspect.signature(func)
    assert signature.parameters["ctx"].annotation is object
    result = await func("ctx-value", content="thought", message_type="thought")
    assert result == {"ok": True}
    assert calls[0][0] == "ctx-value"
    assert calls[0][1] == "thenvoi_send_event"

