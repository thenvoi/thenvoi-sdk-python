"""Tests for tool schema conversion helpers."""

from __future__ import annotations

from pydantic import BaseModel
import pytest

from thenvoi.runtime.tool_schema_provider import ToolSchemaProvider


class _VisibleToolInput(BaseModel):
    """Visible tool description."""

    query: str


class _MemoryToolInput(BaseModel):
    """Memory tool description."""

    memory_id: str


def _provider() -> ToolSchemaProvider:
    return ToolSchemaProvider(
        tool_models={
            "visible_tool": _VisibleToolInput,
            "memory_tool": _MemoryToolInput,
        },
        memory_tool_names=frozenset({"memory_tool"}),
    )


def test_get_openai_tool_schemas_filters_memory_tools_by_default() -> None:
    provider = _provider()

    schemas = provider.get_openai_tool_schemas()

    assert len(schemas) == 1
    tool = schemas[0]["function"]
    assert tool["name"] == "visible_tool"
    assert tool["description"] == "Visible tool description."
    assert "title" not in tool["parameters"]


def test_get_anthropic_tool_schemas_includes_memory_tools_when_requested() -> None:
    provider = _provider()

    schemas = provider.get_anthropic_tool_schemas(include_memory=True)

    assert {schema["name"] for schema in schemas} == {"visible_tool", "memory_tool"}
    for schema in schemas:
        assert "title" not in schema["input_schema"]


def test_get_tool_schemas_rejects_invalid_format() -> None:
    provider = _provider()

    with pytest.raises(ValueError, match="Invalid format"):
        provider.get_tool_schemas("xml")


def test_tool_models_property_returns_registered_models() -> None:
    provider = _provider()

    assert set(provider.tool_models) == {"visible_tool", "memory_tool"}
    assert provider.tool_models["visible_tool"] is _VisibleToolInput
