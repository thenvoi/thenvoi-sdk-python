"""Unit tests for runtime tool binding helpers."""

from __future__ import annotations

from typing import Any, Literal
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from thenvoi.runtime.tool_bindings import ToolBindingRegistry


class _NestedModel(BaseModel):
    value: int


class _ArgsModel(BaseModel):
    text: str
    optional_number: int | None = None
    values: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    mode: Literal["fast", "slow"] = "fast"
    nested: _NestedModel


class _OverrideModel(BaseModel):
    replacement: str


def test_claude_mcp_schema_maps_annotations_to_runtime_types() -> None:
    registry = ToolBindingRegistry({"demo_tool": _ArgsModel})

    schema = registry.claude_mcp_schema("demo_tool")

    assert schema["room_id"] is str
    assert schema["text"] is str
    assert schema["optional_number"] is int
    assert schema["values"] is list
    assert schema["metadata"] is dict
    assert schema["mode"] is str
    assert schema["nested"] is dict


def test_claude_mcp_schema_can_skip_room_id() -> None:
    registry = ToolBindingRegistry({"demo_tool": _ArgsModel})

    schema = registry.claude_mcp_schema("demo_tool", include_room_id=False)

    assert "room_id" not in schema
    assert schema["text"] is str


def test_pydantic_parameters_preserve_required_vs_optional_defaults() -> None:
    registry = ToolBindingRegistry({"demo_tool": _ArgsModel})

    parameters = registry.pydantic_parameters("demo_tool")
    by_name = {param.name: param for param in parameters}

    assert by_name["text"].default is by_name["text"].empty
    assert by_name["optional_number"].default is None
    assert by_name["text"].annotation is str


def test_crewai_schemas_respects_tool_order_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ToolBindingRegistry({"alpha": _ArgsModel, "beta": _ArgsModel})
    monkeypatch.setattr(
        registry,
        "tool_names",
        lambda *, include_memory_tools: ["beta", "alpha"],
    )

    schemas = registry.crewai_schemas(
        include_memory_tools=False,
        overrides={"alpha": _OverrideModel},
    )

    assert list(schemas) == ["beta", "alpha"]
    assert schemas["beta"] is _ArgsModel
    assert schemas["alpha"] is _OverrideModel


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (Any, str),
        (Literal[True, False], bool),
        (Literal[1, 2], int),
        (Literal[1.5, 2.5], float),
        (Literal["a", "b"], str),
        (list[int], list),
        (dict[str, str], dict),
        (int | None, int),
    ],
)
def test_annotation_to_runtime_type(annotation: Any, expected: type[Any]) -> None:
    assert ToolBindingRegistry._annotation_to_runtime_type(annotation) is expected


def test_claude_mcp_schemas_for_all_enabled_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ToolBindingRegistry({"alpha": _ArgsModel, "beta": _ArgsModel})
    monkeypatch.setattr(
        registry,
        "iter_models",
        MagicMock(return_value=[("alpha", _ArgsModel), ("beta", _ArgsModel)]),
    )

    schemas = registry.claude_mcp_schemas(include_memory_tools=False)

    assert set(schemas) == {"alpha", "beta"}
    assert schemas["alpha"]["room_id"] is str
    assert schemas["beta"]["text"] is str
