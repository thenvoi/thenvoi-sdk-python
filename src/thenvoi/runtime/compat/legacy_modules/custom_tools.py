"""Compatibility shim implementation for ``thenvoi.runtime.custom_tools``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenvoi.compat.shims import make_registered_module_shim

if TYPE_CHECKING:
    from thenvoi.runtime.tooling.custom_tools import (
        CustomToolDef as CustomToolDef,
        custom_tool_to_anthropic_schema as custom_tool_to_anthropic_schema,
        custom_tool_to_openai_schema as custom_tool_to_openai_schema,
        custom_tools_to_schemas as custom_tools_to_schemas,
        execute_custom_tool as execute_custom_tool,
        find_custom_tool as find_custom_tool,
        get_custom_tool_name as get_custom_tool_name,
    )

REMOVAL_VERSION, __all__, __getattr__ = make_registered_module_shim(
    "thenvoi.runtime.custom_tools",
)
