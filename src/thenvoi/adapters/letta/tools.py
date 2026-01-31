"""
Tool utilities for Letta adapter.

Registers Thenvoi tools with Letta server using schemas from runtime.tools.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import TYPE_CHECKING, Any, Callable, get_type_hints

from thenvoi.runtime.tools import TOOL_MODELS, get_tool_description

if TYPE_CHECKING:
    from letta_client import Letta

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Thenvoi Tool Names
# ══════════════════════════════════════════════════════════════════════════════

# Tools to register with Letta (subset of TOOL_MODELS - excludes send_event which is internal)
THENVOI_TOOL_NAMES = frozenset(
    [
        "send_message",
        "add_participant",
        "remove_participant",
        "lookup_peers",
        "get_participants",
        "create_chatroom",
    ]
)


# ══════════════════════════════════════════════════════════════════════════════
# Schema Generation
# ══════════════════════════════════════════════════════════════════════════════


def _get_letta_tool_schema(tool_name: str) -> dict[str, Any]:
    """
    Get JSON schema for a tool in Letta format.

    Uses Pydantic models from runtime.tools as the single source of truth.
    For send_message, renames 'content' to 'message' to match Letta's expectations.

    Args:
        tool_name: Name of the tool

    Returns:
        JSON schema dict for Letta
    """
    model = TOOL_MODELS[tool_name]
    schema = model.model_json_schema()

    # Remove Pydantic-specific keys
    schema.pop("title", None)

    # Special case: Letta expects 'message' not 'content' for send_message
    if tool_name == "send_message" and "properties" in schema:
        props = schema["properties"]
        if "content" in props:
            # Rename content -> message
            props["message"] = props.pop("content")
            props["message"]["description"] = "The message content to send"

            # Update required list
            if "required" in schema and "content" in schema["required"]:
                schema["required"] = [
                    "message" if r == "content" else r for r in schema["required"]
                ]

    return schema


def _generate_stub_source(tool_name: str) -> str:
    """
    Generate minimal Python stub source code for a tool.

    Letta API requires source_code, but we override the schema via json_schema.
    This stub just needs valid Python syntax.

    Args:
        tool_name: Name of the tool

    Returns:
        Minimal Python function source code
    """
    description = get_tool_description(tool_name)
    return f'''
def {tool_name}(**kwargs) -> str:
    """{description}"""
    return "executed"
'''


# ══════════════════════════════════════════════════════════════════════════════
# Tool Registration
# ══════════════════════════════════════════════════════════════════════════════


def register_thenvoi_tools(client: "Letta") -> dict[str, str]:
    """
    Register Thenvoi tools with Letta server.

    Uses JSON schemas from runtime.tools (single source of truth).
    Passes json_schema parameter to override Letta's source code parsing.

    Args:
        client: Letta client instance

    Returns:
        Dict mapping tool name to tool ID
    """
    tool_ids: dict[str, str] = {}

    for name in THENVOI_TOOL_NAMES:
        if name not in TOOL_MODELS:
            logger.warning(f"Tool {name} not found in TOOL_MODELS, skipping")
            continue

        try:
            schema = _get_letta_tool_schema(name)
            description = get_tool_description(name)
            source_code = _generate_stub_source(name)

            tool = client.tools.upsert(
                source_code=source_code,
                description=description,
                json_schema=schema,
                tags=["thenvoi"],
            )
            tool_ids[name] = tool.id
            logger.debug(f"Registered Thenvoi tool: {name} (id: {tool.id})")
        except Exception as e:
            logger.error(f"Failed to register tool {name}: {e}")
            raise

    logger.info(f"Registered {len(tool_ids)} Thenvoi tools with Letta")
    return tool_ids


def get_letta_tool_ids(
    client: "Letta",
    thenvoi_tool_ids: dict[str, str],
    include_memory_tools: bool = True,
) -> list[str]:
    """
    Get tool IDs to attach to a Letta agent.

    Includes Thenvoi tools and optionally Letta's memory tools.

    Args:
        client: Letta client instance
        thenvoi_tool_ids: Dict from register_thenvoi_tools()
        include_memory_tools: Whether to include memory_replace, memory_insert, etc.

    Returns:
        List of tool IDs to pass to agents.create(tool_ids=...)
    """
    tool_ids = list(thenvoi_tool_ids.values())

    if include_memory_tools:
        # Find Letta's built-in memory tools
        memory_tool_names = {"memory_replace", "memory_insert", "conversation_search"}
        for tool in client.tools.list():
            if tool.name in memory_tool_names and "letta" in (tool.tags or []):
                tool_ids.append(tool.id)

    return tool_ids


# ══════════════════════════════════════════════════════════════════════════════
# Custom Tool Builder (for user-defined tools)
# ══════════════════════════════════════════════════════════════════════════════


class CustomToolBuilder:
    """
    Builder for custom Letta tools.

    Supports creating tools from:
    - Functions with type hints
    - Async functions

    Example:
        builder = CustomToolBuilder()

        @builder.tool
        def calculate(operation: str, a: float, b: float) -> str:
            '''Perform a calculation.'''
            if operation == "add":
                return str(a + b)
            # ...

        # Get tool definitions for Letta
        tool_defs = builder.get_tool_definitions()
    """

    def __init__(self):
        self._tools: dict[str, dict[str, Any]] = {}
        self._executors: dict[str, Callable[..., Any]] = {}

    def tool(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator to register a function as a tool."""
        name = func.__name__
        doc = func.__doc__ or f"Execute {name}"

        # Get type hints for parameters
        hints = get_type_hints(func)
        sig = inspect.signature(func)

        # Build JSON schema for parameters
        properties: dict[str, dict[str, Any]] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_type = hints.get(param_name, str)
            json_type = self._python_type_to_json(param_type)

            properties[param_name] = {
                "type": json_type,
                "description": f"Parameter {param_name}",
            }

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        self._tools[name] = {
            "name": name,
            "description": doc.strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

        self._executors[name] = func

        return func

    def register(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """
        Register a tool programmatically.

        Args:
            name: Tool name
            description: Tool description
            func: Function to execute
            parameters: Optional JSON schema for parameters
        """
        self._tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters or {"type": "object", "properties": {}},
        }
        self._executors[name] = func

    def _python_type_to_json(self, py_type: type) -> str:
        """Convert Python type to JSON schema type."""
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }
        return type_map.get(py_type, "string")

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get all registered tool definitions."""
        return list(self._tools.values())

    def get_tool_names(self) -> list[str]:
        """Get names of all registered tools."""
        return list(self._tools.keys())

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._executors

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a registered tool."""
        if tool_name not in self._executors:
            raise ValueError(f"Unknown tool: {tool_name}")

        func = self._executors[tool_name]

        # Handle async and sync functions
        if asyncio.iscoroutinefunction(func):
            return await func(**arguments)
        else:
            return func(**arguments)
