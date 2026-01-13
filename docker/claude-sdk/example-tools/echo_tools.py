"""
Example custom tools for Claude SDK Docker.

This file demonstrates how to create custom tools that can be mounted
into the Claude SDK Docker container.

Usage:
    Mount this directory to /app/tools in the container.
    
Example:
    docker run -v ./docker/claude-sdk/example-tools:/app/tools:ro \\
               -v ./my-agent/config:/app/config:ro \\
               thenvoi-claude-sdk
"""
from __future__ import annotations

import os
from typing import Any

# Tool registry - the container discovers this list
TOOLS: list[dict[str, Any]] = []


def register_tool(name: str, description: str, parameters: dict[str, Any]):
    """
    Decorator to register a tool.

    Args:
        name: Tool name (used by Claude to call the tool)
        description: What the tool does (shown to Claude)
        parameters: Dict of parameter definitions with type, description, and optional default
    """

    def decorator(func):
        TOOLS.append(
            {
                "name": name,
                "description": description,
                "parameters": parameters,
                "handler": func,
            }
        )
        return func

    return decorator


@register_tool(
    name="echo",
    description="Echo back the input message. Useful for testing.",
    parameters={"message": {"type": "string", "description": "The message to echo back"}},
)
async def echo(args: dict[str, Any]) -> dict[str, Any]:
    """Simple echo tool for testing."""
    message = args.get("message", "")
    return {"status": "success", "echoed": message, "length": len(message)}


@register_tool(
    name="add_numbers",
    description="Add two numbers together.",
    parameters={
        "a": {"type": "number", "description": "First number"},
        "b": {"type": "number", "description": "Second number"},
    },
)
async def add_numbers(args: dict[str, Any]) -> dict[str, Any]:
    """Add two numbers."""
    a = args.get("a", 0)
    b = args.get("b", 0)
    return {"status": "success", "result": a + b, "expression": f"{a} + {b} = {a + b}"}


@register_tool(
    name="list_data_files",
    description="List files in the /app/data directory.",
    parameters={},
)
async def list_data_files(args: dict[str, Any]) -> dict[str, Any]:
    """List files in the data volume."""
    data_path = "/app/data"
    if not os.path.exists(data_path):
        return {"status": "error", "message": "Data directory not found"}

    files = []
    for item in os.listdir(data_path):
        item_path = os.path.join(data_path, item)
        files.append(
            {
                "name": item,
                "is_directory": os.path.isdir(item_path),
                "size": os.path.getsize(item_path) if os.path.isfile(item_path) else None,
            }
        )

    return {"status": "success", "path": data_path, "files": files, "count": len(files)}


@register_tool(
    name="read_data_file",
    description="Read contents of a file from /app/data directory.",
    parameters={
        "filename": {
            "type": "string",
            "description": "Name of the file to read (relative to /app/data)",
        }
    },
)
async def read_data_file(args: dict[str, Any]) -> dict[str, Any]:
    """Read a file from the data volume."""
    filename = args.get("filename", "")
    if not filename:
        return {"status": "error", "message": "Filename is required"}

    # Basic path safety (prevent reading outside /app/data)
    filepath = os.path.normpath(os.path.join("/app/data", filename))
    if not filepath.startswith("/app/data/"):
        return {"status": "error", "message": "Invalid path"}

    if not os.path.exists(filepath):
        return {"status": "error", "message": f"File not found: {filename}"}

    if os.path.isdir(filepath):
        return {"status": "error", "message": f"Path is a directory: {filename}"}

    try:
        with open(filepath) as f:
            content = f.read()
        return {
            "status": "success",
            "filename": filename,
            "content": content,
            "size": len(content),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@register_tool(
    name="write_data_file",
    description="Write content to a file in /app/data directory.",
    parameters={
        "filename": {
            "type": "string",
            "description": "Name of the file to write (relative to /app/data)",
        },
        "content": {"type": "string", "description": "Content to write to the file"},
    },
)
async def write_data_file(args: dict[str, Any]) -> dict[str, Any]:
    """Write a file to the data volume."""
    filename = args.get("filename", "")
    content = args.get("content", "")

    if not filename:
        return {"status": "error", "message": "Filename is required"}

    # Basic path safety (prevent writing outside /app/data)
    filepath = os.path.normpath(os.path.join("/app/data", filename))
    if not filepath.startswith("/app/data/"):
        return {"status": "error", "message": "Invalid path"}

    try:
        # Create parent directories if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w") as f:
            f.write(content)

        return {
            "status": "success",
            "filename": filename,
            "path": filepath,
            "size": len(content),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
