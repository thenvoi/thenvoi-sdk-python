# Generic Claude SDK Docker Design

## Overview

A Docker container that allows users to:
1. Mount their own Python code with custom tools
2. Configure Claude SDK via YAML/environment variables
3. Run custom agents without modifying the base image

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Claude SDK Docker Container                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────────────────────┐  │
│  │   Node.js   │  │   Python    │  │    Thenvoi SDK             │  │
│  │   20.x      │  │   3.11+     │  │    + claude-agent-sdk      │  │
│  └─────────────┘  └─────────────┘  └────────────────────────────┘  │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                          MCP SERVERS                                 │
│  ┌──────────────────────┐  ┌────────────────────────────────────┐  │
│  │  Thenvoi MCP Server  │  │  Custom Tools MCP Server           │  │
│  │  (send_message, etc) │  │  (user-defined tools)              │  │
│  └──────────────────────┘  └────────────────────────────────────┘  │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                        VOLUME MOUNTS                                 │
├─────────────────┬──────────────────┬───────────────────────────────┤
│  /app/config    │  /app/tools      │  /app/scripts                 │
│  (agent config) │  (custom tools)  │  (user agent code)            │
│  Read-only      │  Read-only       │  Read-only                    │
├─────────────────┴──────────────────┴───────────────────────────────┤
│  /app/data                                                          │
│  (working data - databases, files, etc.)                            │
│  Read-Write                                                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. Volume Mounts

| Mount Point | Purpose | Access |
|-------------|---------|--------|
| `/app/config` | Agent config YAML, `.env` | Read-only |
| `/app/tools` | Custom Python tool modules | Read-only |
| `/app/scripts` | User's custom agent scripts | Read-only |
| `/app/data` | Working data (DBs, files, outputs) | Read-Write |

### 2. Configuration Files

**`/app/config/agent.yaml`** - Main configuration:
```yaml
# Agent identity (from Thenvoi platform)
agent:
  id: "your-agent-id"
  api_key: "your-api-key"

# Claude SDK settings
claude:
  model: "claude-sonnet-4-5-20250929"
  max_thinking_tokens: 10000  # null to disable
  permission_mode: "acceptEdits"  # default, acceptEdits, plan, bypassPermissions

# Custom instructions
system:
  custom_section: |
    You are a helpful assistant specialized in...

# Tools configuration
tools:
  # Built-in Thenvoi tools (always available)
  thenvoi:
    enabled: true
  
  # Custom user tools (auto-discovered from /app/tools/*.py)
  custom:
    enabled: true

# Execution settings
execution:
  enable_reporting: true   # Send tool_call/tool_result events
  log_level: "INFO"
```

### 3. Custom Tools Interface

Users define tools in `/app/tools/` - all `.py` files are auto-discovered.

**`/app/tools/my_tools.py`**:
```python
"""Custom tools for Claude SDK."""
from typing import Any

# Tool registry - the container discovers these
TOOLS = []

def register_tool(name: str, description: str, parameters: dict):
    """Decorator to register a tool."""
    def decorator(func):
        TOOLS.append({
            "name": name,
            "description": description,
            "parameters": parameters,
            "handler": func,
        })
        return func
    return decorator

@register_tool(
    name="search_database",
    description="Search the local database for records",
    parameters={
        "query": {"type": "string", "description": "Search query"},
        "limit": {"type": "integer", "description": "Max results", "default": 10},
    }
)
async def search_database(args: dict[str, Any]) -> dict[str, Any]:
    """Search database implementation."""
    query = args.get("query", "")
    limit = args.get("limit", 10)
    
    # Access data from mounted volume
    # /app/data is read-write
    import sqlite3
    conn = sqlite3.connect("/app/data/my_database.db")
    # ... implementation
    
    return {"results": [...], "count": len(results)}

@register_tool(
    name="write_file",
    description="Write content to a file in the data directory",
    parameters={
        "filename": {"type": "string", "description": "File name"},
        "content": {"type": "string", "description": "File content"},
    }
)
async def write_file(args: dict[str, Any]) -> dict[str, Any]:
    """Write file to data volume."""
    filename = args["filename"]
    content = args["content"]
    
    path = f"/app/data/{filename}"
    with open(path, "w") as f:
        f.write(content)
    
    return {"status": "success", "path": path}
```

### 4. Dockerfile

```dockerfile
# Generic Claude SDK Docker Container
# Supports custom tools and volume mounts

FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20+ (required for Claude CLI)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI globally
RUN npm install -g @anthropic-ai/claude-code

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy SDK source
COPY pyproject.toml ./
COPY src/ ./src/

# Install with Claude SDK extras
RUN uv sync --extra claude_sdk

# Create mount point directories
RUN mkdir -p /app/config /app/tools /app/scripts /app/data

# Copy entrypoint and runner
COPY docker/claude-sdk/entrypoint.sh /entrypoint.sh
COPY docker/claude-sdk/runner.py /app/runner.py
RUN chmod +x /entrypoint.sh

# Environment defaults
ENV LOG_LEVEL=INFO
ENV CONFIG_PATH=/app/config/agent.yaml
ENV TOOLS_PATH=/app/tools
ENV SCRIPTS_PATH=/app/scripts
ENV DATA_PATH=/app/data

ENTRYPOINT ["/entrypoint.sh"]
CMD ["run"]
```

### 5. Entrypoint Script

**`docker/claude-sdk/entrypoint.sh`**:
```bash
#!/bin/bash
set -e

# Load .env if exists (properly handles comments and quotes)
if [ -f /app/config/.env ]; then
    set -a
    source /app/config/.env
    set +a
fi

case "$1" in
    run)
        # Run the default agent runner
        exec uv run --extra claude_sdk python /app/runner.py
        ;;
    custom)
        # Run user's custom script
        shift
        exec uv run --extra claude_sdk python /app/scripts/"$@"
        ;;
    shell)
        # Interactive shell
        exec /bin/bash
        ;;
    *)
        # Pass through to any command
        exec "$@"
        ;;
esac
```

### 6. Generic Runner

**`docker/claude-sdk/runner.py`**:
```python
#!/usr/bin/env python3
"""
Generic Claude SDK Agent Runner.

Loads configuration from /app/config/agent.yaml
Discovers and registers tools from /app/tools/
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Check for PyYAML
try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)

try:
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        tool,
        create_sdk_mcp_server,
    )
except ImportError:
    print("ERROR: claude-agent-sdk is required. Install with: pip install claude-agent-sdk")
    sys.exit(1)

from thenvoi import Agent
from thenvoi.adapters import ClaudeSDKAdapter
from thenvoi.runtime.tools import get_tool_description

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load agent configuration from YAML."""
    try:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in config file: {e}")
        sys.exit(1)


def discover_tools(tools_path: str) -> list[dict]:
    """
    Discover and load custom tools from modules.
    
    Auto-discovers all .py files in the tools directory (except those starting with _).
    Each module should have a TOOLS list containing tool definitions.
    """
    tools = []
    tools_dir = Path(tools_path)
    
    if not tools_dir.exists():
        logger.info(f"Tools directory {tools_path} does not exist, skipping custom tools")
        return tools
    
    # Add tools path to sys.path for imports
    sys.path.insert(0, str(tools_dir))
    
    for py_file in sorted(tools_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
            
        module_name = py_file.stem
        logger.info(f"Loading tools from {module_name}")
        
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                logger.warning(f"Could not load spec for {module_name}")
                continue
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for TOOLS registry in module
            if hasattr(module, "TOOLS"):
                module_tools = getattr(module, "TOOLS")
                tools.extend(module_tools)
                logger.info(f"  Found {len(module_tools)} tools in {module_name}")
            else:
                logger.warning(f"  No TOOLS list found in {module_name}")
                
        except Exception as e:
            logger.error(f"Failed to load {module_name}: {e}")
    
    return tools


class CustomToolsClaudeSDKAdapter(ClaudeSDKAdapter):
    """
    Claude SDK adapter with custom tool support.
    
    Creates a separate MCP server for custom tools alongside the Thenvoi MCP server.
    
    NOTE: This requires claude-agent-sdk to support multiple MCP servers.
    If not supported, we need to create a single combined MCP server.
    """
    
    def __init__(self, custom_tools: list[dict], **kwargs):
        super().__init__(**kwargs)
        self._custom_tools = custom_tools
        self._custom_mcp_server = None
    
    async def on_started(self, agent_name: str, agent_description: str) -> None:
        """Create MCP servers after agent metadata is fetched."""
        # Create custom tools MCP server if we have custom tools
        if self._custom_tools:
            self._custom_mcp_server = self._create_custom_mcp_server()
            logger.info(f"Created custom MCP server with {len(self._custom_tools)} tools")
        
        # Call parent to create Thenvoi MCP server and session manager
        await super().on_started(agent_name, agent_description)
        
        # TODO: If claude-agent-sdk supports multiple MCP servers, add custom server here
        # Otherwise, we need to modify parent's _create_mcp_server to include custom tools
    
    def _create_custom_mcp_server(self):
        """
        Create MCP server for custom tools.
        
        Converts discovered tool definitions into claude-agent-sdk tool format.
        """
        adapter = self
        
        def _make_result(data: Any) -> dict[str, Any]:
            return {
                "content": [{"type": "text", "text": json.dumps(data, default=str)}]
            }

        def _make_error(error: str) -> dict[str, Any]:
            return {
                "content": [{"type": "text", "text": json.dumps({"status": "error", "message": error})}],
                "is_error": True,
            }
        
        # Convert custom tool definitions to MCP tools
        mcp_tools = []
        
        for tool_def in self._custom_tools:
            tool_name = tool_def["name"]
            tool_description = tool_def["description"]
            tool_parameters = tool_def["parameters"]
            tool_handler = tool_def["handler"]
            
            # Convert parameters to MCP format
            param_types = {}
            for param_name, param_info in tool_parameters.items():
                param_type = param_info.get("type", "string")
                # Map to Python types for MCP
                type_map = {"string": str, "integer": int, "boolean": bool, "number": float}
                param_types[param_name] = type_map.get(param_type, str)
            
            # Create wrapper function that calls the handler
            async def create_tool_wrapper(handler, name):
                @tool(name, tool_description, param_types)
                async def tool_wrapper(args: dict[str, Any]) -> dict[str, Any]:
                    try:
                        result = await handler(args)
                        return _make_result(result)
                    except Exception as e:
                        logger.error(f"Tool {name} failed: {e}", exc_info=True)
                        return _make_error(str(e))
                return tool_wrapper
            
            # Note: This is a simplified version - actual implementation needs async handling
            mcp_tools.append((tool_name, tool_description, param_types, tool_handler))
        
        # Create the MCP server with all custom tools
        # NOTE: Actual implementation depends on claude-agent-sdk API
        # This is a placeholder showing the intended structure
        logger.info(f"Custom MCP server would have tools: {[t[0] for t in mcp_tools]}")
        
        return mcp_tools  # Placeholder - return tool definitions for now


async def main():
    """Run the generic Claude SDK agent."""
    
    # Setup logging
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Load configuration
    config_path = os.getenv("CONFIG_PATH", "/app/config/agent.yaml")
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        logger.error("Mount your config to /app/config/agent.yaml")
        sys.exit(1)
    
    config = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")
    
    # Extract settings
    agent_config = config.get("agent", {})
    claude_config = config.get("claude", {})
    system_config = config.get("system", {})
    tools_config = config.get("tools", {})
    exec_config = config.get("execution", {})
    
    # Validate required fields (config takes priority, then env vars)
    agent_id = agent_config.get("id") or os.getenv("THENVOI_AGENT_ID")
    api_key = agent_config.get("api_key") or os.getenv("THENVOI_API_KEY")
    
    if not agent_id or not api_key:
        logger.error("agent.id and agent.api_key are required (in config or env vars)")
        sys.exit(1)
    
    # Discover custom tools if enabled
    custom_tools = []
    if tools_config.get("custom", {}).get("enabled", True):
        tools_path = os.getenv("TOOLS_PATH", "/app/tools")
        custom_tools = discover_tools(tools_path)
        logger.info(f"Discovered {len(custom_tools)} custom tools")
    
    # Create adapter - use custom adapter if we have custom tools
    if custom_tools:
        adapter = CustomToolsClaudeSDKAdapter(
            custom_tools=custom_tools,
            model=claude_config.get("model", "claude-sonnet-4-5-20250929"),
            custom_section=system_config.get("custom_section"),
            max_thinking_tokens=claude_config.get("max_thinking_tokens"),
            permission_mode=claude_config.get("permission_mode", "acceptEdits"),
            enable_execution_reporting=exec_config.get("enable_reporting", True),
        )
    else:
        adapter = ClaudeSDKAdapter(
            model=claude_config.get("model", "claude-sonnet-4-5-20250929"),
            custom_section=system_config.get("custom_section"),
            max_thinking_tokens=claude_config.get("max_thinking_tokens"),
            permission_mode=claude_config.get("permission_mode", "acceptEdits"),
            enable_execution_reporting=exec_config.get("enable_reporting", True),
        )
    
    # Create agent
    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=os.getenv("THENVOI_WS_URL", "wss://api.thenvoi.com/ws"),
        rest_url=os.getenv("THENVOI_REST_API_URL", "https://api.thenvoi.com"),
    )
    
    logger.info("Starting Claude SDK agent...")
    logger.info(f"  Agent ID: {agent_id}")
    logger.info(f"  Model: {claude_config.get('model', 'claude-sonnet-4-5-20250929')}")
    logger.info(f"  Custom tools: {len(custom_tools)}")
    logger.info("Press Ctrl+C to stop")
    
    try:
        await agent.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Example Tools

**`docker/claude-sdk/example-tools/echo_tools.py`**:
```python
"""
Example custom tools for Claude SDK Docker.

This file demonstrates how to create custom tools that can be mounted
into the Claude SDK Docker container.

Usage:
    Mount this directory to /app/tools in the container.
"""
from typing import Any

# Tool registry - the container discovers this list
TOOLS = []


def register_tool(name: str, description: str, parameters: dict):
    """
    Decorator to register a tool.
    
    Args:
        name: Tool name (used by Claude to call the tool)
        description: What the tool does (shown to Claude)
        parameters: Dict of parameter definitions with type, description, and optional default
    """
    def decorator(func):
        TOOLS.append({
            "name": name,
            "description": description,
            "parameters": parameters,
            "handler": func,
        })
        return func
    return decorator


@register_tool(
    name="echo",
    description="Echo back the input message. Useful for testing.",
    parameters={
        "message": {
            "type": "string",
            "description": "The message to echo back"
        }
    }
)
async def echo(args: dict[str, Any]) -> dict[str, Any]:
    """Simple echo tool for testing."""
    message = args.get("message", "")
    return {
        "status": "success",
        "echoed": message,
        "length": len(message)
    }


@register_tool(
    name="add_numbers",
    description="Add two numbers together.",
    parameters={
        "a": {
            "type": "number",
            "description": "First number"
        },
        "b": {
            "type": "number",
            "description": "Second number"
        }
    }
)
async def add_numbers(args: dict[str, Any]) -> dict[str, Any]:
    """Add two numbers."""
    a = args.get("a", 0)
    b = args.get("b", 0)
    return {
        "status": "success",
        "result": a + b,
        "expression": f"{a} + {b} = {a + b}"
    }


@register_tool(
    name="list_data_files",
    description="List files in the /app/data directory.",
    parameters={}
)
async def list_data_files(args: dict[str, Any]) -> dict[str, Any]:
    """List files in the data volume."""
    import os
    
    data_path = "/app/data"
    if not os.path.exists(data_path):
        return {"status": "error", "message": "Data directory not found"}
    
    files = []
    for item in os.listdir(data_path):
        item_path = os.path.join(data_path, item)
        files.append({
            "name": item,
            "is_directory": os.path.isdir(item_path),
            "size": os.path.getsize(item_path) if os.path.isfile(item_path) else None
        })
    
    return {
        "status": "success",
        "path": data_path,
        "files": files,
        "count": len(files)
    }


@register_tool(
    name="read_data_file",
    description="Read contents of a file from /app/data directory.",
    parameters={
        "filename": {
            "type": "string",
            "description": "Name of the file to read (relative to /app/data)"
        }
    }
)
async def read_data_file(args: dict[str, Any]) -> dict[str, Any]:
    """Read a file from the data volume."""
    import os
    
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
        with open(filepath, "r") as f:
            content = f.read()
        return {
            "status": "success",
            "filename": filename,
            "content": content,
            "size": len(content)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@register_tool(
    name="write_data_file",
    description="Write content to a file in /app/data directory.",
    parameters={
        "filename": {
            "type": "string",
            "description": "Name of the file to write (relative to /app/data)"
        },
        "content": {
            "type": "string",
            "description": "Content to write to the file"
        }
    }
)
async def write_data_file(args: dict[str, Any]) -> dict[str, Any]:
    """Write a file to the data volume."""
    import os
    
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
            "size": len(content)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
```

---

## Docker Compose Example

**`docker-compose.claude-sdk.yml`**:
```yaml
version: "3.8"

x-common-env: &common-env
  THENVOI_REST_API_URL: ${THENVOI_REST_API_URL:-https://api.thenvoi.com}
  THENVOI_WS_URL: ${THENVOI_WS_URL:-wss://api.thenvoi.com/ws}
  ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
  LOG_LEVEL: ${LOG_LEVEL:-INFO}

services:
  # Basic Claude SDK agent with config only
  claude-basic:
    build:
      context: .
      dockerfile: docker/claude-sdk/Dockerfile
    environment:
      <<: *common-env
    volumes:
      - ./my-agent/config:/app/config:ro
    command: ["run"]

  # Claude SDK with custom tools
  claude-with-tools:
    build:
      context: .
      dockerfile: docker/claude-sdk/Dockerfile
    environment:
      <<: *common-env
    volumes:
      - ./my-agent/config:/app/config:ro
      - ./my-agent/tools:/app/tools:ro
      - ./my-agent/data:/app/data:rw
    command: ["run"]

  # Claude SDK with example tools (for testing)
  claude-example:
    build:
      context: .
      dockerfile: docker/claude-sdk/Dockerfile
    environment:
      <<: *common-env
    volumes:
      - ./my-agent/config:/app/config:ro
      - ./docker/claude-sdk/example-tools:/app/tools:ro
      - ./my-agent/data:/app/data:rw
    command: ["run"]

  # Run user's custom script
  claude-custom-script:
    build:
      context: .
      dockerfile: docker/claude-sdk/Dockerfile
    environment:
      <<: *common-env
    volumes:
      - ./my-agent/config:/app/config:ro
      - ./my-agent/tools:/app/tools:ro
      - ./my-agent/scripts:/app/scripts:ro
      - ./my-agent/data:/app/data:rw
    command: ["custom", "my_custom_agent.py"]

  # Development mode with shell
  claude-dev:
    build:
      context: .
      dockerfile: docker/claude-sdk/Dockerfile
    environment:
      <<: *common-env
    volumes:
      - ./my-agent/config:/app/config:ro
      - ./my-agent/tools:/app/tools:ro
      - ./my-agent/scripts:/app/scripts:ro
      - ./my-agent/data:/app/data:rw
    command: ["shell"]
    stdin_open: true
    tty: true
```

---

## User Directory Structure

Users would create a directory like:

```
my-agent/
├── config/
│   ├── agent.yaml          # Main configuration
│   └── .env                 # Secrets (optional, gitignored)
├── tools/
│   ├── database_tools.py   # Custom database tools
│   ├── api_tools.py        # Custom API tools
│   └── file_tools.py       # Custom file tools
├── scripts/
│   └── my_custom_agent.py  # Custom agent script (optional)
└── data/
    ├── my_database.db      # Working data (read-write)
    └── outputs/            # Generated files
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `THENVOI_AGENT_ID` | Agent ID (override config) | - |
| `THENVOI_API_KEY` | API key (override config) | - |
| `THENVOI_REST_API_URL` | REST API URL | `https://api.thenvoi.com` |
| `THENVOI_WS_URL` | WebSocket URL | `wss://api.thenvoi.com/ws` |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `LOG_LEVEL` | Logging level | `INFO` |
| `CONFIG_PATH` | Config file path | `/app/config/agent.yaml` |
| `TOOLS_PATH` | Tools directory | `/app/tools` |
| `DATA_PATH` | Data directory | `/app/data` |

**Priority**: Config file values take precedence over environment variables.

---

## Usage Examples

### 1. Basic Usage
```bash
# Build the image
docker build -t thenvoi-claude-sdk -f docker/claude-sdk/Dockerfile .

# Run with mounted config
docker run -v ./my-agent/config:/app/config:ro \
           -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
           thenvoi-claude-sdk
```

### 2. With Custom Tools
```bash
docker run -v ./my-agent/config:/app/config:ro \
           -v ./my-agent/tools:/app/tools:ro \
           -v ./my-agent/data:/app/data:rw \
           -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
           thenvoi-claude-sdk
```

### 3. With Example Tools (Testing)
```bash
docker run -v ./my-agent/config:/app/config:ro \
           -v ./docker/claude-sdk/example-tools:/app/tools:ro \
           -v ./my-agent/data:/app/data:rw \
           -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
           thenvoi-claude-sdk
```

### 4. Run Custom Script
```bash
docker run -v ./my-agent/config:/app/config:ro \
           -v ./my-agent/scripts:/app/scripts:ro \
           -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
           thenvoi-claude-sdk custom my_agent.py
```

### 5. Interactive Development
```bash
docker run -it \
           -v ./my-agent:/app/my-agent:rw \
           -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
           thenvoi-claude-sdk shell
```

---

## Implementation Notes

### MCP Server Integration

The current design assumes custom tools can be added as a separate MCP server. 

**TODO**: Verify with `claude-agent-sdk` documentation:
1. Does it support multiple MCP servers? (e.g., `mcp_servers={"thenvoi": ..., "custom": ...}`)
2. If not, modify `ClaudeSDKAdapter._create_mcp_server()` to accept additional tools

If multiple servers aren't supported, the alternative is to:
- Fork or extend `ClaudeSDKAdapter`
- Override `_create_mcp_server()` to create a single server with both Thenvoi and custom tools

### What's Needed to Implement

1. **Create `docker/claude-sdk/` directory** with:
   - `Dockerfile`
   - `entrypoint.sh`
   - `runner.py`
   - `example-tools/echo_tools.py`

2. **Verify/add `claude_sdk` extra** in `pyproject.toml`:
   ```toml
   [project.optional-dependencies]
   claude_sdk = ["claude-agent-sdk", "pyyaml"]
   ```

3. **Test MCP integration** - verify custom tools work with Claude SDK

4. **Test scenarios**:
   - Basic config-only run
   - Custom tools loading and execution
   - Custom script execution
   - Data volume read/write
