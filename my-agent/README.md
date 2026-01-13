# My Agent - Claude SDK Docker Example

This directory contains the configuration and data for running a Claude SDK agent in Docker.

## Directory Structure

```
my-agent/
├── config/
│   ├── agent.yaml       # Main agent configuration
│   ├── env.example      # Example environment variables (copy to .env)
│   └── .env             # Your environment variables (gitignored)
├── tools/
│   └── *.py             # Your custom tool Python files
├── scripts/
│   └── *.py             # Your custom agent scripts
└── data/
    └── ...              # Working data (read-write volume)
```

## Quick Start

1. **Configure credentials**:
   ```bash
   # Edit agent.yaml with your Thenvoi credentials
   vim config/agent.yaml
   
   # Or use environment variables
   cp config/env.example config/.env
   vim config/.env
   ```

2. **Run the agent**:
   ```bash
   # Basic run (no custom tools)
   docker compose -f docker-compose.claude-sdk.yml up claude-basic
   
   # With example tools
   docker compose -f docker-compose.claude-sdk.yml up claude-example
   
   # With your custom tools
   docker compose -f docker-compose.claude-sdk.yml up claude-with-tools
   ```

3. **Add custom tools**:
   - Copy `docker/claude-sdk/example-tools/echo_tools.py` to `tools/`
   - Modify or create new tool files following the same pattern
   - Each file must have a `TOOLS` list with tool definitions

## Custom Tools

Create Python files in `tools/` with this structure:

```python
from typing import Any

TOOLS = []

def register_tool(name: str, description: str, parameters: dict):
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
    name="my_tool",
    description="Description for Claude",
    parameters={"arg": {"type": "string", "description": "Argument description"}}
)
async def my_tool(args: dict[str, Any]) -> dict[str, Any]:
    return {"status": "success", "result": args.get("arg")}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `THENVOI_AGENT_ID` | Agent ID (override config) | - |
| `THENVOI_API_KEY` | API key (override config) | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `LOG_LEVEL` | Logging level | `INFO` |
