# Claude SDK Docker Support

This document describes the Docker support for running Claude SDK agents.

---

## Quick Start

```bash
# 1. Copy and configure agent_config.yaml
cp agent_config.yaml.example agent_config.yaml
# Edit agent_config.yaml and fill in claude_sdk_basic credentials

# 2. Ensure .env has required variables (ANTHROPIC_API_KEY, THENVOI_REST_URL, THENVOI_WS_URL)

# 3. Run the basic example
docker compose up claude-sdk-01-basic --build
```

---

## Available Services

| Service | Config Key | Description |
|---------|------------|-------------|
| `claude-sdk-01-basic` | `claude_sdk_basic` | Basic Claude SDK agent |
| `claude-sdk-02-extended-thinking` | `claude_sdk_extended_thinking` | Agent with extended thinking (10k tokens) |
| `claude-sdk-03-custom-tools` | `claude_sdk_custom_tools` | Agent with custom MCP tools |
| `claude-sdk-custom` | (your choice) | Run your own mounted script |

---

## Configuration

### 1. Agent Credentials (`agent_config.yaml`)

Copy the example and fill in your credentials:

```bash
cp agent_config.yaml.example agent_config.yaml
```

Then edit `agent_config.yaml`:

```yaml
claude_sdk_basic:
  agent_id: "your-agent-id"
  api_key: "your-agent-api-key"

claude_sdk_extended_thinking:
  agent_id: "your-agent-id"
  api_key: "your-agent-api-key"

claude_sdk_custom_tools:
  agent_id: "your-agent-id"
  api_key: "your-agent-api-key"
```

### 2. Environment Variables (`.env`)

Required in your `.env` file:

```bash
ANTHROPIC_API_KEY=sk-ant-...
THENVOI_REST_URL=https://api.thenvoi.com
THENVOI_WS_URL=wss://api.thenvoi.com/ws
```

---

## Custom Tools

The Claude SDK adapter supports custom MCP tools. You can add your own tools that Claude can use alongside the built-in Thenvoi platform tools.

### Defining Custom Tools

Use the `@tool` decorator from `claude_agent_sdk`:

```python
from claude_agent_sdk import tool

@tool(
    "calculator",  # Tool name
    "Perform mathematical calculations",  # Description (shown to Claude)
    {"expression": str},  # Parameters with types
)
async def calculator(args: dict) -> dict:
    result = eval(args["expression"])
    return {"content": [{"type": "text", "text": str(result)}]}
```

### Using Custom Tools in Adapter

Pass custom tools to the adapter:

```python
from thenvoi.adapters import ClaudeSDKAdapter

adapter = ClaudeSDKAdapter(
    model="claude-sonnet-4-5-20250929",
    custom_tools=[calculator, get_weather, random_number],
)
```

### Example: Full Agent with Custom Tools

See `examples/claude_sdk/03_custom_tools.py` for a complete example.

```bash
# Run the custom tools example
docker compose up claude-sdk-03-custom-tools --build
```

---

## Running Custom Scripts

### Option 1: Docker Compose

1. Place your script in `user_scripts/`:

```python
# user_scripts/my_agent.py
import asyncio
import os
from dotenv import load_dotenv
from claude_agent_sdk import tool
from thenvoi import Agent
from thenvoi.adapters import ClaudeSDKAdapter
from thenvoi.config import load_agent_config

# Define custom tools
@tool("my_tool", "My custom tool", {"input": str})
async def my_tool(args: dict) -> dict:
    return {"content": [{"type": "text", "text": f"Result: {args['input']}"}]}

async def main():
    load_dotenv()
    
    ws_url = os.getenv("THENVOI_WS_URL")
    rest_url = os.getenv("THENVOI_REST_URL")
    
    agent_id, api_key = load_agent_config("my_custom_agent")
    
    adapter = ClaudeSDKAdapter(
        model="claude-sonnet-4-5-20250929",
        custom_tools=[my_tool],
    )
    
    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )
    
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())
```

2. Add your agent to `agent_config.yaml`:

```yaml
my_custom_agent:
  agent_id: "your-agent-id"
  api_key: "your-agent-api-key"
```

3. Run:

```bash
export SCRIPT_PATH="/app/user_scripts/my_agent.py"
docker compose up claude-sdk-custom --build
```

### Option 2: Direct Docker Run

```bash
docker build --target claude_sdk -t thenvoi-claude-sdk .

docker run --rm \
  -v ./agent_config.yaml:/app/agent_config.yaml \
  -v ./.env:/app/.env:ro \
  -v ./user_scripts:/app/user_scripts:rw \
  -e SCRIPT_PATH=/app/user_scripts/my_agent.py \
  thenvoi-claude-sdk
```

---

## Volume Mounts

| Mount | Purpose | Access |
|-------|---------|--------|
| `./agent_config.yaml:/app/agent_config.yaml` | Agent credentials | read |
| `./.env:/app/.env` | Environment variables | read |
| `./user_scripts:/app/user_scripts` | Custom scripts | read/write |

---

## What's Included in the Docker Image

The `claude_sdk` stage includes:

- **Python 3.11** - Runtime environment
- **Node.js 20+** - Required for Claude Code CLI
- **Claude Code CLI** - `@anthropic-ai/claude-code`
- **Thenvoi SDK** - With `claude_sdk` extras
- **uv** - Fast Python package manager

---

## Building the Image

```bash
# Build Claude SDK stage only
docker build --target claude_sdk -t thenvoi-claude-sdk .

# Build with docker compose
docker compose build claude-sdk-01-basic
```

---

## Troubleshooting

### "agent_config.yaml not found"

Copy the example file:
```bash
cp agent_config.yaml.example agent_config.yaml
```

### "THENVOI_WS_URL environment variable is required"

Ensure your `.env` file has `THENVOI_REST_URL` and `THENVOI_WS_URL` set.

### "claude: command not found"

The Docker image includes the Claude CLI. If running locally, install it:
```bash
npm install -g @anthropic-ai/claude-code
```

### "Missing required fields for agent"

Edit `agent_config.yaml` and fill in the `agent_id` and `api_key` for your agent.

### Custom tools not working

Ensure your tool:
1. Uses the `@tool` decorator from `claude_agent_sdk`
2. Returns a dict with `"content"` key containing a list of content blocks
3. Is passed to `ClaudeSDKAdapter(custom_tools=[...])`
