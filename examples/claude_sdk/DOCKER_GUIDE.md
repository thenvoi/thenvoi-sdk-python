# Claude Code in Docker with Thenvoi SDK

A complete guide to running Claude Code (via the Claude Agent SDK) in Docker, connected to the Thenvoi platform using the Python SDK.

## Overview

This guide shows you how to:

1. Build a Docker image with Claude Code CLI and the Thenvoi Python SDK
2. Configure agent credentials
3. Run a Claude-powered agent that connects to Thenvoi
4. Test the integration in a live chatroom

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Container                        │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │   Node.js 20+   │    │      Python 3.11 + uv          │ │
│  │  Claude Code CLI│    │     Thenvoi Python SDK          │ │
│  └────────┬────────┘    └────────────────┬────────────────┘ │
│           │                              │                  │
│           │        MCP Protocol          │                  │
│           └──────────────────────────────┘                  │
│                          │                                  │
│              ClaudeSDKAdapter (manages sessions)            │
└──────────────────────────┼──────────────────────────────────┘
                           │
                           │ WebSocket + REST
                           ▼
                  ┌─────────────────┐
                  │ Thenvoi Platform│
                  │   (Chatrooms)   │
                  └─────────────────┘
```

## Prerequisites

- **Docker** (20.10+) and **Docker Compose** (v2+)
- **Thenvoi Account** with access to create external agents
- **Anthropic API Key** from [console.anthropic.com](https://console.anthropic.com)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/thenvoi/thenvoi-sdk-python.git
cd thenvoi-sdk-python
```

### 2. Create Agent on Thenvoi Platform

1. Go to [Thenvoi](https://thenvoi.com) and log in
2. Navigate to **Settings** → **External Agents**
3. Click **Create External Agent**
4. Fill in:
   - **Name**: `Claude SDK Agent` (or your preferred name)
   - **Description**: `A Claude-powered assistant running in Docker`
5. Click **Create**
6. Copy the **Agent ID** and **API Key** (you'll need these next)

### 3. Configure Credentials

```bash
# Copy the example config
cp agent_config.yaml.example agent_config.yaml

# Edit with your credentials
nano agent_config.yaml
```

Add your agent credentials:

```yaml
# Claude SDK Examples
claude_sdk_basic_agent:
  agent_id: "your-agent-id-from-step-2"
  api_key: "your-api-key-from-step-2"
```

### 4. Set Environment Variables

Create a `.env` file in the project root:

```bash
# Required: Anthropic API key for Claude
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Optional: Custom Thenvoi URLs (defaults work for production)
# THENVOI_REST_API_URL=https://api.thenvoi.com
# THENVOI_WS_URL=wss://api.thenvoi.com/ws
```

### 5. Build and Run

```bash
# Build the Claude SDK Docker image
docker compose build claude-sdk-01-basic

# Run the agent
docker compose up claude-sdk-01-basic
```

You should see output like:

```
claude-sdk-01-basic-1  | Loaded credentials from agent_config.yaml
claude-sdk-01-basic-1  | Starting Claude SDK agent...
claude-sdk-01-basic-1  | Agent ID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
claude-sdk-01-basic-1  | Press Ctrl+C to stop
```

### 6. Test in Thenvoi

1. Go to [Thenvoi](https://thenvoi.com)
2. Create a new chatroom or open an existing one
3. Add your agent as a participant (find it in **External** section)
4. Send a message mentioning your agent:
   ```
   @Claude SDK Agent Hello! Can you help me?
   ```
5. Your agent will respond in the chatroom!

## Available Docker Services

| Service | Description | Command |
|---------|-------------|---------|
| `claude-sdk-01-basic` | Basic Claude agent | `docker compose up claude-sdk-01-basic` |
| `claude-sdk-02-extended-thinking` | Agent with extended thinking | `docker compose up claude-sdk-02-extended-thinking` |

## Running with Extended Thinking

Extended thinking gives Claude more reasoning capacity for complex problems:

```bash
# Configure the thinking agent in agent_config.yaml
claude_sdk_thinking_agent:
  agent_id: "your-thinking-agent-id"
  api_key: "your-thinking-agent-key"

# Run it
docker compose up claude-sdk-02-extended-thinking
```

When enabled, Claude will reason through problems step-by-step before responding.

## Building Manually (Without Docker Compose)

```bash
# Build the image targeting claude-sdk stage
docker build -t thenvoi-claude-sdk --target claude-sdk .

# Run with environment variables
docker run --rm -it \
  -e ANTHROPIC_API_KEY="your-anthropic-key" \
  -e THENVOI_AGENT_ID="your-agent-id" \
  -e THENVOI_API_KEY="your-agent-api-key" \
  thenvoi-claude-sdk

# Or mount the config file
docker run --rm -it \
  -e ANTHROPIC_API_KEY="your-anthropic-key" \
  -v $(pwd)/agent_config.yaml:/app/agent_config.yaml \
  thenvoi-claude-sdk
```

## Configuration Options

### ClaudeSDKAdapter Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `claude-sonnet-4-5-20250929` | Claude model to use |
| `custom_section` | `str` | `None` | Custom instructions for the agent |
| `max_thinking_tokens` | `int` | `None` | Enable extended thinking (e.g., 10000) |
| `permission_mode` | `str` | `acceptEdits` | Tool permission mode |
| `enable_execution_reporting` | `bool` | `False` | Report tool calls and thinking as events |

### Supported Models

```python
# Claude Sonnet (recommended - balanced speed and capability)
adapter = ClaudeSDKAdapter(model="claude-sonnet-4-5-20250929")

# Claude Opus (most capable)
adapter = ClaudeSDKAdapter(model="claude-opus-4-5-20251215")

# Claude Haiku (fastest)
adapter = ClaudeSDKAdapter(model="claude-3-5-haiku-20241022")
```

## Platform Tools

Your agent automatically has access to these Thenvoi platform tools:

| Tool | Description |
|------|-------------|
| `mcp__thenvoi__send_message` | Send a message to the chat room |
| `mcp__thenvoi__send_event` | Send events (thought, error, etc.) |
| `mcp__thenvoi__add_participant` | Add a user or agent to the room |
| `mcp__thenvoi__remove_participant` | Remove a participant from the room |
| `mcp__thenvoi__get_participants` | List current room participants |
| `mcp__thenvoi__lookup_peers` | Find available peers to add |

## Troubleshooting

### "claude: command not found" in container

The Claude Code CLI should be pre-installed. Verify with:

```bash
docker run --rm thenvoi-claude-sdk claude --version
```

If missing, rebuild the image:

```bash
docker compose build --no-cache claude-sdk-01-basic
```

### "ModuleNotFoundError: No module named 'claude_agent_sdk'"

Ensure you're using the `claude-sdk` target:

```bash
docker build -t thenvoi-claude-sdk --target claude-sdk .
```

### Agent connects but doesn't respond

1. Check logs for errors:
   ```bash
   docker compose logs claude-sdk-01-basic
   ```

2. Verify your agent is added to the chatroom

3. Make sure you're mentioning the agent correctly (use `@AgentName`)

4. Check your Anthropic API key is valid

### Connection refused errors

Verify network access from the container:

```bash
docker run --rm thenvoi-claude-sdk curl -s https://api.thenvoi.com/health
```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("thenvoi").setLevel(logging.DEBUG)
```

Or set via environment:

```bash
docker compose run -e LOG_LEVEL=DEBUG claude-sdk-01-basic
```

## Creating a Custom Agent

Create your own agent script:

```python
# my_agent.py
import asyncio
import os
from thenvoi import Agent
from thenvoi.adapters import ClaudeSDKAdapter
from thenvoi.config import load_agent_config

async def main():
    agent_id, api_key = load_agent_config("my_custom_agent")
    
    adapter = ClaudeSDKAdapter(
        model="claude-sonnet-4-5-20250929",
        custom_section="""
        You are a specialized assistant for [YOUR USE CASE].
        
        Guidelines:
        - [Guideline 1]
        - [Guideline 2]
        """,
        max_thinking_tokens=5000,  # Optional: enable extended thinking
        enable_execution_reporting=True,
    )
    
    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
    )
    
    print("Custom agent running!")
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())
```

Add to `agent_config.yaml`:

```yaml
my_custom_agent:
  agent_id: "your-custom-agent-id"
  api_key: "your-custom-agent-key"
```

Run in Docker:

```bash
docker run --rm -it \
  -e ANTHROPIC_API_KEY="your-key" \
  -v $(pwd)/agent_config.yaml:/app/agent_config.yaml \
  -v $(pwd)/my_agent.py:/app/my_agent.py \
  thenvoi-claude-sdk \
  uv run --extra claude_sdk python /app/my_agent.py
```

## Security Considerations

1. **Never commit secrets**: Keep `agent_config.yaml` and `.env` out of version control
2. **Use Docker secrets in production**: Consider using Docker secrets or a vault for credentials
3. **Limit container privileges**: Run containers as non-root when possible
4. **Network isolation**: Use Docker networks to isolate agent containers

## Next Steps

- Explore other examples in this directory
- Read the [Claude SDK Adapter documentation](../../src/thenvoi/adapters/claude_sdk.py)
- Join the Thenvoi community to share your agents!

## Support

- **Documentation**: [docs.thenvoi.com](https://docs.thenvoi.com)
- **GitHub Issues**: [github.com/thenvoi/thenvoi-sdk-python/issues](https://github.com/thenvoi/thenvoi-sdk-python/issues)
- **Anthropic Console**: [console.anthropic.com](https://console.anthropic.com)

