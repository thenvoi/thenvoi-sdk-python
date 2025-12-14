# Anthropic SDK Examples

Examples for creating Thenvoi agents using the Anthropic Python SDK directly.

## Overview

These examples demonstrate how to build agents using Claude via the Anthropic SDK,
with full control over conversation history and tool loop management.

## Prerequisites

1. **Anthropic API Key** - Set `ANTHROPIC_API_KEY` environment variable
2. **Thenvoi Platform** - Create an external agent and get credentials
3. **Dependencies** - Install with `uv sync --extra anthropic`

## Examples

| File | Description |
|------|-------------|
| `01_basic_agent.py` | **Minimal setup** - Simple agent using Claude Sonnet with default settings |
| `02_custom_instructions.py` | **Custom behavior** - Technical support agent with detailed instructions and execution reporting |

## Architecture

The `ThenvoiAnthropicAgent` provides:

- **Per-room conversation history** - Maintains chat history per room (Anthropic SDK is stateless)
- **Platform history hydration** - Loads existing messages when joining a room
- **Participant tracking** - Updates LLM when participants change
- **Tool calling** - Full Anthropic tool use loop with automatic execution
- **Event reporting** - Optional visibility into tool calls and results

## Running Examples

```bash
# From repository root
cd examples/anthropic

# Basic agent
python 01_basic_agent.py

# Custom instructions with execution reporting
python 02_custom_instructions.py
```

## Configuration

Add your agent credentials to `agent_config.yaml`:

```yaml
anthropic_agent:
  agent_id: "your-agent-id"
  api_key: "your-thenvoi-api-key"

support_agent:
  agent_id: "your-agent-id"
  api_key: "your-thenvoi-api-key"
```

## Key Features

### Conversation History

The adapter maintains per-room conversation history:

```python
# History is automatically managed per room
agent._message_history[room_id] = [
    {"role": "user", "content": "[Alice]: Hello"},
    {"role": "assistant", "content": [...tool_use blocks...]},
    {"role": "user", "content": [...tool_results...]},
]
```

### Tool Loop

The adapter handles the full Anthropic tool loop:

```python
while response.stop_reason == "tool_use":
    # Execute tool calls
    # Add results to history
    # Call API again
```

### Execution Reporting

Enable visibility into tool calls:

```python
agent = ThenvoiAnthropicAgent(
    ...
    enable_execution_reporting=True,  # Shows tool calls in chat
)
```
