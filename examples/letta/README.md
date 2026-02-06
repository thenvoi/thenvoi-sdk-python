# Letta Adapter Examples

This directory contains examples demonstrating the Letta adapter for Thenvoi.

## Prerequisites

### 1. Install Dependencies

```bash
# Install thenvoi-sdk with Letta support
uv sync --extra letta

# Or install from git
uv add "git+https://github.com/thenvoi/thenvoi-sdk-python.git[letta]"
```

### 2. Start Letta Server

For local development, run Letta in Docker with your LLM API keys:

```bash
# Create .env file with your API keys
cat > .env << EOF
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
EOF

# Run Letta with your credentials
docker run -p 8283:8283 --env-file .env letta/letta:latest
```

Or use [Letta Cloud](https://app.letta.com) and set your API key.

### 3. Set Environment Variables

```bash
export THENVOI_AGENT_ID="your-thenvoi-agent-id"
export THENVOI_API_KEY="your-thenvoi-api-key"
export LETTA_BASE_URL="http://localhost:8283"  # or https://api.letta.com
export LETTA_API_KEY="sk-let-..."  # Only needed for Letta Cloud
```

## Examples

### 01 - Basic Per-Room Mode

**File:** `01_per_room_basic.py`

Each Thenvoi room gets its own dedicated Letta agent with isolated memory and conversation history.

```bash
uv run python examples/letta/01_per_room_basic.py
```

**Use case:** Business agents, data analysis, sensitive contexts where conversations should be isolated.

**Architecture:**
```
Room A → Letta Agent A (isolated memory)
Room B → Letta Agent B (isolated memory)
Room C → Letta Agent C (isolated memory)
```

### 02 - Personal Agent (Shared Mode)

**File:** `02_shared_personal.py`

Single Letta agent across all rooms with shared memory. Uses Conversations API for thread-safe parallel room handling.

```bash
uv run python examples/letta/02_shared_personal.py
```

**Use case:** Personal assistants that remember you across different contexts.

**Architecture:**
```
Room A → Conversation 1 ──┐
Room B → Conversation 2 ──┼─→ Single Agent (shared memory)
Room C → Conversation 3 ──┘
```

### 03 - Custom Tools

**File:** `03_custom_tools.py`

Adding custom tools to Letta agents using the `CustomToolBuilder`.

```bash
uv run python examples/letta/03_custom_tools.py
```

**Included tools:**
- `calculate`: Math operations
- `get_current_time`: Current timestamp
- `convert_temperature`: Temperature unit conversion
- `generate_random_number`: Random number generation

### 04 - Data Analysis Agent

**File:** `04_data_agent.py`

A specialized agent for data analysis with custom tools for querying and analyzing datasets.

```bash
uv run python examples/letta/04_data_agent.py
```

**Included tools:**
- `list_datasets`: Discover available data
- `get_dataset`: Retrieve full dataset
- `analyze_numeric_field`: Statistical analysis
- `filter_dataset`: Filter by conditions
- `aggregate_by_field`: Group and sum data

## Configuration Options

### LettaConfig

| Parameter | Description | Default |
|-----------|-------------|---------|
| `api_key` | Letta API key (optional for self-hosted) | `None` |
| `mode` | `LettaMode.PER_ROOM` or `LettaMode.SHARED` | `PER_ROOM` |
| `base_url` | Letta server URL | `https://api.letta.com` |
| `model` | LLM model (format: `provider/model-name`) | `openai/gpt-4o` |
| `embedding_model` | Embedding model (format: `provider/model-name`) | `openai/text-embedding-3-small` |
| `persona` | Agent persona/personality | None |
| `custom_tools` | List of custom tool definitions | `[]` |
| `api_timeout` | API call timeout (seconds) | `30` |

**Model format examples:**
- OpenAI: `openai/gpt-4o`, `openai/gpt-4o-mini`
- Anthropic: `anthropic/claude-3-5-sonnet-20241022`

### SessionConfig

```python
SessionConfig(
    enable_context_hydration=False,  # IMPORTANT: Always False for Letta
)
```

Letta manages its own conversation history server-side, so Thenvoi's context hydration must be disabled.

## State Persistence

The adapter persists state (room→agent/conversation mappings) to a JSON file:

- Default: `~/.thenvoi/letta_adapter_state.json`
- Can be customized via `state_storage_path` parameter

This enables:
- Reusing existing Letta agents on reconnection
- Resuming conversations after restarts
- Tracking room activity and metadata

## Memory Management

### Per-Room Mode

Each agent has:
- `persona`: Agent personality and role
- `participants`: Current room participants

### Shared Mode

The shared agent has:
- `persona`: Agent personality (consistent across rooms)
- `participants`: Current room participants (updated on room entry)
- `room_contexts`: Per-room notes and summaries

## Troubleshooting

### Connection Errors

```
LettaConnectionError: Failed to connect to Letta
```

Ensure the Letta server is running:
```bash
curl http://localhost:8283/v1/health
```

### Agent Not Found

```
LettaAgentNotFoundError: Letta agent not found
```

The persisted agent may have been deleted. Delete the state file to create a new agent:
```bash
rm ~/.thenvoi/letta_*_state.json
```

### Timeout Errors

```
LettaTimeoutError: Letta operation 'send_message' timed out
```

Increase the timeout in config:
```python
LettaConfig(
    api_timeout=60,  # Increase from default 30s
)
```
