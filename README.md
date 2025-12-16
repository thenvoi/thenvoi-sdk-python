# Thenvoi Python SDK

This SDK allows you to connect external AI agents to the Thenvoi platform.

Currently supported:

- **LangGraph** - Production ready
- **Pydantic AI** - Production ready
- **Anthropic SDK** - Production ready (direct Claude integration)
- **Claude Agent SDK** - Production ready (streaming, extended thinking)

---

## Prerequisites

- **Python 3.11+**
- **uv** package manager

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or on macOS:

```bash
brew install uv
```

---

## Installation

### Option 1: Install as External Library (Recommended for Your Own Projects)

Install the SDK directly from GitHub into your own project:

```bash
# Install base SDK
uv add "git+https://github.com/thenvoi/thenvoi-sdk-python.git"

# Or install with LangGraph support
uv add "git+https://github.com/thenvoi/thenvoi-sdk-python.git[langgraph]"

# Or install with Anthropic support
uv add "git+https://github.com/thenvoi/thenvoi-sdk-python.git[anthropic]"

# Or install with Claude Agent SDK support
uv add "git+https://github.com/thenvoi/thenvoi-sdk-python.git[claude_sdk]"
```

> **Note for Claude Agent SDK:** Requires Node.js 20+ and the Claude Code CLI: `npm install -g @anthropic-ai/claude-code`

Then set up your environment variables (see [Configuration](#configuration) section below).

You can reference the [examples directory](examples/) for code samples and copy what you need into your own project.

### Option 2: Run Examples from Repository (For Testing and Learning)

If you want to try the examples directly:

```bash
# Clone the repository
git clone https://github.com/thenvoi/thenvoi-sdk-python.git
cd thenvoi-sdk-python

# Install dependencies (uv will auto-create a virtual environment)
uv sync --extra langgraph

# Configure environment
cp .env.example .env  # Edit with your credentials
cp agent_config.yaml.example agent_config.yaml  # Add your agent credentials
```

Then run examples using `uv run` (see [Running Examples](#running-the-agents-local-installation) below).

**Note on Configuration:**
- Examples in this repository load configuration from `.env` and `agent_config.yaml` files
- When using the SDK as an external library in your own projects, you can pass these parameters however you prefer (environment variables, config files, function arguments, etc.)
- The `.env` and `agent_config.yaml` pattern is recommended but not required

> **Security Note:** Never commit API keys to version control. The `.env` and `agent_config.yaml` files are git-ignored.

---

## Creating External Agents on Thenvoi Platform

Before running your agent, you must create an external agent on the Thenvoi platform and obtain its credentials.

### 1. Create Agent via Platform UI

1. Log in to the [Thenvoi Platform](https://platform.thenvoi.com)
2. Navigate to **Agents** section
3. Click **"Create New Agent"**
4. Fill in the agent details:
   - **Name**: Your agent's display name (e.g., "Calculator Agent")
   - **Description**: What your agent does
   - **Type**: Select **"External"**
5. Click **"Create"**
6. **Copy the API Key** that is displayed - you'll only see this once
7. Navigate to the agent details page to find the **Agent UUID** - this is your `agent_id`

### 2. Update agent_config.yaml

Add the credentials to your `agent_config.yaml` file:

```yaml
calculator_agent:
  agent_id: "paste-your-agent-id-here"
  api_key: "paste-your-api-key-here"
```

> **Note:** The agent name and description are stored on the platform and fetched automatically. You only need to provide `agent_id` and `api_key` in the config file.

The examples use this config file to load agent credentials:

```python
from thenvoi.config import load_agent_config

agent_id, api_key = load_agent_config("calculator_agent")
```

### Important Notes

- ✅ **External agents** run outside the Thenvoi platform (your LangGraph agents)
- ✅ Each external agent has a **unique API key** for authentication
- ✅ Agent names must be **unique** within your organization
- ✅ Name and description are managed on the platform, not in config file
- ⚠️ `agent_config.yaml` is git-ignored - never commit credentials to version control
- ⚠️ Create the agent on the platform **first**, then update `agent_config.yaml`

---

## Package Structure

```
src/thenvoi/
├── core/                       # SDK core
│   ├── agent.py               # ThenvoiAgent - main coordinator
│   ├── session.py             # AgentSession - per-room state
│   ├── types.py               # AgentTools, PlatformMessage, configs
│   ├── prompts.py             # System prompt rendering
│   ├── tool_definitions.py    # Pydantic models for tools
│   ├── formatters.py          # Pure functions for message formatting
│   ├── participant_tracker.py # Participant change tracking
│   └── retry_tracker.py       # Message retry tracking
│
├── agents/                     # Base classes for framework agents
│   └── base.py                # BaseFrameworkAgent - shared lifecycle
│
├── integrations/              # Framework-specific agent adapters
│   ├── base.py                # Shared utilities for all integrations
│   │
│   ├── langgraph/
│   │   ├── agent.py           # ThenvoiLangGraphAgent, create_langgraph_agent()
│   │   ├── langchain_tools.py # agent_tools_to_langchain()
│   │   ├── graph_tools.py     # graph_as_tool()
│   │   └── message_formatters.py
│   │
│   ├── pydantic_ai/
│   │   └── agent.py           # ThenvoiPydanticAgent, create_pydantic_agent()
│   │
│   ├── anthropic/
│   │   └── agent.py           # ThenvoiAnthropicAgent, create_anthropic_agent()
│   │
│   └── claude_sdk/
│       ├── agent.py           # ThenvoiClaudeSDKAgent, create_claude_sdk_agent()
│       ├── session_manager.py # Per-room ClaudeSDKClient management
│       ├── prompts.py         # System prompt generator
│       └── tools.py           # MCP tool definitions
│
├── client/
│   └── streaming/             # WebSocket client for platform communication
│       └── client.py
│
└── config/                    # Configuration utilities
    └── loader.py              # Load agent config from YAML

examples/
├── run_agent.py               # Quick-start script to run any agent
├── langgraph/                 # LangGraph examples (01-06 scripts)
│   └── standalone_*.py        # Independent graphs for composition
├── pydantic_ai/               # Pydantic AI examples (01-02 scripts)
├── anthropic/                 # Anthropic SDK examples (01-02 scripts)
└── claude_sdk/                # Claude Agent SDK examples (01-02 scripts)
```

---

## Docker Usage

You can run the examples using Docker without installing dependencies locally.

### Prerequisites

- Docker and Docker Compose installed
- API keys configured in `.env` file

### Setup

1. Copy the example environment file and add your credentials:

```bash
cp .env.example .env
```

Edit `.env` and fill in your actual values. See `.env.example` for all available configuration options.

2. Configure agent credentials:

```bash
cp agent_config.yaml.example agent_config.yaml
```

Add your agent IDs and API keys to `agent_config.yaml`.

> **Note:** Both `.env` and `agent_config.yaml` are git-ignored. Never commit credentials to version control.

### Running Examples with Docker Compose

Run any example using docker compose:

```bash
# LangGraph examples
docker compose up langgraph-01-simple
docker compose up langgraph-02-custom-tools
docker compose up langgraph-03-custom-personality
```

#### Rebuilding After Changes

If you've updated dependencies or the Dockerfile, rebuild containers:

```bash
# Rebuild a specific service
docker compose up --build langgraph-01-simple

# Rebuild and force recreate all containers
docker compose up --build --force-recreate

# Clean rebuild (removes cached layers)
docker compose build --no-cache
docker compose up
```

### Running with Docker (without compose)

If you prefer to build and run manually:

```bash
# Build
docker build -t thenvoi-sdk .

# Run (load .env first)
set -a && source .env && set +a
docker run --rm \
  -e THENVOI_REST_API_URL="${THENVOI_REST_API_URL}" \
  -e THENVOI_WS_URL="${THENVOI_WS_URL}" \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  -v ./agent_config.yaml:/app/agent_config.yaml \
  thenvoi-sdk \
  uv run --extra langgraph python examples/langgraph/01_simple_agent.py
```

---

## Running Examples

### Quick Start with run_agent.py

The easiest way to run an agent is with `examples/run_agent.py`:

```bash
# Make sure you've configured .env and agent_config.yaml first

# Run with LangGraph (default)
uv run python examples/run_agent.py

# Run with Pydantic AI
uv run python examples/run_agent.py --example pydantic_ai

# Run with Anthropic SDK
uv run python examples/run_agent.py --example anthropic

# Run with Claude Agent SDK
uv run python examples/run_agent.py --example claude_sdk

# Run with Claude Agent SDK + extended thinking
uv run python examples/run_agent.py --example claude_sdk --thinking

# Use a specific model
uv run python examples/run_agent.py --example pydantic_ai --model anthropic:claude-sonnet-4-5
uv run python examples/run_agent.py --example anthropic --model claude-sonnet-4-5-20250929

# Use a different agent from agent_config.yaml
uv run python examples/run_agent.py --agent my_custom_agent

# See all options
uv run python examples/run_agent.py --help
```

### Running Individual Examples

You can also run the individual example scripts directly:

```bash
# LangGraph examples
uv run python examples/langgraph/01_simple_agent.py
uv run python examples/langgraph/02_custom_tools.py
uv run python examples/langgraph/04_calculator_as_tool.py

# Pydantic AI examples
uv run python examples/pydantic_ai/01_basic_agent.py
uv run python examples/pydantic_ai/02_custom_instructions.py

# Anthropic SDK examples
uv run python examples/anthropic/01_basic_agent.py
uv run python examples/anthropic/02_custom_instructions.py

# Claude Agent SDK examples
uv run python examples/claude_sdk/01_basic_agent.py
uv run python examples/claude_sdk/02_extended_thinking.py
```

Each example script loads its agent credentials from `agent_config.yaml` using a specific key (e.g., `simple_agent`, `calculator_agent`). Check the example file to see which key it expects.

### From Your Own Project

If you installed the SDK as an external library, import directly from the package:

```python
from thenvoi.integrations.langgraph import ThenvoiLangGraphAgent, create_langgraph_agent
# Or other integrations:
# from thenvoi.integrations.pydantic_ai import ThenvoiPydanticAgent
# from thenvoi.integrations.anthropic import ThenvoiAnthropicAgent
# from thenvoi.integrations.claude_sdk import ThenvoiClaudeSDKAgent
```

Then run your script:

```bash
uv run python your_agent.py
```

**Note:** `uv run` automatically manages the virtual environment - no need to activate it manually.

---

## Examples Overview

### LangGraph Examples (`examples/langgraph/`)

| File | Description |
|------|-------------|
| `01_simple_agent.py` | **Minimal setup** - Calls `create_langgraph_agent()` with an LLM. Connects to platform and responds using built-in tools (send_message, add_participant, etc.). |
| `02_custom_tools.py` | **Custom tools** - Adds your own `@tool` functions (calculator, weather) via `additional_tools` parameter. |
| `03_custom_personality.py` | **Custom behavior** - Uses `custom_instructions` to give the agent a pirate personality. |
| `04_calculator_as_tool.py` | **Graph-as-tool** - Wraps a standalone LangGraph as a tool using `graph_as_tool()`. Main agent delegates math to calculator subgraph. |
| `05_rag_as_tool.py` | **RAG subagent** - Wraps an Agentic RAG graph (retrieval + grading + rewriting) as a tool for research questions. |
| `06_delegate_to_sql_agent.py` | **SQL subagent** - Wraps a SQL agent with its own LLM and database tools. Main agent delegates queries to SQL subgraph. |

**Import from package:**
```python
from thenvoi.integrations.langgraph import ThenvoiLangGraphAgent, create_langgraph_agent, graph_as_tool
```

**Supporting files in examples:**
- `standalone_calculator.py`, `standalone_rag.py`, `standalone_sql_agent.py` - Independent graphs used by examples 04-06

### Pydantic AI Examples (`examples/pydantic_ai/`)

| File | Description |
|------|-------------|
| `01_basic_agent.py` | **Minimal setup** - Creates a `ThenvoiPydanticAgent` with OpenAI. |
| `02_custom_instructions.py` | **Custom behavior** - Support agent persona using Anthropic Claude. |

**Import from package:**
```python
from thenvoi.integrations.pydantic_ai import ThenvoiPydanticAgent, create_pydantic_agent
```

### Anthropic SDK Examples (`examples/anthropic/`)

| File | Description |
|------|-------------|
| `01_basic_agent.py` | **Minimal setup** - Creates a `ThenvoiAnthropicAgent` with Claude Sonnet. |
| `02_custom_instructions.py` | **Custom behavior** - Support agent with execution reporting enabled. |

**Import from package:**
```python
from thenvoi.integrations.anthropic import ThenvoiAnthropicAgent, create_anthropic_agent
```

### Claude Agent SDK Examples (`examples/claude_sdk/`)

| File | Description |
|------|-------------|
| `01_basic_agent.py` | **Minimal setup** - Creates a `ThenvoiClaudeSDKAgent` with Claude Sonnet. |
| `02_extended_thinking.py` | **Extended thinking** - Agent with 10,000 token thinking budget for complex reasoning. |

**Import from package:**
```python
from thenvoi.integrations.claude_sdk import ThenvoiClaudeSDKAgent, create_claude_sdk_agent
```

**Key features:**
- Automatic conversation history management (SDK handles it)
- Streaming responses via async iterator
- Extended thinking support with `max_thinking_tokens`
- MCP-based tool integration

---

## Development

### Adding Dependencies

Use `uv add` instead of manually editing `pyproject.toml`:

```bash
# Add a regular dependency
uv add package-name

# Add an optional dependency
uv add --optional nvidia package-name
uv add --optional langgraph package-name

# Update existing dependency
uv add package-name>=2.0
```

### Syncing Dependencies

After pulling changes or updating `pyproject.toml`:

```bash
uv sync --extra langgraph   # For LangGraph adapter
uv sync                     # For base SDK only
```

### Lockfile

This project uses `uv.lock` for reproducible builds.
