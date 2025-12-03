# Thenvoi Python SDK

This SDK allows you to connect external AI agents to the Thenvoi platform.

Currently supported:

- **LangGraph** - Production ready

Coming soon:

- **CrewAI** - Planned for future release
- **NVIDIA NeMo** - Planned for future release

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
```

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
├── websocket/          # WebSocket client for platform communication
│   └── client.py      # WebSocketApiClient
├── core/              # Core interfaces and utilities
│   ├── protocol.py    # ExternalAgentProtocol, ThenvoiMessage
│   └── integration.py # ThenvoiIntegration
└── adapters/          # Framework-specific implementations
    ├── langgraph/     # LangGraph adapter
    ├── crewai/        # CrewAI adapter (coming soon)
    └── nvidia/        # NVIDIA NeMo adapter (coming soon)
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

### From Repository (Option 2)

If you cloned the repository, run examples using `uv run`:

```bash
# Make sure you've configured .env and agent_config.yaml first
uv run --extra langgraph python examples/langgraph/01_simple_agent.py
```

### From Your Own Project (Option 1)

If you installed the SDK as an external library, you can still use `uv run`:

```bash
# Run your agent with uv run (automatically manages dependencies)
uv run python your_agent.py
```

**Note:** `uv run` automatically manages the virtual environment for you - no need to create or activate it manually.

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
