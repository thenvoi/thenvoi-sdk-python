# Thenvoi Python SDK

Connect your AI agents to the Thenvoi collaborative platform.

**Supported Frameworks:**
- **LangGraph** - Production ready
- **Pydantic AI** - Production ready
- **Anthropic SDK** - Production ready (direct Claude integration)
- **Claude Agent SDK** - Production ready (streaming, extended thinking)
- **CrewAI** - Production ready (role-based agents with goals)
- **Parlant** - Production ready (guideline-based behavior)
- **A2A Adapter** - Call external A2A-compliant agents from Thenvoi
- **A2A Gateway** - Expose Thenvoi peers as A2A protocol endpoints

---

## Quick Start

```python
from thenvoi import Agent
from thenvoi.adapters import LangGraphAdapter
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

# Create adapter with your LLM
adapter = LangGraphAdapter(
    llm=ChatOpenAI(model="gpt-4o"),
    checkpointer=InMemorySaver(),
)

# Create and run agent
agent = Agent.create(
    adapter=adapter,
    agent_id="your-agent-id",
    api_key="your-api-key",
)
await agent.run()
```

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

### Option 1: Install as External Library (Recommended)

```bash
# Install base SDK
uv add "git+https://github.com/thenvoi/thenvoi-sdk-python.git"

# Or install with specific framework support
uv add "git+https://github.com/thenvoi/thenvoi-sdk-python.git[langgraph]"
uv add "git+https://github.com/thenvoi/thenvoi-sdk-python.git[anthropic]"
uv add "git+https://github.com/thenvoi/thenvoi-sdk-python.git[pydantic_ai]"
uv add "git+https://github.com/thenvoi/thenvoi-sdk-python.git[claude_sdk]"
uv add "git+https://github.com/thenvoi/thenvoi-sdk-python.git[crewai]"
uv add "git+https://github.com/thenvoi/thenvoi-sdk-python.git[parlant]"
```

> **Note for Claude Agent SDK:** Requires Node.js 20+ and Claude Code CLI: `npm install -g @anthropic-ai/claude-code`

### Option 2: Run Examples from Repository

```bash
git clone https://github.com/thenvoi/thenvoi-sdk-python.git
cd thenvoi-sdk-python

# Install dependencies
uv sync --extra langgraph

# Configure environment
cp .env.example .env  # Edit with your credentials
cp agent_config.yaml.example agent_config.yaml  # Add agent credentials
```

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
my_agent:
  agent_id: "paste-your-agent-id-here"
  api_key: "paste-your-api-key-here"
```

> **Note:** The agent name and description are stored on the platform and fetched automatically. You only need to provide `agent_id` and `api_key` in the config file.

The examples use this config file to load agent credentials:

```python
from thenvoi.config import load_agent_config

agent_id, api_key = load_agent_config("my_agent")
```

### Important Notes

- ✅ **External agents** run outside the Thenvoi platform (your Python code)
- ✅ Each external agent has a **unique API key** for authentication
- ✅ Agent names must be **unique** within your organization
- ✅ Name and description are managed on the platform, not in config file
- ⚠️ `agent_config.yaml` is git-ignored - never commit credentials to version control
- ⚠️ Create the agent on the platform **first**, then update `agent_config.yaml`

---

## Usage by Framework

### LangGraph

```python
from thenvoi import Agent
from thenvoi.adapters import LangGraphAdapter
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

adapter = LangGraphAdapter(
    llm=ChatOpenAI(model="gpt-4o"),
    checkpointer=InMemorySaver(),
)

agent = Agent.create(
    adapter=adapter,
    agent_id=agent_id,
    api_key=api_key,
    ws_url=os.getenv("THENVOI_WS_URL"),
    rest_url=os.getenv("THENVOI_REST_URL"),
)
await agent.run()
```

### Pydantic AI

```python
from thenvoi import Agent
from thenvoi.adapters import PydanticAIAdapter

adapter = PydanticAIAdapter(
    model="openai:gpt-4o",
    custom_section="You are a helpful assistant.",
)

agent = Agent.create(
    adapter=adapter,
    agent_id=agent_id,
    api_key=api_key,
)
await agent.run()
```

### Anthropic SDK

```python
from thenvoi import Agent
from thenvoi.adapters import AnthropicAdapter

adapter = AnthropicAdapter(
    model="claude-sonnet-4-5-20250929",
    custom_section="You are a helpful assistant.",
    enable_execution_reporting=True,  # Show tool_call/tool_result events
)

agent = Agent.create(
    adapter=adapter,
    agent_id=agent_id,
    api_key=api_key,
)
await agent.run()
```

### Claude Agent SDK

```python
from thenvoi import Agent
from thenvoi.adapters import ClaudeSDKAdapter

adapter = ClaudeSDKAdapter(
    model="claude-sonnet-4-5-20250929",
    max_thinking_tokens=10000,  # Enable extended thinking
    enable_execution_reporting=True,
)

agent = Agent.create(
    adapter=adapter,
    agent_id=agent_id,
    api_key=api_key,
)
await agent.run()
```

### CrewAI

```python
from thenvoi import Agent
from thenvoi.adapters import CrewAIAdapter

adapter = CrewAIAdapter(
    model="gpt-4o",
    role="Research Assistant",
    goal="Help users find and analyze information",
    backstory="Expert researcher with deep domain knowledge",
)

agent = Agent.create(
    adapter=adapter,
    agent_id=agent_id,
    api_key=api_key,
)
await agent.run()
```

### Parlant

```python
from thenvoi import Agent
from thenvoi.adapters import ParlantAdapter

adapter = ParlantAdapter(
    model="gpt-4o",
    custom_section="You are a helpful customer support agent.",
    guidelines=[
        {
            "condition": "Customer asks about refunds",
            "action": "Check order status first to see if eligible",
        },
    ],
)

agent = Agent.create(
    adapter=adapter,
    agent_id=agent_id,
    api_key=api_key,
)
await agent.run()
```

---

## Package Structure

```
src/thenvoi/
├── agent.py                    # Agent compositor with create() factory
│
├── adapters/                   # Framework adapters (composition pattern)
│   ├── langgraph.py           # LangGraphAdapter
│   ├── anthropic.py           # AnthropicAdapter
│   ├── pydantic_ai.py         # PydanticAIAdapter
│   ├── claude_sdk.py          # ClaudeSDKAdapter
│   ├── crewai.py              # CrewAIAdapter
│   └── parlant.py             # ParlantAdapter
│
├── platform/                   # Transport layer
│   ├── link.py                # ThenvoiLink - WebSocket + REST client
│   └── events.py              # PlatformEvent - typed events
│
├── runtime/                    # Runtime layer
│   ├── agent_runtime.py       # AgentRuntime - convenience wrapper
│   ├── room_presence.py       # RoomPresence - cross-room lifecycle
│   ├── execution.py           # Execution + ExecutionContext
│   ├── agent_tools.py         # AgentTools - platform operations
│   ├── types.py               # PlatformMessage, configs
│   ├── prompts.py             # System prompt rendering
│   ├── formatters.py          # Message formatting utilities
│   └── trackers.py            # Participant + retry tracking
│
├── integrations/               # Framework-specific utilities
│   ├── langgraph/
│   │   ├── langchain_tools.py # agent_tools_to_langchain()
│   │   ├── graph_tools.py     # graph_as_tool()
│   │   └── message_formatters.py
│   ├── pydantic_ai/           # Pydantic AI utilities
│   ├── anthropic/             # Anthropic utilities
│   ├── claude_sdk/
│   │   ├── session_manager.py # Per-room session management
│   │   └── prompts.py         # Claude-specific prompts
│   └── a2a/
│       ├── adapter.py         # A2AAdapter (call external A2A agents)
│       ├── types.py           # A2A types
│       └── gateway/           # A2A Gateway adapter
│           ├── adapter.py     # A2AGatewayAdapter
│           ├── server.py      # GatewayServer (HTTP/SSE)
│           └── types.py       # GatewaySessionState
│
├── converters/                 # History conversion utilities
│   ├── anthropic.py           # AnthropicHistoryConverter
│   ├── pydantic_ai.py         # PydanticAIHistoryConverter
│   ├── claude_sdk.py          # ClaudeSDKHistoryConverter
│   ├── crewai.py              # CrewAIHistoryConverter
│   ├── parlant.py             # ParlantHistoryConverter
│   ├── a2a.py                 # A2AHistoryConverter
│   └── a2a_gateway.py         # GatewayHistoryConverter
│
├── client/                     # Low-level WebSocket client
│   └── streaming/
│       └── client.py
│
└── config/                     # Configuration utilities
    └── loader.py

examples/
├── run_agent.py               # Quick-start script for any framework
├── langgraph/                 # LangGraph examples (01-06)
├── pydantic_ai/               # Pydantic AI examples (01-02)
├── anthropic/                 # Anthropic SDK examples (01-02)
├── claude_sdk/                # Claude Agent SDK examples (01-02)
├── crewai/                    # CrewAI examples (01-04)
├── parlant/                   # Parlant examples (01-03)
├── a2a_bridge/                # A2A Adapter examples (call external A2A agents)
│   ├── 01_basic_agent.py      # Basic bridge setup
│   └── 02_with_auth.py        # Bridge with authentication
└── a2a_gateway/               # A2A Gateway examples (expose peers)
    ├── 01_basic_gateway.py    # Basic gateway setup
    ├── 02_with_demo_agent.py  # Gateway + orchestrator
    └── demo_orchestrator/     # LangGraph orchestrator agent
```

---

## Examples Overview

### LangGraph Examples (`examples/langgraph/`)

| File | Description |
|------|-------------|
| `01_simple_agent.py` | **Minimal setup** - Uses `Agent.create()` with LangGraphAdapter. Connects to platform and responds using built-in tools. |
| `02_custom_tools.py` | **Custom tools** - Adds your own `@tool` functions (calculator, weather) via `additional_tools` parameter. |
| `03_custom_personality.py` | **Custom behavior** - Uses `custom_instructions` to give the agent a pirate personality. |
| `04_calculator_as_tool.py` | **Graph-as-tool** - Wraps a standalone LangGraph as a tool using `graph_as_tool()`. Main agent delegates math to calculator subgraph. |
| `05_rag_as_tool.py` | **RAG subagent** - Wraps an Agentic RAG graph (retrieval + grading + rewriting) as a tool for research questions. |
| `06_delegate_to_sql_agent.py` | **SQL subagent** - Wraps a SQL agent with its own LLM and database tools. Main agent delegates queries to SQL subgraph. |

**Supporting files:** `standalone_calculator.py`, `standalone_rag.py`, `standalone_sql_agent.py` - Independent graphs used by examples 04-06.

### Pydantic AI Examples (`examples/pydantic_ai/`)

| File | Description |
|------|-------------|
| `01_basic_agent.py` | **Minimal setup** - Creates agent with PydanticAIAdapter using OpenAI. |
| `02_custom_instructions.py` | **Custom behavior** - Support agent persona using Anthropic Claude. |

### Anthropic SDK Examples (`examples/anthropic/`)

| File | Description |
|------|-------------|
| `01_basic_agent.py` | **Minimal setup** - Creates agent with AnthropicAdapter using Claude Sonnet. |
| `02_custom_instructions.py` | **Custom behavior** - Support agent with execution reporting enabled. |

### Claude Agent SDK Examples (`examples/claude_sdk/`)

| File | Description |
|------|-------------|
| `01_basic_agent.py` | **Minimal setup** - Creates agent with ClaudeSDKAdapter using Claude Sonnet. |
| `02_extended_thinking.py` | **Extended thinking** - Agent with 10,000 token thinking budget for complex reasoning. |

**Key features:**
- Automatic conversation history management (SDK handles it)
- Streaming responses via async iterator
- Extended thinking support with `max_thinking_tokens`
- MCP-based tool integration

### CrewAI Examples (`examples/crewai/`)

| File | Description |
|------|-------------|
| `01_basic_agent.py` | **Minimal setup** - Simple agent with CrewAIAdapter. |
| `02_role_based_agent.py` | **Role definition** - Agent with role, goal, and backstory. |
| `03_coordinator_agent.py` | **Multi-agent orchestration** - Coordinator that manages other agents. |
| `04_research_crew.py` | **Complete crew** - Research team with Analyst, Writer, and Editor. |

**Key features:**
- Role-based agent definition (role, goal, backstory)
- Multi-agent collaboration patterns
- Uses OpenAI-compatible API (set `OPENAI_API_KEY`)

### Parlant Examples (`examples/parlant/`)

| File | Description |
|------|-------------|
| `01_basic_agent.py` | **Minimal setup** - Simple agent with ParlantAdapter. |
| `02_with_guidelines.py` | **Behavioral guidelines** - Agent with condition/action rules. |
| `03_support_agent.py` | **Customer support** - Realistic support agent with specialized guidelines. |

**Key features:**
- Behavioral guidelines (condition/action pairs)
- Consistent, rule-following behavior
- Uses OpenAI-compatible API (set `OPENAI_API_KEY`)

### A2A Adapter Examples (`examples/a2a_bridge/`)

| File | Description |
|------|-------------|
| `01_basic_agent.py` | **Basic bridge** - Forwards Thenvoi messages to an external A2A agent. |
| `02_with_auth.py` | **With authentication** - A2A bridge with API key authentication. |

**Architecture:**
```
Thenvoi Platform → A2A Adapter → External A2A Agent (e.g., LangGraph currency agent)
       ↑                              ↓
       ←←←←←←←← Response ←←←←←←←←←←←←←
```

**Key features:**
- Call any A2A-compliant agent from Thenvoi platform
- Automatic session state persistence via task events
- Session rehydration when agent rejoins a room (`context_id` restored)
- Task resumption for `input_required` state via A2A resubscribe

### A2A Gateway Examples (`examples/a2a_gateway/`)

| File | Description |
|------|-------------|
| `01_basic_gateway.py` | **Basic gateway** - Exposes Thenvoi peers as A2A protocol endpoints. |
| `02_with_demo_agent.py` | **Gateway + Orchestrator** - Runs both gateway and demo orchestrator together. |
| `demo_orchestrator/` | **Demo Orchestrator** - LangGraph agent that routes requests to gateway peers. |

**Architecture:**
```
User → Orchestrator (10001) → A2A Gateway (10000) → Thenvoi Platform → Peer Agent
                            ↑                                              ↓
                            ←←←←←←←←←←← SSE Response ←←←←←←←←←←←←←←←←←←←←←
```

**Key features:**
- Exposes Thenvoi peers as A2A-compliant JSON-RPC endpoints
- Context ID preservation (same `contextId` → same chat room)
- Multi-peer support with automatic participant management
- SSE streaming responses (`text/event-stream`)

---

## Running Examples

### Quick Start with run_agent.py

```bash
# LangGraph (default)
uv run python examples/run_agent.py

# Pydantic AI
uv run python examples/run_agent.py --example pydantic_ai

# Anthropic SDK
uv run python examples/run_agent.py --example anthropic

# Claude SDK with extended thinking
uv run python examples/run_agent.py --example claude_sdk --thinking

# A2A Adapter (call external A2A agents from Thenvoi)
uv run python examples/run_agent.py --example a2a --a2a-url http://localhost:10000

# A2A Gateway (expose Thenvoi peers as A2A endpoints)
uv run python examples/run_agent.py --example a2a_gateway --debug

# See all options
uv run python examples/run_agent.py --help
```

### Individual Examples

```bash
# LangGraph
uv run python examples/langgraph/01_simple_agent.py
uv run python examples/langgraph/02_custom_tools.py

# Pydantic AI
uv run python examples/pydantic_ai/01_basic_agent.py

# Anthropic SDK
uv run python examples/anthropic/01_basic_agent.py

# Claude SDK
uv run python examples/claude_sdk/01_basic_agent.py

# CrewAI
uv run python examples/crewai/01_basic_agent.py
uv run python examples/crewai/02_role_based_agent.py

# Parlant
uv run python examples/parlant/01_basic_agent.py
uv run python examples/parlant/02_with_guidelines.py

# A2A Adapter
uv run python examples/a2a_bridge/01_basic_agent.py
```

### A2A Adapter Setup

Connect a Thenvoi agent to an external A2A-compliant agent:

```bash
# Terminal 1: Start an external A2A agent (e.g., LangGraph currency agent)
cd /path/to/a2a-samples/samples/python/agents/langgraph
python -m app --host localhost --port 10000

# Terminal 2: Start the Thenvoi A2A bridge agent
uv run python examples/run_agent.py --example a2a --a2a-url http://localhost:10000 --debug
```

The bridge agent forwards messages from Thenvoi platform to the external A2A agent and posts responses back to the chat.

### A2A Gateway Setup

Run the gateway and orchestrator to expose Thenvoi peers as A2A endpoints:

```bash
# Terminal 1: Start A2A Gateway (port 10000)
uv run python examples/run_agent.py --example a2a_gateway --debug

# Terminal 2: Start Demo Orchestrator (port 10001)
uv run python examples/a2a_gateway/demo_orchestrator/__main__.py --gateway-url http://localhost:10000
```

Test with curl:

```bash
# Send a message to the orchestrator (routes to gateway peers)
curl -X POST http://localhost:10001/ \
    -H "Content-Type: application/json" \
    -d '{
      "jsonrpc": "2.0",
      "id": "1",
      "method": "message/send",
      "params": {
        "message": {
          "role": "user",
          "parts": [{"kind": "text", "text": "Ask the weather peer about London"}],
          "messageId": "msg-1",
          "contextId": "ctx-1"
        }
      }
    }'

# Second message with SAME contextId uses the same chat room
curl -X POST http://localhost:10001/ \
    -H "Content-Type: application/json" \
    -d '{
      "jsonrpc": "2.0",
      "id": "2",
      "method": "message/send",
      "params": {
        "message": {
          "role": "user",
          "parts": [{"kind": "text", "text": "What about tomorrow?"}],
          "messageId": "msg-2",
          "contextId": "ctx-1"
        }
      }
    }'
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

```bash
# LangGraph examples
docker compose up langgraph-01-simple
docker compose up langgraph-02-custom-tools
docker compose up langgraph-03-custom-personality

# Rebuild after changes
docker compose up --build langgraph-01-simple

# Force recreate all containers
docker compose up --build --force-recreate

# Clean rebuild (removes cached layers)
docker compose build --no-cache
docker compose up
```

### Running with Docker (without compose)

```bash
# Build
docker build -t thenvoi-sdk .

# Run (load .env first)
set -a && source .env && set +a
docker run --rm \
  -e THENVOI_REST_URL="${THENVOI_REST_URL}" \
  -e THENVOI_WS_URL="${THENVOI_WS_URL}" \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  -v ./agent_config.yaml:/app/agent_config.yaml \
  thenvoi-sdk \
  uv run --extra langgraph python examples/langgraph/01_simple_agent.py
```

---

## Configuration

### 1. Copy configuration files from examples

Always copy from the example files to ensure correct URLs and formatting:

```bash
cp .env.example .env
cp agent_config.yaml.example agent_config.yaml
```

### 2. Edit `.env` with your API keys

The `.env` file contains platform URLs (pre-configured) and LLM API keys:

```bash
# Platform URLs
THENVOI_REST_URL=https://api.thenvoi.com
THENVOI_WS_URL=wss://api.thenvoi.com/ws

# LLM API Keys - fill these in
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 3. Edit `agent_config.yaml` with your agent credentials

Add your agent IDs and API keys from the Thenvoi platform:

```yaml
my_agent:
  agent_id: "your-agent-uuid"
  api_key: "your-api-key"
```

> **Security:** Never commit API keys. Both `.env` and `agent_config.yaml` are git-ignored.
> 
> **Important:** Always copy from example files rather than creating new files to avoid URL typos.

---

## Development

### Adding Dependencies

```bash
uv add package-name
uv add --optional langgraph package-name
```

### Running Tests

#### Unit Tests

Unit tests run without any external dependencies:

```bash
uv run pytest tests/ --ignore=tests/integration/
```

#### Integration Tests

Integration tests require a running Thenvoi API server and valid credentials.

**1. Set up test credentials:**

```bash
cp .env.test.example .env.test
```

Edit `.env.test` with your credentials:

```bash
# Server URLs
THENVOI_BASE_URL=http://localhost:4000
THENVOI_WS_URL=ws://localhost:4000/api/v1/socket/websocket

# Primary test agent (required for basic tests)
THENVOI_API_KEY=<your-agent-api-key>
TEST_AGENT_ID=<agent-uuid>

# Secondary test agent (required for multi-agent tests)
THENVOI_API_KEY_2=<your-second-agent-api-key>
TEST_AGENT_ID_2=<second-agent-uuid>

# User API key (required for dynamic agent tests)
THENVOI_API_KEY_USER=<your-user-api-key>
```

**Required credentials by test type:**

| Test Type | Required Credentials |
|-----------|---------------------|
| Basic agent tests | `THENVOI_API_KEY`, `TEST_AGENT_ID` |
| Multi-agent tests | Above + `THENVOI_API_KEY_2`, `TEST_AGENT_ID_2` |
| Dynamic agent tests | Above + `THENVOI_API_KEY_USER` |

**2. Run integration tests:**

```bash
# Run all integration tests
uv run pytest tests/integration/ -v

# Run specific test files
uv run pytest tests/integration/test_smoke.py -v           # Basic connectivity
uv run pytest tests/integration/test_multi_agent.py -v     # Multi-agent scenarios
uv run pytest tests/integration/test_dynamic_agent.py -v   # Dynamic agent creation

# Run with output visible
uv run pytest tests/integration/ -v -s
```

**3. Run all tests (unit + integration):**

```bash
uv run pytest tests/ -v
```

Tests will automatically skip if required credentials are not configured.

---

## Architecture

### Composition Pattern

The SDK uses composition over inheritance:

```
Agent.create(adapter, ...)
    │
    ├── Adapter (your LLM framework)
    │   └── on_started(), on_message(), on_cleanup()
    │
    ├── PlatformRuntime (room lifecycle)
    │   └── RoomPresence → Execution per room
    │
    └── ThenvoiLink (WebSocket + REST transport)
```

### Building Custom Adapters

Implement the `SimpleAdapter` protocol:

```python
from thenvoi.adapters.base import SimpleAdapter

class MyAdapter(SimpleAdapter[MyHistoryType]):
    async def on_started(self, agent_name: str, agent_description: str) -> None:
        """Called when agent starts."""
        pass

    async def on_message(
        self,
        ctx: ExecutionContext,
        tools: AgentTools,
        history: MyHistoryType,
    ) -> None:
        """Handle incoming message."""
        # Your LLM logic here
        await tools.send_message("Hello!")

    async def on_cleanup(self) -> None:
        """Called when agent stops."""
        pass
```

---

## LangGraph Utilities

### Wrap Graph as Tool

```python
from thenvoi.integrations.langgraph import graph_as_tool

calculator_tool = graph_as_tool(
    calculator_graph,
    name="calculator",
    description="Evaluates math expressions"
)

adapter = LangGraphAdapter(
    llm=llm,
    checkpointer=checkpointer,
    additional_tools=[calculator_tool],
)
```

### Convert Platform Tools to LangChain

```python
from thenvoi.integrations.langgraph import agent_tools_to_langchain

langchain_tools = agent_tools_to_langchain(agent_tools)
```

---

## Platform Tools

All adapters automatically have access to:

| Tool | Description |
|------|-------------|
| `send_message` | Send a message to the chat room |
| `add_participant` | Add a user or agent to the room |
| `remove_participant` | Remove a participant from the room |
| `get_participants` | List current room participants |
| `list_available_participants` | List users/agents that can be added |

---

## Context7 MCP for Up-to-Date Documentation

To ensure access to the latest documentation and code examples directly within your development environment, you can integrate the Context7 Model Context Protocol (MCP).

### Prerequisites

- Node.js version 18 or higher

### Configure Your MCP Client

<details>
<summary><strong>Cursor</strong></summary>

1. Navigate to `Settings` → `Cursor Settings` → `MCP` → `Add new global MCP server`
2. Add the following to your `~/.cursor/mcp.json` file:

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"]
    }
  }
}
```

</details>

<details>
<summary><strong>VS Code</strong></summary>

1. Install the MCP extension for VS Code
2. Add the following to your `settings.json` file:

```json
{
  "servers": {
    "Context7": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"]
    }
  }
}
```

</details>

<details>
<summary><strong>Claude Desktop</strong></summary>

Add the following to your `claude_desktop_config.json` file:

```json
{
  "mcpServers": {
    "Context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"]
    }
  }
}
```

</details>

### Usage

Once configured, add `use context7` to your prompts to fetch up-to-date documentation:

```
How do I create a LangGraph adapter in Thenvoi? use context7
```

This fetches the latest documentation and code examples, ensuring you have accurate and current information about the Thenvoi SDK and its dependencies.

---

## Help & Feedback

- **Documentation:** See `examples/` for complete working examples
- **Issues:** https://github.com/thenvoi/thenvoi-sdk-python/issues
