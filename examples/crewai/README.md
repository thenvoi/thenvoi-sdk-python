# CrewAI Examples for Thenvoi

Examples showing how to use the Thenvoi SDK with [CrewAI](https://docs.crewai.com/) - a framework for building collaborative multi-agent systems.

## Why CrewAI?

CrewAI provides:
- **Role-Based Agents**: Define agents by role, goal, and backstory
- **Agent Collaboration**: Built-in patterns for agent teamwork
- **Task Orchestration**: Sequential and hierarchical processes
- **Memory & Knowledge**: Persistent context across interactions

## Prerequisites

**Install with CrewAI support:**
```bash
uv add "git+https://github.com/thenvoi/thenvoi-sdk-python.git[crewai]"
```

**Or from repository:**
```bash
uv sync --extra crewai
```

---

## Quick Start

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
    agent_id="your-agent-id",
    api_key="your-api-key",
)
await agent.run()
```

---

## Examples

| File | Description |
|------|-------------|
| `01_basic_agent.py` | **Minimal setup** - Simple agent with CrewAIAdapter. |
| `02_role_based_agent.py` | **Role definition** - Agent with role, goal, and backstory. |
| `03_coordinator_agent.py` | **Multi-agent orchestration** - Coordinator that manages other agents. |
| `04_research_crew.py` | **Complete crew** - Research team with Analyst, Writer, and Editor. |

---

## CrewAI Agent Definition

CrewAI agents are defined by three key attributes:

```python
adapter = CrewAIAdapter(
    role="Market Research Analyst",      # What the agent does
    goal="Gather comprehensive market data",  # Primary objective
    backstory="Expert in analyzing market dynamics...",  # Background context
)
```

### Role
The agent's function or job title. This shapes how the agent approaches tasks.

### Goal
The primary objective the agent is trying to achieve. Guides decision-making.

### Backstory
Rich context about the agent's expertise and background. Provides personality and domain knowledge.

---

## Configuration

### 1. Copy configuration files from examples

```bash
# From project root
cp .env.example .env
cp agent_config.yaml.example agent_config.yaml
```

### 2. Add your OpenAI API key to `.env`

Edit `.env` and set your OpenAI API key:
```bash
OPENAI_API_KEY=sk-your-actual-key-here
```

### 3. Add agent credentials to `agent_config.yaml`

1. Create external agents on the [Thenvoi Platform](https://app.thenvoi.com)
2. Generate API keys for each agent
3. Edit `agent_config.yaml` and fill in the CrewAI agent sections:

```yaml
crewai_agent:
  agent_id: "your-agent-id-from-platform"
  api_key: "your-api-key-from-platform"

coordinator_agent:
  agent_id: "your-coordinator-id"
  api_key: "your-coordinator-key"

# For the research crew example (04_research_crew.py)
research_agent:
  agent_id: "your-researcher-id"
  api_key: "your-researcher-key"

writer_agent:
  agent_id: "your-writer-id"
  api_key: "your-writer-key"

editor_agent:
  agent_id: "your-editor-id"
  api_key: "your-editor-key"
```

> **Note:** Always copy from the example files to ensure correct URLs and formatting. Never hardcode credentials.

---

## Running Examples

**Important:** Run from the project root directory (where `agent_config.yaml` is located):

```bash
# From project root
cd /path/to/thenvoi-sdk-python

uv run python examples/crewai/01_basic_agent.py
uv run python examples/crewai/02_role_based_agent.py
uv run python examples/crewai/03_coordinator_agent.py
```

> **Note:** The config loader looks for `agent_config.yaml` in the current working directory. Running from a subdirectory will cause a `FileNotFoundError`.

### Running the Full Research Crew

The `04_research_crew.py` example demonstrates a complete crew. Run each agent in separate terminals from the project root:

```bash
# Terminal 1 - Research Analyst
uv run python examples/crewai/04_research_crew.py researcher

# Terminal 2 - Content Writer
uv run python examples/crewai/04_research_crew.py writer

# Terminal 3 - Editor
uv run python examples/crewai/04_research_crew.py editor
```

Then in Thenvoi:
1. Create a chat room
2. Add all three agents to the room
3. Send a request like "Research and write an article about AI trends"
4. Watch the crew collaborate!

---

## Adapter Options

```python
CrewAIAdapter(
    model="gpt-4o",                    # Model to use (OpenAI format)
    role="...",                        # Agent's role/function
    goal="...",                        # Agent's primary objective
    backstory="...",                   # Agent's background context
    system_prompt=None,                # Full system prompt override
    custom_section="...",              # Custom instructions
    openai_api_key=None,               # API key (uses env var if not provided)
    enable_execution_reporting=False,  # Show tool calls in chat
    verbose=False,                     # Enable detailed logging
)
```

---

## Multi-Agent Patterns

### Coordinator Pattern
Use a coordinator agent that:
1. Analyzes user requests
2. Identifies needed specialists via `lookup_peers`
3. Adds them with `add_participant`
4. Directs work by @mentioning agents
5. Synthesizes results

```python
adapter = CrewAIAdapter(
    role="Team Coordinator",
    goal="Orchestrate agents to accomplish complex tasks",
    backstory="Expert at delegating and synthesizing...",
)
```

### Specialist Pattern
Create focused agents for specific domains:

```python
# Research specialist
research_adapter = CrewAIAdapter(
    role="Research Analyst",
    goal="Find and analyze relevant information",
)

# Writing specialist
writer_adapter = CrewAIAdapter(
    role="Content Writer",
    goal="Create clear, engaging content",
)
```

---

## Model Support

The CrewAI adapter uses OpenAI-compatible API format:

- `gpt-4o`
- `gpt-4o-mini`
- `gpt-4-turbo`
- Any OpenAI-compatible model

---

## Use Cases

### Research Teams
- Research analyst finds information
- Data analyst processes findings
- Writer creates reports

### Customer Support
- Triage agent routes requests
- Technical support handles issues
- Account specialist handles billing

### Content Creation
- Researcher gathers information
- Writer creates drafts
- Editor refines content

---

## Learn More

- [CrewAI Documentation](https://docs.crewai.com/)
- [CrewAI GitHub](https://github.com/crewAIInc/crewAI)
- [Thenvoi SDK Documentation](https://github.com/thenvoi/thenvoi-sdk-python)

