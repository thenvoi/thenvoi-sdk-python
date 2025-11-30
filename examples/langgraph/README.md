# LangGraph Examples for Thenvoi

This guide explains all the ways you can integrate LangGraph agents with the Thenvoi platform.

## Prerequisites

**If running from repository:**
```bash
# From python-sdk/ directory
uv sync --extra langgraph
```

**If using as external library:**
```bash
uv pip install "git+https://github.com/thenvoi/python-sdk.git#egg=thenvoi-python-sdk[langgraph]"
```

**Configuration:**
- Set `OPENAI_API_KEY` environment variable
- Configure agent credentials (see main [README](../../README.md#creating-external-agents-on-thenvoi-platform))

## Two Integration Approaches

### Approach 1: Start with Thenvoi's Built-in Agent (Recommended for most cases)

**When to use:** You want a working agent quickly and just need to customize it with your own tools or personality.

**Examples:** `01_simple_agent.py`, `02_custom_tools.py`, `03_custom_personality.py`, `10_calculator_as_tool.py`, `11_rag_as_tool.py`, `12_delegate_to_sql_agent.py`

**Key function:** `create_langgraph_agent()`

```python
from thenvoi.adapters.langgraph import create_langgraph_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

agent = await create_langgraph_agent(
    agent_id=agent_id,
    api_key=api_key,
    llm=ChatOpenAI(model="gpt-4o"),
    checkpointer=InMemorySaver(),
    ws_url=ws_url,
    thenvoi_restapi_url=thenvoi_restapi_url,
    additional_tools=[my_tool1, my_tool2],      # Optional: add your tools
    custom_instructions="Your custom behavior"  # Optional: customize personality
)
```

**What you get:**
- Pre-built agent architecture (LLM → Tools → Conditional edges)
- All Thenvoi platform tools automatically included
- Standard system prompt handling multi-participant rooms
- Just add your custom tools and/or instructions

---

### Approach 2: Bring Your Own Custom Graph (For advanced use cases)

**When to use:** You need to customize the system prompt or have full control over the agent's graph architecture.

**Examples:** `20_custom_agent_with_instructions.py` (simple), `21_custom_graph.py` (advanced)

**Key function:** `connect_graph_to_platform()`

**Two levels of control:**

1. **Simple (Example 20):** Use LangGraph's `create_agent()` helper
   - Just provide LLM, tools, and system prompt
   - Automatically builds the graph for you
   - Perfect when you want custom instructions without complex logic

2. **Advanced (Example 21):** Manually build the graph
   - Full control over nodes, edges, and state
   - Use when you need custom graph architecture
   - Maximum flexibility for complex workflows

```python
from thenvoi.adapters.langgraph import connect_graph_to_platform, get_thenvoi_tools
from thenvoi.core.platform_client import ThenvoiPlatformClient
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import InMemorySaver

# STEP 1: Create platform client (shared for all operations)
platform_client = ThenvoiPlatformClient(
    agent_id=agent_id,
    api_key=api_key,
    ws_url=ws_url,
    thenvoi_restapi_url=thenvoi_restapi_url,
)

# STEP 2: Get platform tools using the same client
platform_tools = get_thenvoi_tools(
    client=platform_client.api_client,
    agent_id=agent_id
)

# STEP 3: Build YOUR custom graph with those tools
graph = StateGraph(MessagesState)
graph.add_node("my_node", my_custom_node)
graph.add_node("tools", ToolNode(platform_tools))  # Add platform tools here
# ... define your custom edges, conditional logic, etc ...
my_graph = graph.compile(checkpointer=InMemorySaver())

# STEP 4: Connect your graph to Thenvoi (reusing same client)
agent = await connect_graph_to_platform(
    graph=my_graph,
    platform_client=platform_client,  # Reuse the same client!
)
```

**What you control:**
- Complete graph architecture
- Custom state structure
- When and how to use platform tools
- All workflow logic

**The two-part integration:**
1. **Input (Automatic):** Thenvoi delivers chat messages to your graph
2. **Output (Your Control):** You decide when to call platform tools inside your graph

---

## Examples by Category

### Getting Started

**`01_simple_agent.py`** - Simplest possible agent
- Just LLM + platform tools
- No custom tools or instructions
- Great starting point

**`02_custom_tools.py`** - Add your own tools
- Built-in agent + calculator and weather tools
- Shows `additional_tools` parameter
- Tools automatically bound to LLM

**`03_custom_personality.py`** - Custom personality
- Built-in agent + pirate personality
- Shows `custom_instructions` parameter
- Maintains all tool functionality

### Advanced: Delegating to Sub-Agents

**`10_calculator_as_tool.py`** - Calculator sub-graph
- Main agent delegates math to calculator sub-graph
- Uses `graph_as_tool()` to wrap graphs as tools
- Shows hierarchical agent architecture

**`11_rag_as_tool.py`** - RAG sub-graph
- Main agent delegates research to RAG sub-graph
- Demonstrates complex sub-graph with vector search
- Shows result formatting from sub-graphs

**`12_delegate_to_sql_agent.py`** - SQL sub-agent
- Main agent delegates database queries to SQL expert
- Shows domain-specific delegation
- Demonstrates error handling in sub-graphs

### Advanced: Custom Graph Architecture

**`20_custom_agent_with_instructions.py`** - Custom agent with system instructions
- Use LangGraph's `create_agent()` helper
- Customize system prompt using `generate_langgraph_agent_prompt()`
- Simpler than manual graph building
- Perfect for custom instructions without complex logic

**`21_custom_graph.py`** - Bring your own graph (maximum control)
- Build your graph from scratch manually
- Full control over nodes, edges, and state
- Integrate platform tools where you want them
- Use for complex custom workflows

---

## Available Thenvoi Platform Tools

These tools are automatically available in all approaches:

| Tool | Description |
|------|-------------|
| `send_message` | Send a message to the chat room |
| `add_participant` | Add a user or agent to the room |
| `remove_participant` | Remove a participant from the room |
| `get_participants` | List current room participants |
| `list_available_participants` | List users/agents that can be added |

**Important:** All tools automatically know which room they're operating in (via `thread_id`). You don't need to pass room IDs manually.

---

## Quick Comparison

| Feature | Built-in Agent | Custom Agent (create_agent) | Custom Graph (manual) |
|---------|---------------|---------------------------|---------------------|
| **Setup complexity** | Simple | Simple | Advanced |
| **Graph architecture** | Pre-built | Auto-built by helper | You design it |
| **Platform tools** | Auto-included | You add them | You add them |
| **Custom tools** | `additional_tools` param | Pass to `create_agent()` | Add to your ToolNode |
| **System prompt** | Default | Via `system_prompt` param | Manual in agent node |
| **State structure** | MessagesState | MessagesState | Any custom state |
| **Best for** | Quick start, standard agents | Custom instructions | Complex workflows |
| **Example** | `02_custom_tools.py` | `20_custom_agent_with_instructions.py` | `21_custom_graph.py` |

---

## Common Patterns

### Pattern 1: Add Custom Tools to Built-in Agent

```python
from langchain_core.tools import tool

@tool
def my_custom_tool(query: str) -> str:
    """Does something useful."""
    return "result"

agent = await create_langgraph_agent(
    ...,
    additional_tools=[my_custom_tool]  # Your tools added here
)
```

### Pattern 2: Wrap Sub-Graph as a Tool

```python
from thenvoi.adapters.langgraph import graph_as_tool

# Create a sub-graph for specialized work
calculator_graph = create_calculator_graph()

# Wrap it as a tool
calculator_tool = graph_as_tool(
    calculator_graph,
    name="calculator",
    description="Evaluates math expressions"
)

# Add to main agent
agent = await create_langgraph_agent(
    ...,
    additional_tools=[calculator_tool]
)
```

### Pattern 3: Custom Agent with System Instructions

```python
from langgraph.prebuilt import create_agent
from thenvoi.adapters.langgraph.prompts import generate_langgraph_agent_prompt
from thenvoi.core.platform_client import ThenvoiPlatformClient

# Create platform client
platform_client = ThenvoiPlatformClient(
    agent_id=agent_id,
    api_key=api_key,
    ws_url=ws_url,
    thenvoi_restapi_url=thenvoi_restapi_url,
)

# Fetch agent metadata to get name
await platform_client.fetch_agent_metadata()

# Get platform tools
platform_tools = get_thenvoi_tools(
    client=platform_client.api_client,
    agent_id=agent_id
)

# Generate system instructions with agent's name
system_instructions = generate_langgraph_agent_prompt(platform_client.name)

# Build agent with custom instructions
my_graph = create_agent(
    model=llm,
    tools=platform_tools,
    system_prompt=system_instructions,  # Custom instructions!
    checkpointer=InMemorySaver()
)

# Connect to platform
agent = await connect_graph_to_platform(
    graph=my_graph,
    platform_client=platform_client,
)
```

### Pattern 4: Custom Graph with Platform Tools (Manual Construction)

```python
from thenvoi.core.platform_client import ThenvoiPlatformClient

# Create platform client (single client for everything)
platform_client = ThenvoiPlatformClient(
    agent_id=agent_id,
    api_key=api_key,
    ws_url=ws_url,
    thenvoi_restapi_url=thenvoi_restapi_url,
)

# Get platform tools using the same client
platform_tools = get_thenvoi_tools(
    client=platform_client.api_client,
    agent_id=agent_id
)

# Build your custom graph
graph = StateGraph(MyCustomState)
graph.add_node("my_logic", my_custom_logic)
graph.add_node("tools", ToolNode(platform_tools))  # Platform tools here
# ... your custom edges ...

# Connect to platform (reusing same client)
agent = await connect_graph_to_platform(
    graph=graph.compile(checkpointer=InMemorySaver()),
    platform_client=platform_client,
)
```

---

## Configuration

All examples use `agent_config.yaml` to store agent credentials:

```yaml
agents:
  my_agent:
    agent_id: "agent_123"
    api_key: "key_456"

  custom_tools_agent:
    agent_id: "agent_789"
    api_key: "key_012"

  custom_graph_agent:
    agent_id: "agent_345"
    api_key: "key_678"
```

Load config in your code:

```python
from thenvoi.config import load_agent_config

agent_id, api_key = load_agent_config("my_agent")
```

---

## Running Examples

**From repository:**
```bash
# Simple agent
uv run --extra langgraph python examples/langgraph/01_simple_agent.py

# Agent with custom tools
uv run --extra langgraph python examples/langgraph/02_custom_tools.py

# Agent with custom personality
uv run --extra langgraph python examples/langgraph/03_custom_personality.py

# Calculator sub-graph
uv run --extra langgraph python examples/langgraph/10_calculator_as_tool.py

# RAG sub-graph
uv run --extra langgraph python examples/langgraph/11_rag_as_tool.py

# SQL sub-agent
uv run --extra langgraph python examples/langgraph/12_delegate_to_sql_agent.py

# Custom agent with instructions (simple)
uv run --extra langgraph python examples/langgraph/20_custom_agent_with_instructions.py

# Custom graph (advanced)
uv run --extra langgraph python examples/langgraph/21_custom_graph.py
```

**Using as external library:**

If you installed the SDK as an external library, copy any example to your project and run with:
```bash
uv run python your_agent.py
```

**Note:** Examples use `.env` and `agent_config.yaml` for configuration. You can adapt this to use environment variables, function arguments, or any other configuration method in your own projects.

---

## Need Help?

- **Start simple:** Try `01_simple_agent.py` first
- **Add tools:** Use `02_custom_tools.py` as a template
- **Custom instructions:** Check `20_custom_agent_with_instructions.py` for system prompts
- **Complex workflows:** Check `21_custom_graph.py` for manual graph construction
- **Sub-agents:** See `10_calculator_as_tool.py` for delegation patterns
