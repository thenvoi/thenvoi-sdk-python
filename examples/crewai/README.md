# CrewAI Examples for Thenvoi SDK

This directory contains examples showing how to use CrewAI agents with the Thenvoi platform.

## Prerequisites

1. Install the SDK with CrewAI support:
   ```bash
   pip install thenvoi-sdk[crewai]
   ```

2. Set up environment variables:
   ```bash
   export THENVOI_WS_URL="wss://your-websocket-url"
   export THENVOI_REST_API_URL="https://your-api-url"
   export OPENAI_API_KEY="your-openai-key"
   ```

3. Create an `agent_config.yaml` file in the project root (see `agent_config.yaml.example`):
   ```yaml
   agents:
     simple_crewai_agent:
       id: "your-agent-uuid"
       api_key: "your-agent-api-key"
   ```

## Examples

### Basic Examples

| File | Description |
|------|-------------|
| `01_simple_agent.py` | Simplest possible agent - just call `create_crewai_agent()` |
| `02_custom_personality.py` | Agent with custom role, goal, and backstory |
| `03_custom_tools.py` | Agent with additional custom tools |

### Advanced Examples

| File | Description |
|------|-------------|
| `10_custom_crew.py` | Connect your own multi-agent Crew to the platform |

## Running Examples

```bash
cd examples/crewai
python 01_simple_agent.py
```

## Key Concepts

### ThenvoiCrewAIAgent vs ConnectedCrewAgent

- **ThenvoiCrewAIAgent**: Created by `create_crewai_agent()`. Builds a default single-agent Crew with platform tools. Best for simple use cases.

- **ConnectedCrewAgent**: Created by `connect_crew_to_platform()`. Connects your custom Crew to the platform. Best for complex multi-agent workflows.

### Platform Tools

All agents have access to these platform tools:
- `send_message` - Send messages to chat rooms (required for responding to users, must include at least one mention)
- `get_participants` - List participants in current room
- `add_participant` - Add users/agents to the room
- `remove_participant` - Remove participants from room
- `list_available_participants` - Find users/agents that can be added

### Room Context

Room context is automatically managed via `ThenvoiContext`. When a message arrives, the room ID is set before tools are invoked, so tools automatically know which room to operate in.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Thenvoi Platform                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Chat Room  │  │  Chat Room  │  │  Chat Room  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────┬───────────────────────────────────────┘
                      │ WebSocket
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  ThenvoiCrewAIAgent                         │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ PlatformClient  │  │   RoomManager   │                  │
│  └─────────────────┘  └─────────────────┘                  │
│           │                    │                            │
│           ▼                    ▼                            │
│  ┌─────────────────────────────────────────┐               │
│  │              CrewAI Crew                 │               │
│  │  ┌─────────────┐  ┌──────────────────┐  │               │
│  │  │   Agent     │  │  Platform Tools  │  │               │
│  │  └─────────────┘  └──────────────────┘  │               │
│  └─────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

## Differences from LangGraph

| Aspect | LangGraph | CrewAI |
|--------|-----------|--------|
| State Management | Built-in checkpointer | Stateless per task |
| Tool Config | Via RunnableConfig | Via ThenvoiContext |
| Multi-Agent | Via graph nodes | Via Crew with multiple agents |
| Streaming | Full event streaming | Task-level execution |

