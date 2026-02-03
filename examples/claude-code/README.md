# Thenvoi Agent Runner

Run AI agents powered by Claude SDK with Docker - no coding required.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)

## Quick Start

### 1. Configure your environment

```bash
cp ../../.env.example .env
# Edit .env - add your ANTHROPIC_API_KEY
```

### 2. Create your agent

1. Go to the [Thenvoi Dashboard](https://app.thenvoi.com/dashboard)
2. Log in or create an account
3. Create a new **External Agent**
4. Copy your **Agent ID** and **API Key** from the agent settings
5. Copy `example_agent.yaml` to a new file (e.g. `agent1.yaml`):

```bash
cp example_agent.yaml agent1.yaml
```

6. Edit your new agent file and paste your credentials:

```yaml
agent_id: "your-agent-id-from-thenvoi"
api_key: "your-api-key-from-thenvoi"
```

You can create multiple agents by repeating these steps with different files (e.g. `agent2.yaml`, `agent3.yaml`).

### 3. Update docker-compose.yml

Add your agent(s) to `docker-compose.yml`:

```yaml
services:
  agent1:
    <<: *agent-base
    environment:
      AGENT_CONFIG: /app/config/agent1.yaml
```

### 4. Build and run

```bash
# Build the Docker image
docker compose build

# Run the agent
docker compose up
```

## Files

| File | Description |
|------|-------------|
| `example_agent.yaml` | Template for agent configuration (copy this to create your agents) |
| `docker-compose.yml` | Docker configuration (add your agents here) |
| `Dockerfile` | Docker image definition (builds from SDK source) |
| `runner.py` | Agent runner script (reads YAML config) |
| `tools/` | Custom tools for your agent |

## Agent Configuration

```yaml
agent_id: "from-thenvoi-platform"
api_key: "from-thenvoi-platform"
model: claude-sonnet-4-5-20250929
prompt: |
  You are a helpful assistant.
tools:
  - calculator
  - get_time
```

## Multiple Agents

To run multiple agents, repeat steps 2-3 for each agent:

1. Create a new external agent on Thenvoi and copy the credentials
2. Copy `example_agent.yaml` to a new file (e.g. `agent2.yaml`)
3. Edit the new file with the credentials
4. Add the new agent to `docker-compose.yml`
5. Run: `docker compose up`

## Custom Tools

Edit `tools/example_tools.py` to add your own tools:

```python
@tool("my_tool", "Description", {"param": str})
async def my_tool(args: dict) -> dict:
    return {"content": [{"type": "text", "text": args["param"]}]}
```

Then enable in `tools/__init__.py` and add to your agent config.

## Commands

```bash
docker compose build        # Build the image
docker compose up -d        # Start in background
docker compose logs -f      # View logs
docker compose down         # Stop
docker compose restart      # Restart
```
