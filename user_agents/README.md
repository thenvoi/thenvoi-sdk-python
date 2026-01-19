# Agents

## Quick Start

```bash
# 1. Create .env
cp example.env .env
# Add ANTHROPIC_API_KEY

# 2. Create agent config
cp example_agent.yaml agent1.yaml
# Edit agent1.yaml - set agent_id and api_key

# 3. Run
docker compose up -d
```

## Add More Agents

**Step 1:** Create config file
```bash
cp example_agent.yaml agent2.yaml
# Edit agent2.yaml with new agent_id and api_key
```

**Step 2:** Add to docker-compose.yml
```yaml
  agent2:
    <<: *agent-base
    environment:
      AGENT_CONFIG: /app/user_agents/agent2.yaml
```

**Step 3:** Run
```bash
docker compose up -d
```

## Commands

```bash
docker compose up -d          # Start all agents
docker compose up agent1 -d   # Start specific agent
docker compose down           # Stop all
docker compose logs agent1    # View logs
docker compose restart agent1 # Restart agent
```
