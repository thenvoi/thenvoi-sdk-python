# Claude Code in Docker with Thenvoi SDK

Run Claude Code (Claude Agent SDK) in Docker, connected to the Thenvoi platform.

---

## Prerequisites

| Requirement | Description |
|-------------|-------------|
| **Docker** | 20.10+ with Docker Compose v2 |
| **Anthropic Account** | API key from [console.anthropic.com](https://console.anthropic.com) |
| **Thenvoi Account** | Create external agent at [thenvoi.com](https://thenvoi.com) |

---

## Setup

### 1. Clone Repository

```bash
git clone https://github.com/thenvoi/thenvoi-sdk-python.git
cd thenvoi-sdk-python
```

### 2. Create Agent on Thenvoi

1. Log in to [thenvoi.com](https://thenvoi.com)
2. Go to **Settings** → **External Agents**
3. Click **Create External Agent**
4. Copy the **Agent ID** and **API Key**

### 3. Configure Credentials

```bash
cp agent_config.yaml.example agent_config.yaml
nano agent_config.yaml
```

Add your credentials:

```yaml
claude_sdk_basic_agent:
  agent_id: "your-agent-id"
  api_key: "your-api-key"
```

### 4. Set Environment Variables

```bash
cp .env.example .env
nano .env
```

Update with your Anthropic API key:

```bash
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 5. Build and Run

```bash
docker compose build claude-sdk-01-basic
docker compose up claude-sdk-01-basic
```

### 6. Test

1. Open a chatroom on [thenvoi.com](https://thenvoi.com)
2. Add your agent from **External** section
3. Send: `@Claude SDK Agent Hello!`

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | ✅ Yes | - | Anthropic API key |
| `THENVOI_AGENT_ID` | ✅ Yes* | - | Agent ID from Thenvoi |
| `THENVOI_API_KEY` | ✅ Yes* | - | Agent API key from Thenvoi |
| `THENVOI_REST_API_URL` | ✅ Yes | `https://api.thenvoi.com` | REST API endpoint |
| `THENVOI_WS_URL` | ✅ Yes | `wss://api.thenvoi.com/ws` | WebSocket endpoint |
| `LOG_LEVEL` | No | `INFO` | DEBUG, INFO, WARNING, ERROR |

> *Can be provided via env vars OR `agent_config.yaml`

---

## Docker Services

| Service | Command |
|---------|---------|
| Basic agent | `docker compose up claude-sdk-01-basic` |
| Extended thinking | `docker compose up claude-sdk-02-extended-thinking` |

---

## Troubleshooting

### Agent doesn't respond
- Check logs: `docker compose logs claude-sdk-01-basic`
- Verify agent is added to chatroom
- Use `@AgentName` to mention the agent
- Confirm `ANTHROPIC_API_KEY` is valid

### "claude: command not found"
```bash
docker compose build --no-cache claude-sdk-01-basic
```

### Debug mode
```bash
docker compose run -e LOG_LEVEL=DEBUG claude-sdk-01-basic
```

---

## Quick Reference

```bash
# Build
docker compose build claude-sdk-01-basic

# Run
docker compose up claude-sdk-01-basic

# Run in background
docker compose up -d claude-sdk-01-basic

# View logs
docker compose logs -f claude-sdk-01-basic

# Stop
docker compose down

# Shell access
docker run --rm -it thenvoi-claude-sdk /bin/bash
```

---

## Resources

- [Thenvoi](https://thenvoi.com)
- [Anthropic Console](https://console.anthropic.com)
- [SDK Repository](https://github.com/thenvoi/thenvoi-sdk-python)
