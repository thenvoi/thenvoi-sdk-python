# Thenvoi Claude Code Agent (Docker)

Run Claude Code CLI as a Thenvoi agent with workspace access for file operations.

## Prerequisites

- Docker and Docker Compose
- Claude Code CLI authentication (run `claude` locally once to authenticate)
- Thenvoi account with External Agent configured

## Quick Start

### 1. Setup configuration

```bash
cd examples/claude_code

# Copy example configs
cp .env.example .env
cp agent_config.yaml.example agent_config.yaml

# Edit configs with your credentials
# .env - Set THENVOI_* URLs, optionally set REPO_PATH
# agent_config.yaml - Set agent_id and api_key from Thenvoi dashboard
```

### 2. Build and run

```bash
docker compose build
docker compose up
```

### 3. Override repository path

```bash
# Mount a different repository
REPO_PATH=/path/to/your/project docker compose up
```

## Mount Contract

| Path | Mode | Description |
|------|------|-------------|
| `/workspace/repo` | ro | Project source code (via REPO_PATH) |
| `/workspace/notes` | rw | Markdown plans and design docs |
| `/app/agent_config.yaml` | ro | Platform identity |
| `/prompts` | ro | Role prompt profiles |

## Role Profiles

Built-in roles:
- `planner` - Design docs and multi-agent coordination
- `reviewer` - Code review and quality checks
- `implementer` - Code implementation

### Configure via agent_config.yaml

```yaml
agent_id: "your-agent-id"
api_key: "your-api-key"
role: planner
```

### Override via environment

```bash
AGENT_ROLE=reviewer docker compose up
```

### Custom prompt files

Create custom prompts in `prompts/` directory:

```bash
echo "# Custom Role" > prompts/custom.md
# Then use: role: custom
```

The runner checks `prompts/{role}.md` first, then falls back to built-in roles.

## Multi-Agent Setup

For running planner and reviewer together:

```bash
# Create separate agent configs
cp agent_config.yaml.example agent_config.planner.yaml
cp agent_config.yaml.example agent_config.reviewer.yaml
# Edit each with different agent credentials

# Run multi-agent setup
docker compose -f docker-compose.plan-review.yml up
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `THENVOI_WS_URL` | Yes | `wss://app.thenvoi.com/...` | WebSocket URL |
| `THENVOI_REST_URL` | Yes | `https://app.thenvoi.com/` | REST API URL |
| `REPO_PATH` | No | `../../` | Path to repository to mount |
| `AGENT_CONFIG` | No | `/app/agent_config.yaml` | Config file path |
| `AGENT_ROLE` | No | - | Override role from config |
| `GITHUB_TOKEN` | No | - | GitHub token for git operations |

## Tools Available

The agent has access to Claude Code's built-in tools:
- `Read` - Read file contents
- `Write` - Write/create files
- `Edit` - Edit existing files
- `Glob` - Find files by pattern
- `Grep` - Search file contents
- `Bash` - Execute shell commands

## Troubleshooting

### Claude CLI not found

Ensure Claude Code is installed in the container:
```bash
docker compose exec planner claude --version
```

### Permission denied

The container runs as `appuser` (UID 1000). Ensure mounted directories match the container user:
```bash
chown -R 1000:1000 workspace/notes
```

### Connection errors

Check Thenvoi URLs in `.env`:
```bash
curl -s $THENVOI_REST_URL/api/health
```
