# Multi-Agent Docker Compose

Run a 3-agent team (Claude Code planner + Codex reviewer + Codex implementer) sharing a workspace, connected to the Thenvoi platform.

## Architecture

```
docker compose up
├── claude-planner     (ClaudeSDKAdapter, Claude model)
│   └── Role: planner — designs plans, coordinates agents
├── codex-reviewer     (CodexAdapter, gpt-5.3-codex, reasoning: xhigh)
│   └── Role: reviewer — reviews code, finds regressions
└── codex-implementer  (CodexAdapter, gpt-5.3-codex, reasoning: high)
    └── Role: coding — implements changes end-to-end
```

All services share `/workspace/repo` (mounted from host) and `/workspace/notes` (Docker volume).

## Prerequisites

- Docker and Docker Compose v2
- Anthropic API key (or Claude MAX subscription) for the planner
- OpenAI API key for Codex agents
- Agent credentials from the Thenvoi platform (agent_id + api_key per agent)
- Agents registered in `agent_config.yaml` at the repo root

## Setup

1. **Configure environment:**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys and agent config keys
   ```

2. **Configure planner agent:**

   ```bash
   cp claude-planner.yaml.example claude-planner.yaml
   # Edit with agent_id and api_key from Thenvoi platform
   ```

3. **Ensure agent_config.yaml** exists at the repo root with entries for the reviewer and implementer agents (keyed by `CODEX_REVIEWER_AGENT_KEY` and `CODEX_IMPLEMENTER_AGENT_KEY`).

4. **Build and run:**

   ```bash
   docker compose build
   docker compose up -d
   ```

5. **View logs:**

   ```bash
   docker compose logs -f
   docker compose logs -f claude-planner
   docker compose logs -f codex-reviewer
   docker compose logs -f codex-implementer
   ```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `THENVOI_REST_URL` | `https://app.thenvoi.com` | Platform REST API |
| `THENVOI_WS_URL` | `wss://app.thenvoi.com/...` | Platform WebSocket |
| `ANTHROPIC_API_KEY` | — | Anthropic API key for planner |
| `OPENAI_API_KEY` | — | OpenAI API key for Codex agents |
| `REPO_PATH` | `.` | Host path to mount as `/workspace/repo` |
| `CODEX_REVIEWER_AGENT_KEY` | `reviewer` | Agent config key for reviewer |
| `CODEX_REVIEWER_REASONING_EFFORT` | `xhigh` | Reasoning effort for reviewer |
| `CODEX_IMPLEMENTER_AGENT_KEY` | `implementer` | Agent config key for implementer |
| `CODEX_IMPLEMENTER_REASONING_EFFORT` | `high` | Reasoning effort for implementer |

### Planner Prompts

The planner loads role prompts from `prompts/`. The default `prompts/planner.md` is a symlink to `examples/claude_code/prompts/planner.md`. Customize or replace it as needed.

## Troubleshooting

- **"Config file not found"**: Ensure `claude-planner.yaml` exists (not just the `.example`).
- **"Missing required config fields"**: Fill in `agent_id` and `api_key` in `claude-planner.yaml`.
- **Agent not connecting**: Check `THENVOI_REST_URL` / `THENVOI_WS_URL` in `.env`.
- **Codex model errors**: Verify `OPENAI_API_KEY` has access to `gpt-5.3-codex`.
- **Permission errors on `/workspace/repo`**: Ensure `REPO_PATH` points to an accessible directory.

## Stopping

```bash
docker compose down           # Stop containers
docker compose down -v        # Stop and remove volumes (shared_notes)
```
