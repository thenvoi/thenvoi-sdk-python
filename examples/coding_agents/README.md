# Multi-Agent Docker Compose

Run a 3-agent team (Claude SDK planner + Codex reviewer + Codex implementer) sharing a workspace, connected to the Thenvoi platform.

## Architecture

```
docker compose up
├── planner        (ClaudeSDKAdapter, Claude model)
│   └── Role: planner — designs plans, coordinates agents
├── reviewer       (CodexAdapter, gpt-5.3-codex, reasoning: xhigh)
│   └── Role: reviewer — reviews plans and code, finds gaps and risks
└── implementer    (CodexAdapter, gpt-5.3-codex, reasoning: high)
    └── Role: coding — implements changes end-to-end
```

All services share `/workspace/repo` (mounted from host) and `/workspace/notes` + `/workspace/state` (Docker volumes).

The planner saves plans to `/workspace/notes/plan.md`. The reviewer cross-checks plans against source code. The implementer executes approved plans.

## Prerequisites

- Docker and Docker Compose v2
- Anthropic API key for the planner
- OpenAI API key for Codex agents
- Agent credentials from the Thenvoi platform (agent_id + api_key per agent)

## Setup

1. **Configure environment:**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys and repo path
   ```

2. **Configure agents:**

   ```bash
   cp agent_config.yaml.example agent_config.yaml
   # Edit with agent_id and api_key for each agent from Thenvoi platform
   ```

3. **Build and run:**

   ```bash
   docker compose build
   docker compose up -d
   ```

4. **View logs:**

   ```bash
   docker compose logs -f
   docker compose logs -f planner
   docker compose logs -f reviewer
   docker compose logs -f implementer
   ```

## How It Works

1. The **planner** receives a task and creates `/workspace/notes/plan.md` with a phased implementation plan
2. The planner @mentions the **reviewer** to review the plan
3. The reviewer reads the plan, cross-checks against source code, and provides structured feedback ([Critical], [Risk], [Gap], [Suggestion])
4. If changes are requested, the planner updates the plan and re-requests review
5. Once approved, the **implementer** executes the plan phases in `/workspace/repo`
6. Humans can join the conversation at any point to provide guidance or make decisions

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `THENVOI_REST_URL` | `https://app.thenvoi.com` | Platform REST API |
| `THENVOI_WS_URL` | `wss://app.thenvoi.com/...` | Platform WebSocket |
| `ANTHROPIC_API_KEY` | -- | Anthropic API key for planner |
| `OPENAI_API_KEY` | -- | OpenAI API key for reviewer/implementer |
| `REPO_PATH` | `.` | Host path to mount as `/workspace/repo` |
| `REVIEWER_AGENT_KEY` | `reviewer` | Agent config key for reviewer |
| `REVIEWER_MODEL` | `gpt-5.3-codex` | Model for reviewer |
| `REVIEWER_REASONING_EFFORT` | `xhigh` | Reasoning effort for reviewer |
| `IMPLEMENTER_AGENT_KEY` | `implementer` | Agent config key for implementer |
| `IMPLEMENTER_MODEL` | `gpt-5.3-codex` | Model for implementer |
| `IMPLEMENTER_REASONING_EFFORT` | `high` | Reasoning effort for implementer |

### Planner Prompts

The planner loads role prompts from `prompts/planner.md`. The Claude SDK runner injects workspace context (paths to `/workspace/repo` and `/workspace/notes`) automatically. Customize the prompt as needed.

## Troubleshooting

- **"Config file not found"**: Ensure `agent_config.yaml` exists (not just the `.example`).
- **"Missing required config fields"**: Fill in `agent_id` and `api_key` for each agent in `agent_config.yaml`.
- **Agent not connecting**: Check `THENVOI_REST_URL` / `THENVOI_WS_URL` in `.env`.
- **Codex model errors**: Verify `OPENAI_API_KEY` has access to `gpt-5.3-codex`.
- **Permission errors on `/workspace/repo`**: Ensure `REPO_PATH` points to an accessible directory.

## Cleanup

```bash
docker compose down           # Stop containers
docker compose down -v        # Stop and remove volumes (shared_notes, shared_state)
```
