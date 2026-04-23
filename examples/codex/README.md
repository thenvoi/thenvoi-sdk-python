# Codex Examples for Thenvoi

Examples for creating Thenvoi agents backed by [OpenAI Codex](https://github.com/openai/codex) app-server, via stdio or WebSocket transport.

## Prerequisites

1. **Install Codex CLI**: follow https://github.com/openai/codex
2. **Sign in**: `codex login`
3. **Thenvoi Platform** — create agents and add them to `agent_config.yaml`. Default key is `darter` (override with `AGENT_KEY` / `CODEX_*_AGENT_KEY`).
4. **Dependencies** — `uv sync --extra codex`
5. **Environment variables** in `.env`:
   - `THENVOI_WS_URL`, `THENVOI_REST_URL`

## Running locally (no Docker)

```bash
uv run examples/codex/01_basic_agent.py
```

Relevant env overrides:

| Variable | Default | Purpose |
|---|---|---|
| `AGENT_KEY` | `darter` | Agent key in `agent_config.yaml` |
| `CODEX_TRANSPORT` | `stdio` | `stdio` or `ws` |
| `CODEX_WS_URL` | `ws://127.0.0.1:8765` | Only used when `CODEX_TRANSPORT=ws` |
| `CODEX_ROLE` | unset | If set, loads `prompts/<role>.md` into the system prompt |
| `CODEX_MODEL` | Codex default | e.g. `gpt-5.3-codex` |
| `CODEX_APPROVAL_MODE` | `manual` | `manual`, `auto_accept`, `auto_decline` |
| `CODEX_TURN_TASK_MARKERS` | `false` | Emit turn/task boundary markers |

For `ws` transport, start the app-server first: `codex app-server --listen ws://127.0.0.1:8765`

## Running via Docker

Three compose files are provided — each builds from `docker/codex/Dockerfile` (at repo root) and mounts the repo as `/workspace/repo`:

| File | Services | Purpose |
|---|---|---|
| `docker-compose.yml` | `codex-agent` | Single Codex agent (default role: `darter`) |
| `docker-compose.multi.yml` | `codex-darter`, `codex-reviewer` | Two agents sharing workspace volumes |
| `docker-compose.plan-review.yml` | `codex-planner`, `codex-reviewer` | Planner/reviewer pair |

Run from the `examples/codex/` directory:

```bash
cd examples/codex
docker compose up --build                       # single agent
docker compose -f docker-compose.multi.yml up   # multiple agents
docker compose -f docker-compose.plan-review.yml up
```

Each service expects `agent_config.yaml` at the repo root and reads `../../.env` for Thenvoi/LLM credentials. The container reaches the host's Thenvoi platform via `host.docker.internal`; override via `THENVOI_REST_URL_DOCKER` / `THENVOI_WS_URL_DOCKER`.
