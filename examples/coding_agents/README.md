# Multi-Agent Docker Compose (Bridge + Specialists)

Run a local 3-service room with a bridge coordinator and two specialists:

- `bridge` (Claude SDK): coordinator service
- `planner` (Claude SDK): planning specialist
- `reviewer` (Codex): review specialist

All services share the same workspace volumes and connect to Thenvoi.

## Architecture

```text
docker compose up
├── bridge      (coordinator, exposes /health)
├── planner     (specialist, waits for bridge health)
└── reviewer    (specialist, waits for bridge health)
```

Shared volumes:
- `/workspace/repo`
- `/workspace/notes`
- `/workspace/state`
- `/workspace/context`

## Prerequisites

- Docker + Docker Compose v2
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- Thenvoi credentials for `bridge`, `planner`, and `reviewer` in `agent_config.yaml`

## Quickstart

1. Create local config files:

```bash
cp .env.example .env
cp agent_config.yaml.example agent_config.yaml
```

2. Fill required values:
- `.env`: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`
- `agent_config.yaml`: `bridge.agent_id/api_key`, `planner.agent_id/api_key`, `reviewer.agent_id/api_key`

`THENVOI_REST_URL` and `THENVOI_WS_URL` are optional. If you omit them, the stack uses the hosted Thenvoi defaults.

3. Start services:

```bash
docker compose build
docker compose up -d
```

4. Check health and logs:

```bash
docker compose ps
docker compose logs -f bridge
docker compose logs -f planner
docker compose logs -f reviewer
```

## Required Variables and Defaults

`docker-compose.yml` requires:
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`

Platform URLs are optional:
- `THENVOI_REST_URL` defaults to `https://app.thenvoi.com`
- `THENVOI_WS_URL` defaults to `wss://app.thenvoi.com/api/v1/socket/websocket`

If a required API key is missing, `docker compose up` fails immediately with a clear variable error.

## Readiness Gating

- `bridge` exposes `GET /health` on `BRIDGE_HEALTH_PORT` (default `18080`).
- `planner` and `reviewer` both:
  - depend on `bridge` health (`depends_on: condition: service_healthy`)
  - run `wait_for_bridge.sh` before starting their runners
- If bridge health is not reachable within `BRIDGE_WAIT_TIMEOUT_S`, specialists exit non-zero.

## Smoke Test

Run an end-to-end room messaging check after the stack is up:

```bash
uv run python examples/coding_agents/smoke_test_room_messaging.py
```

What it validates:
- Creates a room using `bridge` credentials.
- Adds `reviewer` as participant.
- Sends a mention from bridge to reviewer.
- Verifies reviewer reply before timeout.

Optional flags:

```bash
uv run python examples/coding_agents/smoke_test_room_messaging.py \
  --timeout-seconds 120 \
  --poll-interval-seconds 2 \
  --bridge-key bridge \
  --reviewer-key reviewer
```

## Troubleshooting

- `variable is required` during compose startup:
  - missing required `.env` value; set it and re-run `docker compose up -d`.
- `bridge` unhealthy:
  - inspect `docker compose logs bridge`.
  - verify `BRIDGE_HEALTH_PORT` is not blocked/conflicting.
- specialists exit with `Timed out waiting for bridge health`:
  - bridge did not become healthy within timeout.
  - inspect bridge logs and increase `BRIDGE_WAIT_TIMEOUT_S` if needed.
- Thenvoi auth failures:
  - verify `agent_config.yaml` IDs and API keys.

## Restart and Recovery

Clean lifecycle validation:

```bash
docker compose up -d
uv run python examples/coding_agents/smoke_test_room_messaging.py
docker compose restart
docker compose ps
```

## Teardown

```bash
docker compose down
docker compose down -v
```

Use `down -v` to fully remove volumes and shared state.
