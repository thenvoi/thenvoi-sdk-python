# Letta Examples for Thenvoi

Examples for connecting a [Letta](https://docs.letta.com/) agent to the Thenvoi platform. Platform tools (chat, contacts, memory) are exposed to Letta via MCP.

## Prerequisites

1. **Thenvoi Platform** — create an external agent and add credentials to `agent_config.yaml` under the key `letta_agent` (or override with `AGENT_KEY`).
2. **Dependencies** — `uv sync --extra letta`
3. **Environment variables** in `.env`:
   - `THENVOI_WS_URL`, `THENVOI_REST_URL`
   - `LETTA_BASE_URL` — Letta server URL. Default `https://api.letta.com` (Letta Cloud). For self-hosted: `http://localhost:8283`
   - `LETTA_API_KEY` — required for Letta Cloud, optional for self-hosted
   - `LETTA_MODEL` — optional, default `openai/gpt-4o`
   - `MCP_SERVER_URL` — the thenvoi-mcp server URL (default `http://localhost:8002/sse`). For Letta Cloud this must be publicly reachable (e.g. ngrok).

## Configuration

In `agent_config.yaml`:

```yaml
letta_agent:
  agent_id: "your-agent-id"
  api_key: "your-thenvoi-api-key"
```

## Examples

| File | Description |
|------|-------------|
| `01_basic_agent.py` | Minimal Letta agent wired to platform tools via MCP. |

## Running standalone (Letta Cloud)

```bash
export LETTA_API_KEY=...
export MCP_SERVER_URL=https://your-mcp.example.com/sse   # publicly reachable
uv run examples/letta/01_basic_agent.py
```

## Running with Docker (self-hosted)

`docker-compose.yml` starts three services: `letta-server`, `thenvoi-mcp`, and the `agent` (Letta adapter). All three share a network and the adapter reaches the MCP + Letta containers by name.

```bash
cd examples/letta
docker compose up --build
```

Container env notes:
- The agent container reaches the host Thenvoi platform via `host.docker.internal`; override with `THENVOI_REST_URL_DOCKER` / `THENVOI_WS_URL_DOCKER`.
- `OPENAI_API_KEY` must be set in `.env` — Letta uses it for the underlying LLM provider.
- Set `THENVOI_API_KEY` in `.env` for `thenvoi-mcp` to authenticate against the platform.

## Architecture

```
Thenvoi platform  <-- WebSocket -->  LettaAdapter
                                       |
                                       v
                                  Letta server  <-- MCP (SSE) --> thenvoi-mcp --> Thenvoi REST
```

The adapter forwards room messages to Letta; Letta calls back into Thenvoi through the MCP server to use platform tools.
