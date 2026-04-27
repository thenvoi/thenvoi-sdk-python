# OpenCode Examples for Thenvoi

Example for creating a Thenvoi agent that delegates to an OpenCode server.

## Prerequisites

1. **Install OpenCode**: `npm install -g opencode-ai`
2. **Start the OpenCode server**: `opencode serve --hostname=127.0.0.1 --port=4096`
3. **Thenvoi Platform** — create an external agent and add credentials to `agent_config.yaml`. The example uses the key `darter` by default (override with `AGENT_KEY`).
4. **Dependencies** — `uv sync --extra opencode`
5. **Environment variables** in `.env`:
   - `THENVOI_WS_URL`
   - `THENVOI_REST_URL`

## Configuration

In `agent_config.yaml`:

```yaml
darter:
  agent_id: "your-agent-id"
  api_key: "your-thenvoi-api-key"
```

## Optional environment variables

| Variable | Default | Purpose |
|---|---|---|
| `AGENT_KEY` | `darter` | Key in `agent_config.yaml` |
| `OPENCODE_BASE_URL` | `http://127.0.0.1:4096` | OpenCode server URL |
| `OPENCODE_PROVIDER_ID` | `opencode` | Provider ID for model lookup |
| `OPENCODE_MODEL_ID` | `minimax-m2.5-free` | Model to use |
| `OPENCODE_AGENT` | unset | Optional OpenCode agent role |
| `OPENCODE_APPROVAL_MODE` | `manual` | `manual` or `auto` tool approval |

## Examples

| File | Description |
|------|-------------|
| `01_basic_agent.py` | Minimal agent delegating to a running OpenCode server. |

## Running

```bash
uv run examples/opencode/01_basic_agent.py
```
