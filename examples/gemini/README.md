# Gemini SDK Examples for Thenvoi

Example for creating a Thenvoi agent using the Google Generative AI (Gemini) SDK.

## Prerequisites

1. **Gemini API Key** — set `GEMINI_API_KEY` environment variable
2. **Thenvoi Platform** — create an external agent and add credentials to `agent_config.yaml` under the key `gemini_agent`
3. **Dependencies** — `uv sync --extra gemini`
4. **Environment variables** in `.env`:
   - `THENVOI_WS_URL` (e.g. `wss://app.band.ai/api/v1/socket/websocket`)
   - `THENVOI_REST_URL` (e.g. `https://app.band.ai`)

## Configuration

In `agent_config.yaml`:

```yaml
gemini_agent:
  agent_id: "your-agent-id"
  api_key: "your-thenvoi-api-key"
```

## Examples

| File | Description |
|------|-------------|
| `01_basic_agent.py` | Minimal agent using `gemini-2.5-flash` with platform tools. |

## Running

```bash
uv run examples/gemini/01_basic_agent.py
```

## Architecture

`GeminiAdapter` handles tool registration and function-calling loops automatically. Platform tools (`thenvoi_send_message`, `thenvoi_add_participant`, etc.) are registered as Gemini function declarations and executed in the tool loop.
