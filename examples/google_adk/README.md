# Google ADK Examples for Thenvoi

Examples for creating Thenvoi agents using the Google Agent Development Kit (ADK) with Gemini models.

## Prerequisites

1. **Google Gemini API Key** — set `GOOGLE_API_KEY` (or `GOOGLE_GENAI_API_KEY`)
2. **Thenvoi Platform** — create an external agent and add credentials to `agent_config.yaml` under the key `google_adk_agent`
3. **Dependencies** — `uv sync --extra google_adk`
4. **Environment variables** in `.env`:
   - `THENVOI_WS_URL` (e.g. `wss://app.thenvoi.com/api/v1/socket/websocket`)
   - `THENVOI_REST_URL` (e.g. `https://app.thenvoi.com`)

## Configuration

In `agent_config.yaml`:

```yaml
google_adk_agent:
  agent_id: "your-agent-id"
  api_key: "your-thenvoi-api-key"
```

## Examples

| File | Description |
|------|-------------|
| `01_basic_agent.py` | Minimal agent using `gemini-2.5-flash` with platform tools only. |
| `02_custom_instructions.py` | Custom system prompt, model selection, and execution reporting. |
| `03_custom_tools.py` | Adds custom tools alongside platform tools via `additional_tools`. |

## Running

```bash
uv run examples/google_adk/01_basic_agent.py
uv run examples/google_adk/02_custom_instructions.py
uv run examples/google_adk/03_custom_tools.py
```

## Architecture

`GoogleADKAdapter` wraps ADK's `Runner` and `LlmAgent`. Platform tools are bridged into ADK's `BaseTool` system automatically. Custom tools defined with Pydantic schemas are registered the same way.
