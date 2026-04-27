# AgentCore Examples

Examples for integrating AWS Bedrock AgentCore with Thenvoi via the `thenvoi-bridge` application.

These examples are **dev tooling** — they do not use the SDK's `Agent` abstraction directly. They target the bridge layer that routes messages between Thenvoi rooms and a Bedrock AgentCore runtime.

## Files

| File | Purpose |
|---|---|
| `agentcore_llm_server.py` | Standalone MCP server (uses `FastMCP` + `anthropic`) that wraps Claude as a chat tool. Deploy as an AgentCore runtime or run locally to develop against. |
| `run_agentcore.py` | Manual test runner for the `thenvoi-bridge` AgentCore handler. Imports from a sibling `thenvoi-bridge/` checkout — **not** a standalone example. |

## `agentcore_llm_server.py` — Running locally

```bash
ANTHROPIC_API_KEY=sk-... uv run examples/agentcore/agentcore_llm_server.py
```

Serves MCP over HTTP on port 8000. Deploy as a container to AgentCore by packaging and registering via `create_agent_runtime` in `bedrock-agentcore-control`.

Env vars:
- `ANTHROPIC_API_KEY` (required)
- `ANTHROPIC_MODEL` (default: `claude-sonnet-4-5-20250929`)
- `SYSTEM_PROMPT` (optional override)

## `run_agentcore.py` — Local bridge test

Requires a sibling `thenvoi-bridge/` checkout alongside this repo (`..//thenvoi-bridge`). Runs the bridge with the AgentCore handler wired up.

Requires `.env.test` (or `ENV_FILE`) with:
- `THENVOI_AGENT_ID`, `THENVOI_API_KEY`, `AGENT_MAPPING`
- `AGENTCORE_RUNTIME_ARN`, `AWS_DEFAULT_REGION`
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- `AGENTCORE_MCP_TOOL` (optional — name of the MCP tool to invoke)

```bash
uv run python examples/agentcore/run_agentcore.py
```
