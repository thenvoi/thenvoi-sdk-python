# ACP (Agent Client Protocol) Examples

Bidirectional integration between Thenvoi and editors that speak ACP (Zed, Cursor, JetBrains, Neovim).

## Two directions

| Direction | Role | Example files |
|---|---|---|
| **Server** (editor → Thenvoi) | Editor connects to Thenvoi as an ACP agent. Thenvoi routes prompts to platform peers. | `01_basic_acp_server.py`, `03_acp_server_with_routing.py`, `05_acp_server_push_notifications.py`, `07_jetbrains_server.py` |
| **Client** (Thenvoi → external ACP agent) | Thenvoi spawns an external ACP agent (Codex CLI, Claude Code, etc.) as a peer. | `02_acp_client.py`, `04_acp_client_rich_streaming.py`, `06_cursor_client.py` |

## Prerequisites

1. Install: `uv sync --extra acp` (or `pip install thenvoi-sdk[acp]`)
2. Create an agent on the Thenvoi platform and add credentials to `agent_config.yaml`
3. Set environment variables in `.env`:
   - `THENVOI_WS_URL` (e.g. `wss://app.band.ai/api/v1/socket/websocket`)
   - `THENVOI_REST_URL` (e.g. `https://app.band.ai`)
   - `THENVOI_API_KEY` and `THENVOI_AGENT_ID` — only for the JetBrains/CLI path that injects creds via env

For client examples, also install the external ACP agent binary (e.g. `@openai/codex`, `@anthropic-ai/claude-code`).

## Examples

| File | What it shows |
|---|---|
| `01_basic_acp_server.py` | Minimal ACP server — editor sends prompt → Thenvoi peer replies. |
| `02_acp_client.py` | Minimal ACP client — Thenvoi spawns an external ACP agent as a peer. |
| `03_acp_server_with_routing.py` | `AgentRouter` — slash commands (`/codex fix bug`) route to specific peers. |
| `04_acp_client_rich_streaming.py` | Streams rich chunks (tool calls, thoughts) from the spawned agent back to the room. |
| `05_acp_server_push_notifications.py` | `ACPPushHandler` — unsolicited `session_update` notifications when peers act. |
| `06_cursor_client.py` | Cursor-specific client configuration. |
| `07_jetbrains_server.py` | JetBrains IDE configuration (`~/.jetbrains/acp.json`). |

## Running

Server examples are launched by the editor (not by you directly). You configure the editor to spawn one as an ACP agent:

```jsonc
// Zed settings.json
{
  "agent_servers": {
    "Thenvoi": {
      "type": "custom",
      "command": "uv run examples/acp/01_basic_acp_server.py"
    }
  }
}
```

Client examples are runnable standalone:

```bash
uv run examples/acp/02_acp_client.py
```

See the docstring at the top of each file for editor-specific setup.

## CLI entry point

For production use, the SDK ships a `thenvoi-acp` CLI that wraps the server pattern:

```bash
thenvoi-acp --agent-id <AGENT_ID>
# or via env: THENVOI_AGENT_ID=... THENVOI_API_KEY=... thenvoi-acp
```

Use the CLI when you don't need to customize routing or push-handling in Python.
