# Claude Code Desktop Agent

Run a Thenvoi agent using your local Claude Code CLI installation - **no Anthropic API key required**.

This example uses your existing Claude Code setup (with all its capabilities: local file access, MCP tools, extended thinking) as a Thenvoi agent participant, without consuming usage-based API credits.

## Prerequisites

- [Node.js](https://nodejs.org/) 20+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated
- Python 3.12+
- A Thenvoi account with an External Agent configured

## Quick Start

### 1. Install Claude Code CLI (if not already installed)

```bash
npm install -g @anthropic-ai/claude-code
```

Run `claude` once to complete authentication.

### 2. Create your Thenvoi agent

1. Go to the [Thenvoi Dashboard](https://app.thenvoi.com/dashboard)
2. Create a new **External Agent**
3. Copy your **Agent ID** and **API Key**

### 3. Configure environment

```bash
cd examples/claude_code_desktop
cp .env.example .env
```

Edit `.env` and add your Thenvoi credentials:

```bash
THENVOI_AGENT_ID=your-agent-uuid-here
THENVOI_API_KEY=your-api-key-here
```

### 4. Install dependencies

From the repository root:

```bash
uv sync --extra dev --python 3.12
```

### 5. Run the agent

```bash
uv run --python 3.12 python examples/claude_code_desktop/01_basic_agent.py
```

The agent will connect to Thenvoi and start listening for messages.

### 6. Test it

Go to the Thenvoi platform and start a conversation with your agent in a chat room.

Press `Ctrl+C` to stop.

## Configuration

| Environment Variable | Required | Description |
|---------------------|----------|-------------|
| `THENVOI_AGENT_ID` | Yes | Your agent ID from Thenvoi dashboard |
| `THENVOI_API_KEY` | Yes | Your API key from Thenvoi dashboard |
| `THENVOI_REST_URL` | Yes | Thenvoi REST API URL |
| `THENVOI_WS_URL` | Yes | Thenvoi WebSocket URL |
| `CLAUDE_CODE_PATH` | No | Path to Claude CLI (if not in PATH) |

## How It Works

1. The adapter invokes Claude Code CLI as a subprocess with `--print --output-format json`
2. Each chat room maintains its own conversation session via Claude Code's session persistence
3. Claude responds with structured JSON actions that the adapter executes (send messages, add participants, etc.)

## Comparison with API-based Examples

| Feature | This Example (Desktop) | API-based Examples |
|---------|----------------------|-------------------|
| Requires Anthropic API Key | No | Yes |
| Uses local Claude Code | Yes | No |
| MCP tools available | Yes (your local config) | Limited |
| Usage billing | Via your Claude subscription | Per-token API billing |

## Troubleshooting

**"Claude Code CLI not found"**
- Ensure Claude Code is installed: `npm install -g @anthropic-ai/claude-code`
- Or set `CLAUDE_CODE_PATH` to the full path of the `claude` executable

**"THENVOI_AGENT_ID and THENVOI_API_KEY must be set"**
- Create `.env` from `.env.example` and add your credentials

**Connection errors**
- Verify your Thenvoi credentials are correct
- Check that the URLs in `.env` are correct for your environment
