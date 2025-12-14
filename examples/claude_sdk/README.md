# Claude Agent SDK Examples

This directory contains examples of using the Claude Agent SDK with the Thenvoi platform.

## Prerequisites

### 1. Node.js and Claude Code CLI

The Claude Agent SDK requires the Claude Code CLI to be installed:

```bash
# Install Node.js 20+
# On macOS:
brew install node@20

# On Ubuntu/Debian:
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Claude Code CLI globally
npm install -g @anthropic-ai/claude-code

# Verify installation
claude --version
```

### 2. Python Dependencies

```bash
# Install with claude_sdk extras
pip install thenvoi[claude_sdk]

# Or using uv
uv pip install thenvoi[claude_sdk]
```

### 3. Environment Variables

```bash
export THENVOI_AGENT_ID="your-agent-id"
export THENVOI_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## Examples

### 01_basic_agent.py

Basic agent with standard configuration:

```bash
python 01_basic_agent.py
```

Features:
- Standard Claude Sonnet model
- Platform tool integration
- Execution reporting

### 02_extended_thinking.py

Agent with extended thinking enabled for complex reasoning:

```bash
python 02_extended_thinking.py
```

Features:
- Extended thinking with 10,000 token budget
- Thought events reported to chat
- Ideal for complex problem-solving

## Architecture

```
ThenvoiClaudeSDKAgent
├── thenvoi: ThenvoiAgent           # Platform coordinator
├── session_manager: ClaudeSessionManager
│   └── _sessions: Dict[room_id, ClaudeSDKClient]
├── _mcp_server                      # MCP tools (stubs)
│
└── _handle_message(msg, tools)
    ├── Get/create session for room
    ├── Hydrate history (first message)
    ├── Inject participants (on change)
    ├── client.query(message)
    └── _process_response()
        ├── TextBlock → collect
        ├── ThinkingBlock → report
        └── ToolUseBlock → execute
```

## Key Differences from Anthropic SDK Example

| Aspect | ThenvoiAnthropicAgent | ThenvoiClaudeSDKAgent |
|--------|----------------------|----------------------|
| Library | `anthropic` | `claude-agent-sdk` |
| History | Manual `_message_history` | SDK manages automatically |
| Tools | JSON schema | MCP `@tool` decorator |
| Response | Single response | Async streaming |
| Thinking | Not supported | `max_thinking_tokens` |
| Sessions | No manager | `ClaudeSessionManager` |

## MCP Tool Integration

Tools are defined as MCP stubs in `tools.py`. The actual execution happens in the agent via `AgentTools`:

```python
# MCP tool name → AgentTools method
"mcp__thenvoi__send_message" → tools.send_message()
"mcp__thenvoi__send_event" → tools.send_event()
"mcp__thenvoi__add_participant" → tools.add_participant()
# etc.
```

## Phase 2: Real MCP Server

In a future update, we'll add an example showing connection to a real Thenvoi MCP server instead of in-process stubs. This will enable:
- Direct tool execution without interception
- Real-time platform integration
- Reduced latency

## Troubleshooting

### "claude: command not found"
Install the Claude Code CLI:
```bash
npm install -g @anthropic-ai/claude-code
```

### "ModuleNotFoundError: No module named 'claude_agent_sdk'"
Install the claude_sdk extras:
```bash
pip install thenvoi[claude_sdk]
```

### Session not found for room
Ensure the agent is properly connected to the Thenvoi platform and has joined the room.
