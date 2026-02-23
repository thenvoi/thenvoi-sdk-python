#!/bin/bash
set -e

# Entrypoint for Thenvoi Claude Code Agent
#
# Environment variables:
#   AGENT_CONFIG - Path to agent config YAML (default: /app/agent_config.yaml)
#   PROMPT_DIR   - Path to prompt profiles (default: /prompts)
#   WORKSPACE    - Path to workspace (default: /workspace/repo)

# Validate required config
AGENT_CONFIG="${AGENT_CONFIG:-/app/agent_config.yaml}"

if [ ! -f "$AGENT_CONFIG" ]; then
    echo "ERROR: Agent config not found at $AGENT_CONFIG"
    echo "Mount your agent_config.yaml to /app/agent_config.yaml"
    exit 1
fi

# Verify Claude Code CLI is available
if ! command -v claude &> /dev/null; then
    echo "ERROR: Claude Code CLI not found"
    echo "Ensure @anthropic-ai/claude-code is installed"
    exit 1
fi

# Set up environment
export WORKSPACE="${WORKSPACE:-/workspace/repo}"
export NOTES_DIR="${NOTES_DIR:-/workspace/notes}"
export PROMPT_DIR="${PROMPT_DIR:-/prompts}"

echo "Starting Thenvoi Claude Code Agent"
echo "  Config: $AGENT_CONFIG"
echo "  Workspace: $WORKSPACE"
echo "  Notes: $NOTES_DIR"
echo "  Prompts: $PROMPT_DIR"
echo ""

# Execute the command
exec "$@"
