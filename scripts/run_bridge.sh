#!/bin/bash
# Run the unified Letta Bridge (Letta + MCP + Adapter)
#
# Usage:
#   ./scripts/run_bridge.sh                    # Use default bridge_config.yaml
#   ./scripts/run_bridge.sh path/to/config.yaml  # Use specific config
#   ./scripts/run_bridge.sh -d                 # Run in background (detached)
#   ./scripts/run_bridge.sh config.yaml -d     # Specific config, detached
#
# Prerequisites:
#   - Docker installed and running
#   - OPENAI_API_KEY set in environment
#   - bridge_config.yaml with agent credentials

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse arguments
CONFIG_FILE=""
EXTRA_ARGS=""

for arg in "$@"; do
    if [[ "$arg" == -* ]]; then
        EXTRA_ARGS="$EXTRA_ARGS $arg"
    elif [[ -z "$CONFIG_FILE" ]]; then
        CONFIG_FILE="$arg"
    fi
done

# Default config file
if [[ -z "$CONFIG_FILE" ]]; then
    CONFIG_FILE="$PROJECT_ROOT/bridge_config.yaml"
fi

# Convert to absolute path if relative
if [[ ! "$CONFIG_FILE" = /* ]]; then
    CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" 2>/dev/null && pwd)/$(basename "$CONFIG_FILE")"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo ""
    echo "Usage: ./scripts/run_bridge.sh [config_file] [docker-compose options]"
    echo ""
    echo "Create from template:"
    echo "  cp bridge_config.yaml.example bridge_config.yaml"
    echo "  # Edit bridge_config.yaml with your credentials"
    exit 1
fi

echo "Using config: $CONFIG_FILE"

# Extract values using Python with project venv (has pyyaml installed)
extract_config() {
    uv run python -c "
import yaml
with open('$CONFIG_FILE') as f:
    config = yaml.safe_load(f)
path = '$1'.split('.')
val = config
for p in path:
    val = val.get(p, '') if isinstance(val, dict) else ''
print(val or '')
"
}

# Export for docker-compose
export THENVOI_API_KEY="$(extract_config 'agent.api_key')"
export THENVOI_BASE_URL="$(extract_config 'platform.rest_url')"
export CONFIG_FILE

# Check OPENAI_API_KEY
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY not set in environment"
    echo "Letta requires this for LLM calls"
    echo ""
fi

# Show agent info
AGENT_NAME="$(extract_config 'agent.name')"
AGENT_ID="$(extract_config 'agent.id')"
LETTA_MODE="$(extract_config 'letta.mode')"

echo ""
echo "Bridge Configuration:"
echo "  Agent: ${AGENT_NAME:-$AGENT_ID}"
echo "  Mode: ${LETTA_MODE:-per_room}"
echo "  Platform: $(extract_config 'platform.rest_url')"
echo ""

cd "$PROJECT_ROOT"
docker compose -f docker/docker-compose.bridge.yml up --build $EXTRA_ARGS
