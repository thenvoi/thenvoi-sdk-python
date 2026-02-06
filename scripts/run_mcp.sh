#!/bin/bash
# Run MCP server for a specific agent from agent_config.yaml
#
# Usage:
#   ./scripts/run_mcp.sh <agent_key> [port]
#
# Examples:
#   ./scripts/run_mcp.sh darter         # Run MCP for darter agent on port 8002
#   ./scripts/run_mcp.sh darter 8003    # Run on custom port
#   ./scripts/run_mcp.sh letta_agent    # Run MCP for letta_agent
#
# Prerequisites:
#   - Docker installed and running
#   - agent_config.yaml exists with agent credentials
#   - docker/Dockerfile.mcp exists

set -e

AGENT_KEY=${1:-darter}
PORT=${2:-8002}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

# Check if agent_config.yaml exists
if [ ! -f "agent_config.yaml" ]; then
    echo "Error: agent_config.yaml not found in $PROJECT_DIR"
    exit 1
fi

# Check if Dockerfile.mcp exists
if [ ! -f "docker/Dockerfile.mcp" ]; then
    echo "Error: docker/Dockerfile.mcp not found"
    exit 1
fi

# Extract API key from agent_config.yaml using Python
API_KEY=$(python3 -c "
import yaml
with open('agent_config.yaml') as f:
    config = yaml.safe_load(f)
agent = config.get('$AGENT_KEY', {})
print(agent.get('api_key', ''))
")

if [ -z "$API_KEY" ]; then
    echo "Error: Agent '$AGENT_KEY' not found in agent_config.yaml or has no api_key"
    echo ""
    echo "Available agents:"
    python3 -c "
import yaml
with open('agent_config.yaml') as f:
    config = yaml.safe_load(f)
for key in config:
    print(f'  - {key}')
"
    exit 1
fi

CONTAINER_NAME="mcp-agent-$AGENT_KEY"

# Stop old container if running
echo "Stopping existing container '$CONTAINER_NAME' (if any)..."
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true

# Build the image
echo "Building thenvoi-mcp image..."
docker build -f docker/Dockerfile.mcp -t thenvoi-mcp .

# Run the container
echo "Starting MCP server for '$AGENT_KEY' on port $PORT..."
docker run -d --name "$CONTAINER_NAME" \
  -p "$PORT:8000" \
  -e "THENVOI_API_KEY=$API_KEY" \
  -e "THENVOI_BASE_URL=${THENVOI_BASE_URL:-http://host.docker.internal:4000}" \
  --restart unless-stopped \
  thenvoi-mcp

echo ""
echo "MCP server for '$AGENT_KEY' is running!"
echo "  Container: $CONTAINER_NAME"
echo "  Port: $PORT"
echo "  SSE endpoint: http://localhost:$PORT/sse"
echo ""
echo "To view logs: docker logs -f $CONTAINER_NAME"
echo "To stop: docker stop $CONTAINER_NAME"
