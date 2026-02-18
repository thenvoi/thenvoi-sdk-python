#!/bin/bash
# Force reinstall thenvoi-client-rest from local wheel
# Usage: ./scripts/reinstall-rest-client.sh
#
# Requires: scripts/config.yaml (copy from scripts/config.example.yaml)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Please copy scripts/config.example.yaml to scripts/config.yaml and configure it."
    exit 1
fi

# Parse wheel_dir from YAML config (simple grep for key: "value" format)
WHEEL_DIR=$(grep -E '^wheel_dir:' "$CONFIG_FILE" | sed -E 's/^wheel_dir:[[:space:]]*"?([^"]*)"?[[:space:]]*$/\1/')

if [ -z "$WHEEL_DIR" ]; then
    echo "Error: wheel_dir not configured in $CONFIG_FILE"
    exit 1
fi

if [ ! -d "$WHEEL_DIR" ]; then
    echo "Error: Wheel directory not found: $WHEEL_DIR"
    exit 1
fi

# Find the latest wheel
LATEST_WHEEL=$(ls -t "$WHEEL_DIR"/thenvoi_client_rest-*.whl 2>/dev/null | head -1)

if [ -z "$LATEST_WHEEL" ]; then
    echo "Error: No wheel found in $WHEEL_DIR"
    exit 1
fi

WHEEL_NAME=$(basename "$LATEST_WHEEL")
echo "Found wheel: $WHEEL_NAME"

# Extract version from wheel name
VERSION=$(echo "$WHEEL_NAME" | sed -E 's/thenvoi_client_rest-([^-]+)-.*/\1/')
echo "Version: $VERSION"

# Update pyproject.toml with the new wheel path
sed -i '' "s|thenvoi_client_rest-[^-]*-py3-none-any.whl|$WHEEL_NAME|g" pyproject.toml

# Force clean uv cache
echo "Cleaning uv cache..."
uv cache clean --force 2>/dev/null || true

# Remove lockfile to force re-resolution
echo "Removing uv.lock..."
rm -f uv.lock

# Remove venv to ensure clean state
echo "Removing .venv..."
rm -rf .venv

# Sync dependencies
echo "Syncing dependencies..."
uv sync --extra dev

echo "Done! Installed thenvoi-client-rest $VERSION"
