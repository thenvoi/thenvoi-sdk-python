# Multi-stage Dockerfile for Thenvoi Python SDK examples
# Supports LangGraph and Claude SDK adapter examples
#
# Usage:
#   docker build -t thenvoi-sdk .
#   docker build --target claude_sdk -t thenvoi-claude-sdk .
#   docker compose up langgraph-01-simple
#   docker compose up claude-sdk-01-basic

FROM python:3.11-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY examples/ ./examples/

# Create stage for LangGraph adapter
FROM base AS langgraph

# Install dependencies with langgraph extras (fresh resolution from pyproject.toml)
RUN uv sync --extra langgraph

# Set default command
CMD ["uv", "run", "--extra", "langgraph", "python", "examples/langgraph/01_simple_agent.py"]

# =============================================================================
# Claude SDK Stage
# =============================================================================
# Requires Node.js 20+ for Claude Code CLI (@anthropic-ai/claude-code)
#
# Features:
#   - Full Claude Agent SDK support
#   - Extended thinking capabilities
#   - MCP tool integration
#   - Volume mounting for custom scripts
#
# Usage:
#   docker build --target claude_sdk -t thenvoi-claude-sdk .
#   docker run -e ANTHROPIC_API_KEY -e THENVOI_AGENT_ID -e THENVOI_API_KEY thenvoi-claude-sdk
#
# With custom scripts:
#   docker run -v ./my_scripts:/app/user_scripts:rw \
#     -e SCRIPT_PATH=/app/user_scripts/my_agent.py \
#     -e ANTHROPIC_API_KEY -e THENVOI_AGENT_ID -e THENVOI_API_KEY \
#     thenvoi-claude-sdk
# =============================================================================

FROM base AS claude_sdk

# Install Node.js 20+ (required for Claude Code CLI)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI globally
RUN npm install -g @anthropic-ai/claude-code

# Verify installations
RUN node --version && npm --version && claude --version

# Install dependencies with claude_sdk extras
RUN uv sync --extra claude_sdk

# Create directory for user-mounted scripts
RUN mkdir -p /app/user_scripts && chmod 755 /app/user_scripts

# Environment variable to specify custom script path (optional)
ENV SCRIPT_PATH=""

# Default command runs the basic example, or custom script if SCRIPT_PATH is set
CMD ["sh", "-c", "if [ -n \"$SCRIPT_PATH\" ] && [ -f \"$SCRIPT_PATH\" ]; then uv run --extra claude_sdk python \"$SCRIPT_PATH\"; else uv run --extra claude_sdk python examples/claude_sdk/01_basic_agent.py; fi"]

# Default stage is langgraph (for backwards compatibility)
FROM langgraph AS default
