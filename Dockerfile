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
# Claude SDK Stage - Pre-built image for users
# =============================================================================
# Users mount their agents folder and run - no build needed!
#
# Usage:
#   docker run -v ./agents:/app/user_agents \
#     -e AGENT_CONFIG=/app/user_agents/my_agent.yaml \
#     -e ANTHROPIC_API_KEY -e THENVOI_WS_URL -e THENVOI_REST_URL \
#     ghcr.io/thenvoi/thenvoi-claude-sdk:latest
# =============================================================================

FROM base AS claude_sdk

# Install Node.js 20+ (required for Claude Code CLI)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI globally
RUN npm install -g @anthropic-ai/claude-code

# Install dependencies with claude_sdk extras
RUN uv sync --extra claude_sdk

# Create directory for user-mounted content
RUN mkdir -p /app/user_agents/tools

# Run agent from YAML config
CMD ["uv", "run", "--extra", "claude_sdk", "python", "-m", "thenvoi.run_agent"]

# Default stage is langgraph (for backwards compatibility)
FROM langgraph AS default
