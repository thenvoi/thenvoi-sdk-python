# Multi-stage Dockerfile for Thenvoi Python SDK examples
# Supports LangGraph and Claude SDK adapter examples
#
# Usage:
#   docker build -t thenvoi-sdk .
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

# Create stage for Claude SDK adapter
FROM base AS claude-sdk

# Install Node.js 20+ (required for Claude Code CLI)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Verify Node.js installation
RUN node --version && npm --version

# Install Claude Code CLI globally
RUN npm install -g @anthropic-ai/claude-code

# Verify Claude Code CLI installation
RUN claude --version

# Install Python dependencies with claude_sdk extras
RUN uv sync --extra claude_sdk

# Set default command
CMD ["uv", "run", "--extra", "claude_sdk", "python", "examples/claude_sdk/01_basic_agent.py"]

# Default stage is langgraph (for backwards compatibility)
FROM langgraph AS default
