# Generic Claude SDK Docker Container
# Supports custom tools and volume mounts
#
# Usage:
#   docker build -t thenvoi-claude-sdk -f docker/claude-sdk/Dockerfile .
#   docker run -v ./my-agent/config:/app/config:ro \
#              -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
#              thenvoi-claude-sdk

FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20+ (required for Claude CLI)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI globally
RUN npm install -g @anthropic-ai/claude-code

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy SDK source
COPY pyproject.toml ./
COPY src/ ./src/

# Install with Claude SDK extras
RUN uv sync --extra claude_sdk

# Create mount point directories
RUN mkdir -p /app/config /app/tools /app/scripts /app/data

# Copy entrypoint and runner
COPY docker/claude-sdk/entrypoint.sh /entrypoint.sh
COPY docker/claude-sdk/runner.py /app/runner.py
RUN chmod +x /entrypoint.sh

# Environment defaults
ENV LOG_LEVEL=INFO
ENV CONFIG_PATH=/app/config/agent.yaml
ENV TOOLS_PATH=/app/tools
ENV SCRIPTS_PATH=/app/scripts
ENV DATA_PATH=/app/data

ENTRYPOINT ["/entrypoint.sh"]
CMD ["run"]
