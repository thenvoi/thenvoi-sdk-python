# Dockerfile for Thenvoi Claude Code Agent
#
# Build:   docker compose build
# Run:     docker compose up
#
# This image runs Claude Code CLI as a Thenvoi agent with workspace access.
# Requires Claude MAX subscription (no API key needed for CLI).

FROM node:20-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    bash \
    ripgrep \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Configure git to prefer HTTPS over SSH (Codex pattern)
# This avoids SSH auth issues in Docker builds
RUN git config --global url."https://github.com/".insteadOf "git@github.com:" && \
    git config --global url."https://github.com/".insteadOf "ssh://git@github.com/"

# Install Claude Code CLI globally
RUN npm install -g @anthropic-ai/claude-code

# Install uv package manager (pinned version for reproducibility)
RUN curl -LsSf https://astral.sh/uv/0.5.18/install.sh | sh && \
    cp /root/.local/bin/uv /usr/local/bin/uv

# Copy SDK source and install
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install thenvoi SDK (no claude_sdk extra - we use CLI directly)
RUN uv sync

# Copy runner script
COPY docker/claude_code/runner.py ./runner.py

# Copy entrypoint
COPY docker/claude_code/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create workspace directories
RUN mkdir -p /workspace/repo /workspace/notes /prompts

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /workspace /prompts

USER appuser

# Set working directory to workspace for file operations
WORKDIR /workspace/repo

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uv", "run", "python", "/app/runner.py"]
