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

# Configure git to prefer HTTPS over SSH via env vars
# (env vars are always read by git, unlike global config which may be skipped)
ENV GIT_CONFIG_COUNT=2 \
    GIT_CONFIG_KEY_0=url.https://github.com/.insteadOf \
    GIT_CONFIG_VALUE_0=git@github.com: \
    GIT_CONFIG_KEY_1=url.https://github.com/.insteadOf \
    GIT_CONFIG_VALUE_1=ssh://git@github.com/

# Install Claude Code CLI globally
RUN npm install -g @anthropic-ai/claude-code

# Copy SDK source and install with pip
COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --break-system-packages .

# Copy runner script
COPY examples/claude_code/runner.py ./runner.py

# Create workspace directories
RUN mkdir -p /workspace/repo /workspace/notes /prompts

# Trust bind-mounted repo to avoid "dubious ownership" errors
RUN git config --global --add safe.directory /workspace/repo

# Use existing node user (UID 1000) for security
RUN chown -R node:node /app /workspace /prompts

USER node

# Set working directory to workspace for file operations
WORKDIR /workspace/repo

CMD ["python3", "/app/runner.py"]
