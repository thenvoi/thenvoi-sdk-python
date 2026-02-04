# Dockerfile for Thenvoi Claude SDK Agent Runner
#
# Build:   docker build -t thenvoi-claude-sdk .
# Run:     docker compose up
#
# This image provides a YAML-configured agent runner for the Thenvoi platform.

FROM python:3.11-slim

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

# Copy SDK source and install
# Note: uv.lock must exist in the repo root for reproducible builds.
# If building from a partial clone, run `uv lock` first to generate it.
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install thenvoi SDK with claude_sdk extras
# Rewrite SSH URLs to HTTPS to avoid SSH auth issues in Docker
RUN git config --global url."https://github.com/".insteadOf "git@github.com:" && \
    uv sync --extra claude_sdk

# Copy runner, healthcheck, and example tools
COPY examples/claude_code_apikey_docker/runner.py ./runner.py
COPY examples/claude_code_apikey_docker/healthcheck.py ./healthcheck.py
COPY examples/claude_code_apikey_docker/tools/ ./tools/

# Config will be mounted at runtime
VOLUME /app/config

# Run the agent runner
CMD ["uv", "run", "--extra", "claude_sdk", "python", "runner.py"]
