# Dockerfile for Thenvoi Agent Runner
#
# Build:   docker compose build
# Run:     docker compose up
#
# This image provides a YAML-configured agent runner for the Thenvoi platform.
# Each agent is configured via its own YAML file in the agents/ directory.

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
# Note: uv.lock is optional - uv sync will create it if missing
COPY pyproject.toml ./
COPY src/ ./src/

# Install thenvoi SDK with claude_sdk extras
# Rewrite SSH URLs to HTTPS to avoid SSH auth issues in Docker
RUN git config --global url."https://github.com/".insteadOf "git@github.com:" && \
    uv sync --extra claude_sdk

# Copy runner and healthcheck
COPY examples/claude_code_apikey_docker/runner.py ./runner.py
COPY examples/claude_code_apikey_docker/healthcheck.py ./healthcheck.py

# Copy default tools (can be overridden by mounting custom tools)
COPY examples/claude_code_apikey_docker/tools/ ./tools/

# Config will be mounted at runtime
VOLUME /app/config

# Run the agent runner
CMD ["uv", "run", "--extra", "claude_sdk", "python", "runner.py"]
