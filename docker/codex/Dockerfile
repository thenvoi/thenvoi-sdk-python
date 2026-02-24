# ── Build stage ───────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY examples ./examples
COPY docker/codex/strip_uv_sources.py /tmp/strip_uv_sources.py

# Prefer HTTPS for GitHub URLs to avoid SSH key requirements in container builds.
RUN git config --global --add url."https://github.com/".insteadOf "git@github.com:" \
    && git config --global --add url."https://github.com/".insteadOf "ssh://git@github.com/"

# Strip host-specific [tool.uv.sources] so uv resolves from the registry.
RUN python /tmp/strip_uv_sources.py /app/pyproject.toml

# Install Python dependencies and Codex CLI.
RUN uv sync --extra codex
RUN npm i -g @openai/codex@latest

# ── Runtime stage ─────────────────────────────────────────────────────────
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# Runtime-only system dependencies (no build tools).
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    git-lfs \
    gh \
    jq \
    less \
    nodejs \
    openssh-client \
    ripgrep \
    && rm -rf /var/lib/apt/lists/*

# Runtime entrypoint installs optional wheel overrides via `uv pip install`.
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy Python venv and SDK source from builder.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/examples /app/examples

# Copy global Codex CLI from builder.
COPY --from=builder /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=builder /usr/local/bin/codex /usr/local/bin/codex

# Copy entrypoint and smoke test.
COPY docker/codex/entrypoint.sh /app/docker/codex/entrypoint.sh
COPY docker/codex/smoke.sh /app/docker/codex/smoke.sh

# Trust bind-mounted repo to avoid "dubious ownership" errors.
# Additional dirs can be configured at runtime via GIT_SAFE_DIRS env var.
RUN git config --global --add safe.directory /workspace/repo

# Verify Codex CLI is functional.
RUN codex --version \
    && codex app-server --help >/dev/null

RUN useradd -m -u 1000 -s /bin/bash codex \
    && mkdir -p /workspace/repo /workspace/notes /prompts /state /home/codex/.codex \
    && chown -R codex:codex /app /workspace /prompts /state /home/codex

USER codex
ENV HOME=/home/codex
ENV CODEX_HOME=/home/codex/.codex
ENV PATH="/home/codex/.local/bin:/app/.venv/bin:${PATH}"

WORKDIR /workspace/repo

ENTRYPOINT ["bash", "/app/docker/codex/entrypoint.sh"]
CMD ["/app/.venv/bin/python", "examples/run_agent.py", "--example", "codex", "--agent", "darter", "--codex-transport", "stdio", "--no-codex-turn-task-markers"]
