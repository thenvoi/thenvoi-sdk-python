FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

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
    npm \
    openssh-client \
    ripgrep \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY examples ./examples
COPY docker/codex ./docker/codex

# Prefer HTTPS for GitHub URLs to avoid SSH key requirements in container builds.
# Trust bind-mounted repo to avoid "dubious ownership" errors.
RUN git config --global --add url."https://github.com/".insteadOf "git@github.com:" \
    && git config --global --add url."https://github.com/".insteadOf "ssh://git@github.com/" \
    && git config --global --add safe.directory /workspace/repo

# Normalize host-specific/local dev dependency sources for container builds.
# 1) Remove [tool.uv.sources] local absolute wheel mappings that are host-specific.
# 2) The git+https:// phoenix dependency works directly via the HTTPS config above.
# 3) Local wheel overrides can still be applied at runtime via entrypoint env vars.
RUN python - <<'PY'
from pathlib import Path
import re

path = Path("/app/pyproject.toml")
content = path.read_text()
content = re.sub(
    r"\n\[tool\.uv\.sources\]\n(?:[^\[\n].*\n?)+",
    "\n",
    content,
    count=1,
)
path.write_text(content)
PY

# Use publishable dependency resolution inside container.
RUN uv sync --extra codex

# Install Codex CLI from npm and verify app-server command is available.
RUN npm i -g @openai/codex@latest \
    && codex --version \
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
