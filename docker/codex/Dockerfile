# Codex adapter for reviewer agent
FROM python:3.11-slim

# Install system deps + Node.js 20 (required by Codex CLI)
RUN apt-get update && apt-get install -y \
    git curl openssh-client build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Codex CLI globally
RUN npm install -g @openai/codex

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Create non-root user
RUN useradd -m -s /bin/bash codex

WORKDIR /app

# Install SDK from PyPI with codex extra
RUN uv pip install --system "band-sdk[codex]" pyyaml python-dotenv

# Copy the unified agent runner + example files
COPY examples/run_agent.py /app/run_agent.py
COPY examples/coding_agents/tools /app/tools
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create workspace dirs
RUN mkdir -p /workspace/repo /workspace/notes /workspace/state /workspace/context \
    && chown -R codex:codex /workspace /app

USER codex

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "/app/run_agent.py", "--example", "codex", "--agent", "reviewer", \
     "--codex-transport", "stdio", "--codex-sandbox", "external-sandbox"]
