# Claude SDK adapter for planner agent
FROM python:3.11-slim

# Install system deps + Node.js 20 (required by claude-agent-sdk)
RUN apt-get update && apt-get install -y \
    git curl openssh-client build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Create non-root user
RUN useradd -m -s /bin/bash appuser

WORKDIR /app

# Install SDK from PyPI with claude_sdk extra
RUN uv pip install --system "band-sdk[claude_sdk]" pyyaml python-dotenv

# Copy the unified agent runner + example files
COPY examples/run_agent.py /app/run_agent.py
COPY examples/coding_agents/tools /app/tools
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create workspace dirs
RUN mkdir -p /workspace/repo /workspace/notes /workspace/state /workspace/context \
    && chown -R appuser:appuser /workspace /app

USER appuser

# Claude SDK sandbox restricts the agent to CWD — must be /workspace/repo
WORKDIR /workspace/repo

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "/app/run_agent.py", "--example", "claude_sdk", "--agent", "planner", "--streaming"]
