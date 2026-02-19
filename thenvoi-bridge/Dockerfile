FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

# Convert SSH git URLs to HTTPS so submodules clone without SSH keys
RUN git config --global url."https://github.com/".insteadOf "git@github.com:"

# Install uv for reproducible dependency resolution
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv

WORKDIR /app

# Install SDK with bridge extras from the repo root using lockfile
COPY pyproject.toml uv.lock ./
COPY src/ src/
RUN uv pip install --system --no-cache ".[bridge]"

# --- Final stage ---
FROM python:3.11-slim

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

WORKDIR /app

# Copy bridge application
COPY thenvoi-bridge/ thenvoi-bridge/

RUN useradd --no-log-init --create-home --shell /bin/bash appuser

WORKDIR /app/thenvoi-bridge

EXPOSE 8080

USER appuser

# Override this CMD in your docker-compose.yml or Dockerfile
# to point to your own entry point that registers handlers.
# Example: CMD ["python", "run.py"]
CMD ["python", "-m", "bridge_core"]
