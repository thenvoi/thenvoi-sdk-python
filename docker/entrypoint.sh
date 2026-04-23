#!/usr/bin/env bash
# Shared entrypoint for coding_agents containers.
# 1. Clones the repo if not already present (with file-lock to avoid races)
# 2. Optionally generates context files (structure.md, patterns.md, dependencies.md)
# 3. Starts the agent via run_agent.py
set -euo pipefail

# ── Repo initialisation ────────────────────────────────────────────────
LOCK_FILE="/workspace/state/.repo-init.lock"
LOCK_TIMEOUT="${REPO_INIT_LOCK_TIMEOUT_S:-120}"

init_repo() {
    local config="${AGENT_CONFIG:-/app/config/agent_config.yaml}"
    local key="${AGENT_KEY:-planner}"

    # Parse repo settings from agent_config.yaml using Python
    eval "$(python3 -c "
import yaml, sys
with open('${config}') as f:
    cfg = yaml.safe_load(f)
repo = cfg.get('${key}', {}).get('repo', {})
print(f\"REPO_URL={repo.get('url', '')}\")
print(f\"REPO_PATH={repo.get('path', '/workspace/repo')}\")
print(f\"REPO_BRANCH={repo.get('branch', 'main')}\")
print(f\"REPO_INDEX={str(repo.get('index', False)).lower()}\")
")"

    if [ -z "$REPO_URL" ]; then
        echo "[entrypoint] No repo.url configured — skipping clone"
        return
    fi

    if [ -d "$REPO_PATH/.git" ]; then
        echo "[entrypoint] Repo already exists at $REPO_PATH — skipping clone"
        return
    fi

    echo "[entrypoint] Cloning $REPO_URL → $REPO_PATH (branch: $REPO_BRANCH)"

    # SSH strict host-key checking
    if [ "${GIT_SSH_STRICT_HOST_KEY_CHECKING:-true}" = "true" ]; then
        export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=yes"
    else
        export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no"
    fi

    git clone --branch "$REPO_BRANCH" --single-branch "$REPO_URL" "$REPO_PATH"
    echo "[entrypoint] Clone complete"

    # Optional: generate context files
    if [ "$REPO_INDEX" = "true" ]; then
        echo "[entrypoint] Generating context files..."
        mkdir -p /workspace/context
        # Structure: list of files
        (cd "$REPO_PATH" && find . -type f \
            ! -path './.git/*' ! -path './node_modules/*' ! -path './.venv/*' \
            ! -path './__pycache__/*' ! -name '*.pyc' \
            | sort) > /workspace/context/structure.md
        # Dependencies: package files
        for f in package.json pyproject.toml requirements.txt Cargo.toml go.mod; do
            if [ -f "$REPO_PATH/$f" ]; then
                echo "## $f" >> /workspace/context/dependencies.md
                echo '```' >> /workspace/context/dependencies.md
                cat "$REPO_PATH/$f" >> /workspace/context/dependencies.md
                echo '```' >> /workspace/context/dependencies.md
                echo "" >> /workspace/context/dependencies.md
            fi
        done
        # Patterns: look for common patterns
        echo "# Code Patterns" > /workspace/context/patterns.md
        if [ -f "$REPO_PATH/pyproject.toml" ]; then
            echo "- Python project (pyproject.toml)" >> /workspace/context/patterns.md
        fi
        if [ -f "$REPO_PATH/package.json" ]; then
            echo "- Node.js project (package.json)" >> /workspace/context/patterns.md
        fi
        if [ -d "$REPO_PATH/src" ]; then
            echo "- Uses src/ layout" >> /workspace/context/patterns.md
        fi
        if [ -d "$REPO_PATH/tests" ] || [ -d "$REPO_PATH/test" ]; then
            echo "- Has test directory" >> /workspace/context/patterns.md
        fi
        echo "[entrypoint] Context files generated in /workspace/context/"
    fi
}

# Use flock for cross-container locking
mkdir -p /workspace/state
(
    flock -w "$LOCK_TIMEOUT" 200 || { echo "[entrypoint] Timed out waiting for repo init lock"; exit 1; }
    init_repo
) 200>"$LOCK_FILE"

# Create notes dir
mkdir -p /workspace/notes

# ── Symlink agent_config.yaml into CWD ──────────────────────────────────
# load_agent_config() looks for agent_config.yaml in the current working dir.
# The compose file mounts it at AGENT_CONFIG, so we symlink it to $PWD.
CONFIG_FILE="${AGENT_CONFIG:-/app/config/agent_config.yaml}"
if [ -f "$CONFIG_FILE" ] && [ ! -f "$PWD/agent_config.yaml" ]; then
    ln -sf "$CONFIG_FILE" "$PWD/agent_config.yaml"
    echo "[entrypoint] Symlinked $CONFIG_FILE → $PWD/agent_config.yaml"
fi

# ── Start agent ─────────────────────────────────────────────────────────
exec "$@"
