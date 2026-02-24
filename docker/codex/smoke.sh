#!/usr/bin/env bash
set -euo pipefail

echo "[smoke] versions"
python --version
uv --version
command -v codex >/dev/null 2>&1 || { echo "[smoke] FAIL: codex not found"; exit 1; }
codex --version
git --version
gh --version
jq --version
rg --version | head -n 1

echo "[smoke] codex home"
test -d "${CODEX_HOME}"

echo "[smoke] workspace write"
mkdir -p /workspace/notes
printf "# Codex Docker Smoke\n\n- %s\n" "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" > /workspace/notes/smoke.md
test -f /workspace/notes/smoke.md

if [ -d /workspace/repo/.git ]; then
  echo "[smoke] git worktree list"
  git -C /workspace/repo worktree list
fi

echo "[smoke] done"
