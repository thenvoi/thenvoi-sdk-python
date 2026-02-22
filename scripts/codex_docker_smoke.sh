#!/usr/bin/env bash
set -euo pipefail

docker compose -f docker/codex/docker-compose.yml run --rm codex-agent \
  bash /app/docker/codex/smoke.sh
