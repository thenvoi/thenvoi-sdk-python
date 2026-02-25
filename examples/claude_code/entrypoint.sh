#!/bin/bash
set -e

# Validate required mounts exist
if [ ! -d /workspace/repo ]; then
  echo "[entrypoint] ERROR: /workspace/repo is not mounted. Mount your repo as a volume." >&2
  exit 2
fi

# Mark workspace subdirs and extra dirs as git-safe (bind mounts have different ownership)
for dir in /workspace/repo ${GIT_SAFE_DIRS:+${GIT_SAFE_DIRS//,/ }}; do
  [ -d "$dir" ] && git config --global --add safe.directory "$dir"
done

exec "$@"
