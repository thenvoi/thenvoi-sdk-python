#!/bin/bash
set -e

# Mark workspace subdirs and extra dirs as git-safe (bind mounts have different ownership)
for dir in /workspace/repo ${GIT_SAFE_DIRS//,/ }; do
  [ -d "$dir" ] && git config --global --add safe.directory "$dir"
done

exec "$@"
