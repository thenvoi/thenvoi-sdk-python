#!/bin/bash
set -e

# Load .env if exists (properly handles comments and quotes)
if [ -f /app/config/.env ]; then
    set -a
    source /app/config/.env
    set +a
fi

case "$1" in
    run)
        # Run the default agent runner
        exec uv run --extra claude_sdk python /app/runner.py
        ;;
    custom)
        # Run user's custom script
        shift
        exec uv run --extra claude_sdk python /app/scripts/"$@"
        ;;
    shell)
        # Interactive shell
        exec /bin/bash
        ;;
    *)
        # Pass through to any command
        exec "$@"
        ;;
esac
