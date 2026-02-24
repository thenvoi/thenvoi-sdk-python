#!/usr/bin/env bash
set -euo pipefail

resolve_wheel_path() {
  local value="$1"
  if [[ -z "${value}" ]]; then
    echo ""
    return 0
  fi

  if [[ -d "${value}" ]]; then
    if [[ ! -r "${value}" ]]; then
      echo "[codex-entrypoint] Wheel directory is not readable: ${value}" >&2
      exit 2
    fi
    local wheel
    wheel="$(find "${value}" -maxdepth 1 -type f -name '*.whl' 2>/dev/null | sort | head -n1)"
    if [[ -n "${wheel}" ]]; then
      echo "${wheel}"
      return 0
    fi

    echo "[codex-entrypoint] Wheel directory provided but no .whl files found: ${value}" >&2
    exit 2
  fi

  echo "${value}"
}

if [[ -n "${THENVOI_CLIENT_REST_WHEEL:-}" ]]; then
  thenvoi_wheel="$(resolve_wheel_path "${THENVOI_CLIENT_REST_WHEEL}")"
  if [[ -f "${thenvoi_wheel}" ]]; then
    echo "[codex-entrypoint] Installing local thenvoi-client-rest wheel: ${thenvoi_wheel}"
    uv pip install --python /app/.venv/bin/python --force-reinstall "${thenvoi_wheel}"
  else
    echo "[codex-entrypoint] THENVOI_CLIENT_REST_WHEEL is set but file does not exist: ${thenvoi_wheel}" >&2
    exit 2
  fi
fi

if [[ -n "${PHOENIX_CHANNELS_CLIENT_WHEEL:-}" ]]; then
  phoenix_wheel="$(resolve_wheel_path "${PHOENIX_CHANNELS_CLIENT_WHEEL}")"
  if [[ -f "${phoenix_wheel}" ]]; then
    echo "[codex-entrypoint] Installing local phoenix client wheel: ${phoenix_wheel}"
    uv pip install --python /app/.venv/bin/python --force-reinstall "${phoenix_wheel}"
  else
    echo "[codex-entrypoint] PHOENIX_CHANNELS_CLIENT_WHEEL is set but file does not exist: ${phoenix_wheel}" >&2
    exit 2
  fi
fi

exec "$@"
