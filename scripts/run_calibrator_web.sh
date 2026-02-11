#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
FRONTEND_DIR="$ROOT_DIR/frontend/calibrator-ui"
BACKEND_REQUIREMENTS_FILE="${BACKEND_REQUIREMENTS_FILE:-$ROOT_DIR/backend/requirements-web.txt}"
PY_DEPS_STAMP="$VENV_DIR/.calibrator_python_deps_ready"
NPM_DEPS_STAMP="$FRONTEND_DIR/.calibrator_npm_deps_ready"
SKIP_DEPENDENCY_INSTALL="${SKIP_DEPENDENCY_INSTALL:-0}"
ENABLE_BACKEND_RELOAD="${ENABLE_BACKEND_RELOAD:-0}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "$PYTHON_BIN is required." >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required." >&2
  exit 1
fi

if [[ ! -f "$BACKEND_REQUIREMENTS_FILE" ]]; then
  echo "Missing backend dependency list: $BACKEND_REQUIREMENTS_FILE" >&2
  exit 1
fi

ensure_python_deps() {
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "[calibrator-web] Creating Python virtual environment at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi

  local venv_python="$VENV_DIR/bin/python"
  local venv_pip="$VENV_DIR/bin/pip"
  if [[ ! -x "$venv_python" || ! -x "$venv_pip" ]]; then
    echo "Virtual environment is missing python/pip binaries: $VENV_DIR" >&2
    exit 1
  fi

  if [[ ! -f "$PY_DEPS_STAMP" || "$BACKEND_REQUIREMENTS_FILE" -nt "$PY_DEPS_STAMP" ]]; then
    echo "[calibrator-web] Installing backend Python dependencies"
    "$venv_python" -m pip install --upgrade pip
    "$venv_pip" install -r "$BACKEND_REQUIREMENTS_FILE"
    touch "$PY_DEPS_STAMP"
  else
    echo "[calibrator-web] Backend Python dependencies already installed"
  fi
}

ensure_frontend_deps() {
  pushd "$FRONTEND_DIR" >/dev/null

  local needs_install=0
  if [[ ! -d node_modules ]]; then
    needs_install=1
  elif [[ ! -f "$NPM_DEPS_STAMP" ]]; then
    needs_install=1
  elif [[ package-lock.json -nt "$NPM_DEPS_STAMP" || package.json -nt "$NPM_DEPS_STAMP" ]]; then
    needs_install=1
  fi

  if [[ "$needs_install" -eq 1 ]]; then
    echo "[calibrator-web] Installing frontend npm dependencies"
    npm install
    touch "$NPM_DEPS_STAMP"
  else
    echo "[calibrator-web] Frontend npm dependencies already installed"
  fi

  popd >/dev/null
}

cleanup() {
  if [[ -n "${FRONTEND_PID:-}" ]]; then
    kill "$FRONTEND_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

# Kill process on backend port if in use (avoids "address already in use").
# Only kills BACKEND_PORT (8000) - we avoid killing FRONTEND_PORT (5173) because
# 5173 is used by many tools (Vite defaults, Cursor, etc.); killing it can
# terminate our terminal session. Vite will auto-increment if 5173 is busy.
kill_port() {
  local port=$1
  local pids
  pids=$(lsof -ti :"$port" 2>/dev/null) || true
  if [[ -n "$pids" ]]; then
    # Safety: never kill our own script or its parent
    for pid in $pids; do
      [[ "$pid" == "$$" || "$pid" == "$PPID" ]] && continue
      echo "[calibrator-web] Killing existing process on port $port (PID: $pid)"
      kill "$pid" 2>/dev/null || true
    done
    sleep 2
    pids=$(lsof -ti :"$port" 2>/dev/null) || true
    for pid in $pids; do
      [[ "$pid" == "$$" || "$pid" == "$PPID" ]] && continue
      kill -9 "$pid" 2>/dev/null || true
    done
  fi
}
kill_port "$BACKEND_PORT"

if [[ "$SKIP_DEPENDENCY_INSTALL" != "1" ]]; then
  ensure_python_deps
  ensure_frontend_deps
else
  echo "[calibrator-web] Skipping dependency install because SKIP_DEPENDENCY_INSTALL=1"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "No virtual environment python found at $VENV_PYTHON" >&2
  echo "Run without SKIP_DEPENDENCY_INSTALL or set VENV_DIR/PYTHON_BIN correctly." >&2
  exit 1
fi
export PATH="$VENV_DIR/bin:$PATH"

echo "[calibrator-web] Starting backend API on ${BACKEND_HOST}:${BACKEND_PORT}"
BACKEND_UVICORN_ARGS=(
  --host "$BACKEND_HOST"
  --port "$BACKEND_PORT"
)
if [[ "$ENABLE_BACKEND_RELOAD" == "1" ]]; then
  BACKEND_UVICORN_ARGS+=(
    --reload
    --reload-dir "$ROOT_DIR/backend/scripts"
  )
fi
"$VENV_PYTHON" -m uvicorn backend.scripts.calibrator_api.app:app "${BACKEND_UVICORN_ARGS[@]}" &
BACKEND_PID=$!

pushd "$FRONTEND_DIR" >/dev/null

echo "[calibrator-web] Starting frontend on ${FRONTEND_HOST}:${FRONTEND_PORT}"
VITE_API_BASE="http://${BACKEND_HOST}:${BACKEND_PORT}" npm run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT" &
FRONTEND_PID=$!
popd >/dev/null

echo "[calibrator-web] UI:  http://${FRONTEND_HOST}:${FRONTEND_PORT}"
echo "[calibrator-web] API: http://${BACKEND_HOST}:${BACKEND_PORT}/api/health"

wait "$BACKEND_PID" "$FRONTEND_PID"
