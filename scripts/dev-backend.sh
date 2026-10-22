#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$ROOT_DIR/backend"

cd "$BACKEND_DIR"

if [ -f ".venv/bin/activate" ]; then
    echo "Activating Python virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    echo "Activating Python virtual environment (venv)..."
    source venv/bin/activate
fi

echo "Starting Argus FastAPI backend on http://localhost:8000..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
