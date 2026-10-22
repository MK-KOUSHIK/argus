#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
FRONTEND_DIR="$ROOT_DIR/frontend"

cd "$FRONTEND_DIR"

echo "Starting Argus React + Vite frontend on http://localhost:5173..."
npm run dev
