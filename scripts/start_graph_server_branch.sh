#!/usr/bin/env bash
set -euo pipefail

BRANCH_NAME="${1:-original}"
shift || true

ROOT="/zhaoshu/subgraph"
BACKEND_SOURCE="$ROOT/backend_branches/$BRANCH_NAME/retrieve/graph_server.py"

if [[ ! -f "$BACKEND_SOURCE" ]]; then
  echo "Unknown backend branch: $BRANCH_NAME" >&2
  echo "Expected file: $BACKEND_SOURCE" >&2
  exit 1
fi

exec python "$ROOT/graph_server.py" --backend-source "$BACKEND_SOURCE" "$@"
