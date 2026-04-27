#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GRAPH_SERVER_DATASET="${GRAPH_SERVER_DATASET:-webqsp}"
GRAPH_SERVER_SPLIT="${GRAPH_SERVER_SPLIT:-train}"
GRAPH_SERVER_PORT="${GRAPH_SERVER_PORT:-8001}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/graph_server.py" \
  --dataset "${GRAPH_SERVER_DATASET}" \
  --split "${GRAPH_SERVER_SPLIT}" \
  --port "${GRAPH_SERVER_PORT}" \
  "$@"
