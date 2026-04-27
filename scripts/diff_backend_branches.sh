#!/usr/bin/env bash
set -euo pipefail

ROOT="/zhaoshu/subgraph/backend_branches"
diff -u \
  "$ROOT/original/retrieve/graph_server.py" \
  "$ROOT/experiment/retrieve/graph_server.py" || true
