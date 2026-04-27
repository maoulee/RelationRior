#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export KGQA_LLM_API_URL="${KGQA_LLM_API_URL:-https://open.bigmodel.cn/api/paas/v4}"
export KGQA_MODEL_NAME="${KGQA_MODEL_NAME:-glm-4.7-flash}"
export KGQA_TEST_DATA_PATH="${KGQA_TEST_DATA_PATH:-${PROJECT_ROOT}/data/webqsp/webqsp_train.jsonl}"
export KGQA_LIMIT_CASES="${KGQA_LIMIT_CASES:-1}"
export KGQA_MAX_PARALLEL_CASES="${KGQA_MAX_PARALLEL_CASES:-1}"
export KGQA_BEST_OF_N="${KGQA_BEST_OF_N:-1}"
export KGQA_REPORT_FILE="${KGQA_REPORT_FILE:-${PROJECT_ROOT}/reports/test_glm47_flash_webqsp.md}"
export KGQA_KG_API_URL="${KGQA_KG_API_URL:-http://localhost:8001}"
export GRAPH_SERVER_DATASET="${GRAPH_SERVER_DATASET:-webqsp}"
export GRAPH_SERVER_SPLIT="${GRAPH_SERVER_SPLIT:-train}"
export GRAPH_SERVER_PORT="${GRAPH_SERVER_PORT:-8001}"
export KGQA_LLM_TIMEOUT_SEC="${KGQA_LLM_TIMEOUT_SEC:-120}"
export KGQA_USE_GRAFTCP="${KGQA_USE_GRAFTCP:-1}"
export KGQA_GRAFTCP_LOCAL_PORT="${KGQA_GRAFTCP_LOCAL_PORT:-2233}"
export KGQA_GRAFTCP_LOCAL_FIFO="${KGQA_GRAFTCP_LOCAL_FIFO:-/tmp/graftcplocal-2233.fifo}"
export KGQA_PROXY_HTTP_ADDR="${KGQA_PROXY_HTTP_ADDR:-127.0.0.1:8888}"
export KGQA_PROXY_SOCKS5_ADDR="${KGQA_PROXY_SOCKS5_ADDR:-127.0.0.1:8888}"
export KGQA_GRAFTCP_BIN="${KGQA_GRAFTCP_BIN:-/root/.graftcp-antigravity/graftcp/graftcp}"
export KGQA_GRAFTCP_LOCAL_BIN="${KGQA_GRAFTCP_LOCAL_BIN:-/root/.graftcp-antigravity/graftcp/local/graftcp-local}"
export KGQA_GRAFTCP_LOCAL_LOG="${KGQA_GRAFTCP_LOCAL_LOG:-/tmp/graftcp-local.log}"

if [[ -z "${KGQA_LLM_API_KEY:-}" ]]; then
  echo "KGQA_LLM_API_KEY is not set." >&2
  echo "Example:" >&2
  echo "  export KGQA_LLM_API_KEY='your-secret-key'" >&2
  exit 1
fi

if [[ "${KGQA_START_GRAPH_SERVER:-0}" == "1" ]]; then
  "${PROJECT_ROOT}/scripts/start_graph_server.sh" \
    --dataset "${GRAPH_SERVER_DATASET}" \
    --split "${GRAPH_SERVER_SPLIT}" \
    --port "${GRAPH_SERVER_PORT}" &
  GRAPH_PID=$!
  trap 'kill "${GRAPH_PID}" >/dev/null 2>&1 || true' EXIT
  sleep "${KGQA_GRAPH_SERVER_WAIT_SEC:-8}"
fi

if [[ "${KGQA_USE_GRAFTCP}" == "1" ]]; then
  if [[ ! -x "${KGQA_GRAFTCP_BIN}" || ! -x "${KGQA_GRAFTCP_LOCAL_BIN}" ]]; then
    echo "graftcp binaries not found, falling back to direct python run." >&2
    exec "${PYTHON_BIN}" "${PROJECT_ROOT}/test_pipe6.py"
  fi

  "${KGQA_GRAFTCP_LOCAL_BIN}" \
    -listen ":${KGQA_GRAFTCP_LOCAL_PORT}" \
    -http_proxy "${KGQA_PROXY_HTTP_ADDR}" \
    -socks5 "${KGQA_PROXY_SOCKS5_ADDR}" \
    -select_proxy_mode auto \
    -pipepath "${KGQA_GRAFTCP_LOCAL_FIFO}" \
    -logfile "${KGQA_GRAFTCP_LOCAL_LOG}" \
    -logfile-truncate &
  GRAFTCP_LOCAL_PID=$!
  trap 'kill "${GRAFTCP_LOCAL_PID}" >/dev/null 2>&1 || true; [[ -n "${GRAPH_PID:-}" ]] && kill "${GRAPH_PID}" >/dev/null 2>&1 || true' EXIT
  sleep "${KGQA_GRAFTCP_WAIT_SEC:-2}"

  unset http_proxy https_proxy all_proxy no_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
  exec "${KGQA_GRAFTCP_BIN}" \
    -p "${KGQA_GRAFTCP_LOCAL_PORT}" \
    -f "${KGQA_GRAFTCP_LOCAL_FIFO}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/test_pipe6.py"
fi

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/test_pipe6.py"
