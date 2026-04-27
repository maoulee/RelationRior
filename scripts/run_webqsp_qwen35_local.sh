#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_PATH="${MODEL_PATH:-/zhaoshu/llm/Qwen3.5-9B}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen35-9b-local}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
GRAPH_SERVER_PORT="${GRAPH_SERVER_PORT:-8001}"
GRAPH_SERVER_HOST="${GRAPH_SERVER_HOST:-127.0.0.1}"
GRAPH_SERVER_DATASET="${GRAPH_SERVER_DATASET:-webqsp}"
GRAPH_SERVER_SPLIT="${GRAPH_SERVER_SPLIT:-train}"
MODEL_SERVER_LOG="${MODEL_SERVER_LOG:-${PROJECT_ROOT}/output/qwen35_9b_server.log}"
GRAPH_SERVER_LOG="${GRAPH_SERVER_LOG:-${PROJECT_ROOT}/output/graph_server_eval.log}"

export KGQA_LLM_API_URL="${KGQA_LLM_API_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
export KGQA_LLM_API_KEY="${KGQA_LLM_API_KEY:-EMPTY}"
export KGQA_MODEL_NAME="${KGQA_MODEL_NAME:-${MODEL_ALIAS}}"
export KGQA_KG_API_URL="${KGQA_KG_API_URL:-http://${GRAPH_SERVER_HOST}:${GRAPH_SERVER_PORT}}"
export KGQA_TEST_DATA_PATH="${KGQA_TEST_DATA_PATH:-${PROJECT_ROOT}/data/webqsp/webqsp_train.jsonl}"
export KGQA_LLM_MAX_TOKENS="${KGQA_LLM_MAX_TOKENS:-1024}"
export KGQA_LLM_TEMPERATURE="${KGQA_LLM_TEMPERATURE:-0.7}"
export KGQA_LLM_TOP_P="${KGQA_LLM_TOP_P:-0.8}"
export KGQA_LLM_TOP_K="${KGQA_LLM_TOP_K:-20}"
export KGQA_LLM_MIN_P="${KGQA_LLM_MIN_P:-0.0}"
export KGQA_LLM_PRESENCE_PENALTY="${KGQA_LLM_PRESENCE_PENALTY:-1.5}"
export KGQA_LLM_REPETITION_PENALTY="${KGQA_LLM_REPETITION_PENALTY:-1.0}"
export KGQA_ENABLE_THINKING="${KGQA_ENABLE_THINKING:-0}"
export KGQA_LIMIT_CASES="${KGQA_LIMIT_CASES:-1}"
export KGQA_MAX_PARALLEL_CASES="${KGQA_MAX_PARALLEL_CASES:-1}"
export KGQA_BEST_OF_N="${KGQA_BEST_OF_N:-1}"
export KGQA_MAX_TURNS="${KGQA_MAX_TURNS:-6}"
export KGQA_LLM_TIMEOUT_SEC="${KGQA_LLM_TIMEOUT_SEC:-180}"
export KGQA_REPORT_FILE="${KGQA_REPORT_FILE:-${PROJECT_ROOT}/reports/test_qwen35_9b_local_webqsp.md}"

unset http_proxy https_proxy all_proxy no_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
mkdir -p "${PROJECT_ROOT}/output" "${PROJECT_ROOT}/reports"

MODEL_PID=""
GRAPH_PID=""

cleanup() {
  if [[ -n "${MODEL_PID}" ]] && ps -p "${MODEL_PID}" >/dev/null 2>&1; then
    kill "${MODEL_PID}" || true
    wait "${MODEL_PID}" 2>/dev/null || true
  fi
  if [[ -n "${GRAPH_PID}" ]] && ps -p "${GRAPH_PID}" >/dev/null 2>&1; then
    kill "${GRAPH_PID}" || true
    wait "${GRAPH_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

if ! curl -sf -o /dev/null "http://${VLLM_HOST}:${VLLM_PORT}/v1/models"; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    "${PROJECT_ROOT}/scripts/start_local_qwen35_server.sh" \
    > "${MODEL_SERVER_LOG}" 2>&1 &
  MODEL_PID=$!
fi

if ! curl -sf -o /dev/null "http://${GRAPH_SERVER_HOST}:${GRAPH_SERVER_PORT}/health"; then
  "${PROJECT_ROOT}/scripts/start_graph_server.sh" \
    --dataset "${GRAPH_SERVER_DATASET}" \
    --split "${GRAPH_SERVER_SPLIT}" \
    --port "${GRAPH_SERVER_PORT}" \
    > "${GRAPH_SERVER_LOG}" 2>&1 &
  GRAPH_PID=$!
fi

SECONDS_WAITED=0
MODEL_TIMEOUT="${MODEL_SERVER_TIMEOUT:-240}"
until curl -sf -o /dev/null "http://${VLLM_HOST}:${VLLM_PORT}/v1/models"; do
  if [[ "${SECONDS_WAITED}" -ge "${MODEL_TIMEOUT}" ]]; then
    echo "Local Qwen3.5-9B server failed to start within ${MODEL_TIMEOUT}s" >&2
    tail -n 40 "${MODEL_SERVER_LOG}" >&2 || true
    exit 1
  fi
  sleep 5
  SECONDS_WAITED=$((SECONDS_WAITED + 5))
done

SECONDS_WAITED=0
GRAPH_TIMEOUT="${GRAPH_SERVER_TIMEOUT:-90}"
until curl -sf -o /dev/null "http://${GRAPH_SERVER_HOST}:${GRAPH_SERVER_PORT}/health"; do
  if [[ "${SECONDS_WAITED}" -ge "${GRAPH_TIMEOUT}" ]]; then
    echo "Graph server failed to start within ${GRAPH_TIMEOUT}s" >&2
    tail -n 40 "${GRAPH_SERVER_LOG}" >&2 || true
    exit 1
  fi
  sleep 2
  SECONDS_WAITED=$((SECONDS_WAITED + 2))
done

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/test_pipe6.py"
