#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_BIN="${VLLM_BIN:-vllm}"

MODEL_PATH="${MODEL_PATH:-/zhaoshu/llm/Qwen3.5-9B}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen35-9b-local}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-2}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-30720}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
VLLM_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-1}"
VLLM_SKIP_MM_PROFILING="${VLLM_SKIP_MM_PROFILING:-1}"
VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-qwen3}"
VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-false}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# Local serving does not need outbound proxy settings.
unset http_proxy https_proxy all_proxy no_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY

export CUDA_VISIBLE_DEVICES
export VLLM_USE_MODELSCOPE
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --served-model-name "${MODEL_ALIAS}" \
  --tensor-parallel-size "${VLLM_TP_SIZE}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --reasoning-parser "${VLLM_REASONING_PARSER}" \
  --dtype auto \
  --trust-remote-code \
  $( [[ "${VLLM_ENFORCE_EAGER}" == "1" ]] && printf '%s' '--enforce-eager' ) \
  $( [[ "${VLLM_LANGUAGE_MODEL_ONLY}" == "1" ]] && printf '%s' '--language-model-only' ) \
  $( [[ "${VLLM_SKIP_MM_PROFILING}" == "1" ]] && printf '%s' '--skip-mm-profiling' ) \
  "$@"
