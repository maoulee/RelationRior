#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash scripts/train_grpo.sh [extra swift args]

Environment overrides:
  MODEL_PATH
  DATASET_PATH
  OUTPUT_DIR
  CUDA_VISIBLE_DEVICES
  GRAPH_SERVER_PORT
  GRAPH_SERVER_DATASET
  GRAPH_SERVER_SPLIT
  PYTHON_BIN
  SWIFT_BIN

This launcher starts the graph server, waits for /health, then runs:
  swift rlhf --external_plugins plug.py ...
EOF
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-1}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH="${MODEL_PATH:-/zhaoshu/llm/qwen3_4b/}"
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/data/webqsp/webqsp_train_balanced.jsonl}"
PLUGIN_PATH="${PLUGIN_PATH:-${PROJECT_ROOT}/plug.py}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoint/webqsp/grpo_adapter_qwen3}"

GRAPH_SERVER_DATASET="${GRAPH_SERVER_DATASET:-webqsp}"
GRAPH_SERVER_SPLIT="${GRAPH_SERVER_SPLIT:-train}"
GRAPH_SERVER_HOST="${GRAPH_SERVER_HOST:-127.0.0.1}"
GRAPH_SERVER_PORT="${GRAPH_SERVER_PORT:-8001}"
GRAPH_SERVER_LOG="${GRAPH_SERVER_LOG:-${PROJECT_ROOT}/output/graph_server.log}"
GRAPH_SERVER_PID_FILE="${GRAPH_SERVER_PID_FILE:-/tmp/subgraph_graph_server_$$.pid}"

NUM_EPOCHS="${NUM_EPOCHS:-2}"
BATCH_SIZE_PER_DEVICE="${BATCH_SIZE_PER_DEVICE:-1}"
GRADIENT_ACCUM_STEPS="${GRADIENT_ACCUM_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
BETA="${BETA:-0.0}"
TEMPERATURE="${TEMPERATURE:-0.7}"
NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-2}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.3}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-25000}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1500}"
MAX_TURNS="${MAX_TURNS:-7}"
MOVE_MODEL_BATCHES="${MOVE_MODEL_BATCHES:-32}"
SLEEP_LEVEL="${SLEEP_LEVEL:-1}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
SWIFT_BIN="${SWIFT_BIN:-/root/miniconda3/bin/swift}"

export REWARD_COS_IDEAL_LEN="${REWARD_COS_IDEAL_LEN:-400}"
export REWARD_COS_MAX_LEN="${REWARD_COS_MAX_LEN:-700}"
export REWARD_COS_MIN_THINK_LEN="${REWARD_COS_MIN_THINK_LEN:-50}"
export REWARD_COS_BONUS="${REWARD_COS_BONUS:-0.3}"
export REWARD_COS_PENALTY="${REWARD_COS_PENALTY:--0.5}"
export REWARD_FORMAT_SOFT_LIMIT="${REWARD_FORMAT_SOFT_LIMIT:-700}"
export REWARD_FORMAT_HARD_LIMIT="${REWARD_FORMAT_HARD_LIMIT:-1000}"

mkdir -p "${OUTPUT_DIR}" "$(dirname "${GRAPH_SERVER_LOG}")"

cleanup() {
  if [[ -f "${GRAPH_SERVER_PID_FILE}" ]]; then
    GRAPH_SERVER_PID="$(cat "${GRAPH_SERVER_PID_FILE}")"
    if [[ -n "${GRAPH_SERVER_PID}" ]] && ps -p "${GRAPH_SERVER_PID}" >/dev/null 2>&1; then
      kill "${GRAPH_SERVER_PID}" || true
      wait "${GRAPH_SERVER_PID}" 2>/dev/null || true
    fi
    rm -f "${GRAPH_SERVER_PID_FILE}"
  fi
}

trap cleanup EXIT INT TERM

echo "--- Starting graph server ---"
"${PYTHON_BIN}" "${PROJECT_ROOT}/graph_server.py" \
  --dataset "${GRAPH_SERVER_DATASET}" \
  --split "${GRAPH_SERVER_SPLIT}" \
  --port "${GRAPH_SERVER_PORT}" \
  > "${GRAPH_SERVER_LOG}" 2>&1 &
echo $! > "${GRAPH_SERVER_PID_FILE}"

HEALTH_CHECK_URL="http://${GRAPH_SERVER_HOST}:${GRAPH_SERVER_PORT}/health"
TIMEOUT="${GRAPH_SERVER_TIMEOUT:-60}"
SECONDS_WAITED=0
until curl -sf -o /dev/null "${HEALTH_CHECK_URL}"; do
  if [[ "${SECONDS_WAITED}" -ge "${TIMEOUT}" ]]; then
    echo "Graph server failed to start within ${TIMEOUT}s"
    tail -n 30 "${GRAPH_SERVER_LOG}" || true
    exit 1
  fi
  sleep 2
  SECONDS_WAITED=$((SECONDS_WAITED + 2))
done

echo "--- Graph server ready, starting GRPO ---"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
WANDB_MODE="${WANDB_MODE:-offline}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
"${SWIFT_BIN}" rlhf \
  --rlhf_type grpo \
  --model_type qwen3 \
  --loss_type sapo \
  --tau_pos 1.0 \
  --tau_neg 1.05 \
  --model "${MODEL_PATH}" \
  --dataset "${DATASET_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs "${NUM_EPOCHS}" \
  --per_device_train_batch_size "${BATCH_SIZE_PER_DEVICE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUM_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.01 \
  --train_type lora \
  --lora_rank 64 \
  --lora_alpha 128 \
  --lorap_lr_ratio 24 \
  --lora_dropout 0.05 \
  --external_plugins "${PLUGIN_PATH}" \
  --multi_turn_scheduler kg_scheduler \
  --max_turns "${MAX_TURNS}" \
  --overlong_filter true \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_tensor_parallel_size "${VLLM_TP_SIZE}" \
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --completion_length_limit_scope per_round \
  --temperature "${TEMPERATURE}" \
  --reward_funcs f1 plan action reason compliance behavior cosine_len \
  --reward_weights 1.0 0.3 0.5 0.4 0.3 0.5 0.2 \
  --num_generations "${NUM_GENERATIONS}" \
  --deepspeed zero3 \
  --optim adamw_bnb_8bit \
  --torch_dtype bfloat16 \
  --beta "${BETA}" \
  --epsilon 0.2 \
  --epsilon_high 0.28 \
  --logging_steps 1 \
  --save_strategy epoch \
  --save_total_limit 2 \
  --report_to tensorboard \
  --offload_optimizer true \
  --offload_model true \
  --gradient_checkpointing true \
  --log_completions true \
  --remove_unused_columns false \
  --move_model_batches "${MOVE_MODEL_BATCHES}" \
  --sleep_level "${SLEEP_LEVEL}" \
  --vllm_server_pass_dataset true \
  "$@"
