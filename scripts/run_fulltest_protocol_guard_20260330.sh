#!/usr/bin/env bash
set -euo pipefail

ROOT="/zhaoshu/subgraph"
LOG_PATH="$ROOT/output/fulltest_protocol_guard_20260330.log"

exec > >(tee -a "$LOG_PATH") 2>&1

echo "[START] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[INFO] Running full WebQSP test evaluation: baseline -> skill_top3"

export KGQA_LLM_API_URL="http://127.0.0.1:8000/v1"
export KGQA_KG_API_URL="http://127.0.0.1:8001"
export KGQA_MODEL_NAME="qwen35-9b-local"
export KGQA_ENABLE_THINKING="0"
export KGQA_LLM_MAX_TOKENS="2048"
export KGQA_LLM_TEMPERATURE="0.7"
export KGQA_LLM_TOP_P="0.8"
export KGQA_LLM_TOP_K="20"
export KGQA_ENABLE_WEB_SEARCH="0"
export KGQA_ENFORCE_WEB_SEARCH_GATE="0"
export KGQA_WEB_SEARCH_WIKIDATA_FIRST="0"
unset KGQA_GRAPH_SNAPSHOT_DATE || true

COMMON_ARGS=(
  --data-path "$ROOT/data/webqsp/webqsp_test.jsonl"
  --train-data-path "$ROOT/data/webqsp/webqsp_train.jsonl"
  --skills-root "$ROOT/skills/webqsp_train_case_skills_en"
  --variant protocol_guard_action_id_experiment
  --max-turns 8
  --max-concurrency 16
)

echo "[RUN] baseline_no_skills"
PYTHONPATH="$ROOT/src:$ROOT" python "$ROOT/scripts/run_skill_enhanced_test.py" \
  "${COMMON_ARGS[@]}" \
  --no-skills \
  --label fulltest_baseline_protocol_guard_20260330

echo "[DONE] baseline_no_skills $(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "[RUN] skill_top3"
PYTHONPATH="$ROOT/src:$ROOT" python "$ROOT/scripts/run_skill_enhanced_test.py" \
  "${COMMON_ARGS[@]}" \
  --skill-top-k 3 \
  --label fulltest_skill_top3_protocol_guard_20260330

echo "[DONE] skill_top3 $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[END] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
