#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

LOG_PATH="${1:-$ROOT/reports/skill_enhanced_test/prompt_tuning_v1/decision_consistency_worklog.jsonl}"
OUTPUT_ROOT="${2:-$ROOT/reports/skill_enhanced_test/prompt_tuning_v1}"

TRAIN="$ROOT/data/webqsp/webqsp_train.jsonl"
SKILLS="$ROOT/skills/webqsp_train_case_skills_actionspace_v1"
OLD_BAD="$ROOT/tmp/prompt_tuning/old_bad_100.jsonl"
DIAG12="$ROOT/tmp/prompt_tuning/skill_audit_diag12.jsonl"

export KGQA_LLM_API_URL="${KGQA_LLM_API_URL:-http://127.0.0.1:8000/v1}"
export KGQA_MODEL_NAME="${KGQA_MODEL_NAME:-qwen35-9b-local}"
export KGQA_KG_API_URL="${KGQA_KG_API_URL:-http://127.0.0.1:8001}"
export KGQA_ENABLE_THINKING="${KGQA_ENABLE_THINKING:-0}"
export KGQA_LLM_TEMPERATURE="${KGQA_LLM_TEMPERATURE:-0.1}"
export KGQA_LLM_TOP_P="${KGQA_LLM_TOP_P:-1.0}"
export KGQA_LLM_TOP_K="${KGQA_LLM_TOP_K:-20}"
export KGQA_SKILL_REASONING_INJECTION_MODE="${KGQA_SKILL_REASONING_INJECTION_MODE:-per_skill}"
export KGQA_SKILL_AUDIT_MODE="${KGQA_SKILL_AUDIT_MODE:-conflict_only}"
export KGQA_EXTEND_TURNS_ON_FRONTEND_ERROR="${KGQA_EXTEND_TURNS_ON_FRONTEND_ERROR:-1}"
export KGQA_MAX_TURNS_WITH_FRONTEND_REPAIR="${KGQA_MAX_TURNS_WITH_FRONTEND_REPAIR:-16}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUTPUT_ROOT"

run_logged() {
  local label="$1"
  shift
  python "$ROOT/scripts/prompt_tuning/tools/agent_worklog.py" run --log "$LOG_PATH" --label "$label" -- "$@"
}

python "$ROOT/scripts/prompt_tuning/tools/agent_worklog.py" note \
  --log "$LOG_PATH" \
  --message "Running remaining decision-consistency arms via run_decision_consistency_suite.sh"

export KGQA_ENABLE_DECISION_CONSISTENCY=1
export KGQA_DECISION_CONSISTENCY_MIN_TURN="${KGQA_DECISION_CONSISTENCY_MIN_TURN:-2}"
run_logged old_bad_100_arm_B_consistency \
  python -u "$ROOT/scripts/run_skill_enhanced_test.py" \
  --data-path "$OLD_BAD" \
  --train-data-path "$TRAIN" \
  --skills-root "$SKILLS" \
  --variant protocol_guard_action_id_experiment \
  --max-turns 8 \
  --skill-top-k 3 \
  --max-concurrency 32 \
  --output-dir "$OUTPUT_ROOT" \
  --label decision_consistency_old_bad100_armB \
  --stage5-prompt-variant v5_filter_then_answer

export KGQA_ENABLE_DECISION_CONSISTENCY=0
for i in 1 2 3; do
  run_logged "diag12_arm_A_no_consistency_r${i}" \
    python -u "$ROOT/scripts/run_skill_enhanced_test.py" \
    --data-path "$DIAG12" \
    --train-data-path "$TRAIN" \
    --skills-root "$SKILLS" \
    --variant protocol_guard_action_id_experiment \
    --max-turns 8 \
    --skill-top-k 3 \
    --max-concurrency 32 \
    --output-dir "$OUTPUT_ROOT" \
    --label "decision_consistency_diag12_armA_r${i}" \
    --stage5-prompt-variant v5_filter_then_answer
done

export KGQA_ENABLE_DECISION_CONSISTENCY=1
export KGQA_DECISION_CONSISTENCY_MIN_TURN="${KGQA_DECISION_CONSISTENCY_MIN_TURN:-2}"
for i in 1 2 3; do
  run_logged "diag12_arm_B_consistency_r${i}" \
    python -u "$ROOT/scripts/run_skill_enhanced_test.py" \
    --data-path "$DIAG12" \
    --train-data-path "$TRAIN" \
    --skills-root "$SKILLS" \
    --variant protocol_guard_action_id_experiment \
    --max-turns 8 \
    --skill-top-k 3 \
    --max-concurrency 32 \
    --output-dir "$OUTPUT_ROOT" \
    --label "decision_consistency_diag12_armB_r${i}" \
    --stage5-prompt-variant v5_filter_then_answer
done

python "$ROOT/scripts/prompt_tuning/tools/agent_worklog.py" finish \
  --log "$LOG_PATH" \
  --status "decision consistency suite completed"
