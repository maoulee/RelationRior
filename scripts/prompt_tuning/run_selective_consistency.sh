#!/usr/bin/env bash
set -euo pipefail

# Selective consistency experiment script
# Compares: no consistency (Arm A) vs global consistency (Arm B) vs selective consistency (Arm C)
# Focus: unstable subsets where global consistency showed mixed results

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUTPUT_ROOT="${1:-$ROOT/reports/skill_enhanced_test/prompt_tuning_v1/selective_consistency}"
LOG_PATH="$OUTPUT_ROOT/worklog.jsonl"

TRAIN="$ROOT/data/webqsp/webqsp_train.jsonl"
SKILLS="$ROOT/skills/webqsp_train_case_skills_actionspace_v1"
DIAG12="$ROOT/tmp/prompt_tuning/skill_audit_diag12.jsonl"
OLD_BAD="$ROOT/tmp/prompt_tuning/old_bad_100.jsonl"
UNSTABLE="$ROOT/tmp/prompt_tuning/unstable_all.jsonl"

# Base config (same as existing experiments)
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
  --message "Selective consistency experiment: Arm C (selective) on unstable subsets"

COMMON_ARGS=(
  --train-data-path "$TRAIN"
  --skills-root "$SKILLS"
  --variant protocol_guard_action_id_experiment
  --max-turns 8
  --skill-top-k 3
  --max-concurrency 32
  --output-dir "$OUTPUT_ROOT"
  --stage5-prompt-variant v5_filter_then_answer
)

echo "============================================"
echo "Selective Consistency Experiment"
echo "============================================"
echo "Output: $OUTPUT_ROOT"
echo ""

# --- Arm C1: Selective on diag12 (12 cases, fast, directly comparable to existing Arm A/B) ---
echo "[1/3] Arm C1: selective consistency on diag12..."
export KGQA_ENABLE_DECISION_CONSISTENCY=1
export KGQA_CONSISTENCY_MODE=selective
export KGQA_DECISION_CONSISTENCY_MIN_TURN="${KGQA_DECISION_CONSISTENCY_MIN_TURN:-2}"

for i in 1 2 3; do
  run_logged "diag12_armC_selective_r${i}" \
    python -u "$ROOT/scripts/run_skill_enhanced_test.py" \
    --data-path "$DIAG12" \
    --label "selective_consistency_diag12_armC_r${i}" \
    "${COMMON_ARGS[@]}"
done

# --- Arm C2: Selective on old_bad_100 (100 historically difficult cases) ---
echo "[2/3] Arm C2: selective consistency on old_bad_100..."
export KGQA_ENABLE_DECISION_CONSISTENCY=1
export KGQA_CONSISTENCY_MODE=selective

run_logged "old_bad_100_armC_selective" \
  python -u "$ROOT/scripts/run_skill_enhanced_test.py" \
  --data-path "$OLD_BAD" \
  --label "selective_consistency_old_bad100_armC" \
  "${COMMON_ARGS[@]}"

# --- Arm C3: Selective on full unstable set (comprehensive) ---
echo "[3/3] Arm C3: selective consistency on unstable_all..."
if [ -f "$UNSTABLE" ]; then
  export KGQA_ENABLE_DECISION_CONSISTENCY=1
  export KGQA_CONSISTENCY_MODE=selective

  run_logged "unstable_all_armC_selective" \
    python -u "$ROOT/scripts/run_skill_enhanced_test.py" \
    --data-path "$UNSTABLE" \
    --label "selective_consistency_unstable_all_armC" \
    "${COMMON_ARGS[@]}"
else
  echo "Skipping unstable_all (file not found: $UNSTABLE)"
fi

python "$ROOT/scripts/prompt_tuning/tools/agent_worklog.py" finish \
  --log "$LOG_PATH" \
  --status "selective consistency experiment completed"

echo ""
echo "============================================"
echo "Experiment complete. Results in:"
echo "  $OUTPUT_ROOT"
echo ""
echo "Compare against existing results:"
echo "  Arm A (no consistency):  reports/skill_enhanced_test/prompt_tuning_v1/decision_consistency_diag12_armA_r*/report.md"
echo "  Arm B (global):          reports/skill_enhanced_test/prompt_tuning_v1/decision_consistency_diag12_armB_r*/report.md"
echo "  Arm C (selective):       $OUTPUT_ROOT/selective_consistency_diag12_armC_r*/report.md"
echo "============================================"
