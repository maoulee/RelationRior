#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
LLM_API_URL="${KGQA_LLM_API_URL:-http://127.0.0.1:8000/v1}"
LLM_API_KEY="${KGQA_LLM_API_KEY:-EMPTY}"
MODEL_NAME="${KGQA_MODEL_NAME:-qwen35-9b-local}"
KG_API_URL="${KGQA_KG_API_URL:-http://127.0.0.1:8001}"

VARIANTS=(${KGQA_PROMPT_VARIANTS:-original short_reasoning minimal_reasoning})
LIMIT_CASES="${KGQA_LIMIT_CASES:-50}"
MAX_PARALLEL_CASES="${KGQA_MAX_PARALLEL_CASES:-32}"
LLM_MAX_TOKENS="${KGQA_LLM_MAX_TOKENS:-2048}"
BEST_OF_N="${KGQA_BEST_OF_N:-1}"
MAX_TURNS="${KGQA_MAX_TURNS:-6}"
ENABLE_THINKING="${KGQA_ENABLE_THINKING:-0}"
RESULT_DIR="${KGQA_COMPARE_RESULT_DIR:-${PROJECT_ROOT}/reports/prompt_variants}"
SUMMARY_FILE="${KGQA_COMPARE_SUMMARY_FILE:-${RESULT_DIR}/summary_50cases.md}"

mkdir -p "${RESULT_DIR}"

if ! curl -sf -o /dev/null "${LLM_API_URL}/models"; then
  echo "LLM server is not ready at ${LLM_API_URL}" >&2
  exit 1
fi

if ! curl -sf -o /dev/null "${KG_API_URL}/health"; then
  echo "Graph server is not ready at ${KG_API_URL}" >&2
  exit 1
fi

echo "# Prompt Variant Comparison" > "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"
echo "| Variant | Avg Best F1 | Hit@1 | Total Turns | Frontend Validation Errors | Report |" >> "${SUMMARY_FILE}"
echo "|---|---:|---:|---:|---:|---|" >> "${SUMMARY_FILE}"

for variant in "${VARIANTS[@]}"; do
  report_file="${RESULT_DIR}/test_${variant}_p${MAX_PARALLEL_CASES}_t${LLM_MAX_TOKENS}.md"

  echo "Running variant=${variant} cases=${LIMIT_CASES} parallel=${MAX_PARALLEL_CASES} max_tokens=${LLM_MAX_TOKENS}"

  KGQA_PROMPT_VARIANT="${variant}" \
  KGQA_LLM_API_URL="${LLM_API_URL}" \
  KGQA_LLM_API_KEY="${LLM_API_KEY}" \
  KGQA_MODEL_NAME="${MODEL_NAME}" \
  KGQA_KG_API_URL="${KG_API_URL}" \
  KGQA_LIMIT_CASES="${LIMIT_CASES}" \
  KGQA_MAX_PARALLEL_CASES="${MAX_PARALLEL_CASES}" \
  KGQA_LLM_MAX_TOKENS="${LLM_MAX_TOKENS}" \
  KGQA_BEST_OF_N="${BEST_OF_N}" \
  KGQA_MAX_TURNS="${MAX_TURNS}" \
  KGQA_ENABLE_THINKING="${ENABLE_THINKING}" \
  KGQA_REPORT_FILE="${report_file}" \
  "${PYTHON_BIN}" "${PROJECT_ROOT}/test_pipe6.py"

metrics="$("${PYTHON_BIN}" - <<PY
from pathlib import Path
import re
p = Path(r"${report_file}")
text = p.read_text()
def get(key):
    m = re.search(rf"\\*\\*{re.escape(key)}\\*\\*: ([^\\n]+)", text)
    return m.group(1) if m else "NA"
print(
    f"| ${variant} | {get('Avg Best F1')} | {get('Hit@1 (first pred in GT)') if get('Hit@1 (first pred in GT)') != 'NA' else get('Hit@1 (Best F1>=0.5)')} | "
    f"{get('Total Turns')} | {get('Frontend Validation Errors')} | "
    f"[report](${report_file}) |"
)
PY
)"

  echo "${metrics}" >> "${SUMMARY_FILE}"
done

echo "" >> "${SUMMARY_FILE}"
echo "Saved comparison summary to ${SUMMARY_FILE}"
