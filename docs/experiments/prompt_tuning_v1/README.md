# Prompt Tuning v1

## Scope
This experiment isolates Stage-5 prompt variants without changing the main inference pipeline.

Relevant directories:
- Reports: `/zhaoshu/subgraph/reports/skill_enhanced_test/prompt_tuning_v1`
- Temporary splits: `/zhaoshu/subgraph/tmp/prompt_tuning`
- Variant definitions: `/zhaoshu/subgraph/src/subgraph_kgqa/prompt_variants/variants.py`
- Reusable tools: `/zhaoshu/subgraph/scripts/prompt_tuning/tools`

## Current status
- The `pilot10` prompt-variant comparison is complete.
- The trustworthy comparison is summarized in:
  - `/zhaoshu/subgraph/reports/skill_enhanced_test/prompt_tuning_v1/pilot10_summary.md`

## Important note on invalid historical runs
Earlier runs that went through the old `9002` chain and produced all-zero reports have been archived to:

- `/zhaoshu/subgraph/reports/skill_enhanced_test/prompt_tuning_v1/_archive/invalid_glm9002_runs_20260401`

Those archived runs should **not** be used for comparison.

Additional legacy branches were later archived under:

- `/zhaoshu/subgraph/reports/skill_enhanced_test/prompt_tuning_v1/_archive/legacy_prompt_variants_20260402`
- `/zhaoshu/subgraph/reports/skill_enhanced_test/prompt_tuning_v1/_archive/legacy_skill_audit_20260402`
- `/zhaoshu/subgraph/reports/skill_enhanced_test/prompt_tuning_v1/_archive/obsolete_misc_20260402`
- `/zhaoshu/subgraph/reports/skill_enhanced_test/prompt_tuning_v1/_archive/raw_logs_20260402`

## Pilot10 execution note
The completed `pilot10` comparison was executed on the local model chain:
- `qwen35-9b-local`
- local vLLM on `http://127.0.0.1:8000/v1`

## Next recommendation
If expanding beyond `pilot10`, only continue with:
- `v0_baseline`
- `v5_filter_then_answer`

## Agent worklog helper

To make experiment-agent progress visible while a run is still in flight, use:

- `/zhaoshu/subgraph/scripts/prompt_tuning/tools/agent_worklog.py`

Example:

```bash
python /zhaoshu/subgraph/scripts/prompt_tuning/tools/agent_worklog.py start \
  --log /zhaoshu/subgraph/reports/skill_enhanced_test/prompt_tuning_v1/example_worklog.jsonl \
  --title "mixed50 v5 skill audit"

python /zhaoshu/subgraph/scripts/prompt_tuning/tools/agent_worklog.py note \
  --log /zhaoshu/subgraph/reports/skill_enhanced_test/prompt_tuning_v1/example_worklog.jsonl \
  --message "Prepared split and environment."

python /zhaoshu/subgraph/scripts/prompt_tuning/tools/agent_worklog.py run \
  --log /zhaoshu/subgraph/reports/skill_enhanced_test/prompt_tuning_v1/example_worklog.jsonl \
  --label mixed50_v5 \
  -- python /zhaoshu/subgraph/scripts/run_skill_enhanced_test.py ...

python /zhaoshu/subgraph/scripts/prompt_tuning/tools/agent_worklog.py finish \
  --log /zhaoshu/subgraph/reports/skill_enhanced_test/prompt_tuning_v1/example_worklog.jsonl \
  --status success
```

This produces:

- a machine-readable JSONL log
- a markdown companion with timestamps
- per-step stdout/stderr log files

## Agent Worklog Helper

`scripts/prompt_tuning/tools/agent_worklog.py` is a tiny CLI that lets a lab agent
record its shell-level activity in a JSONL log (with a markdown companion).

Quick reference:

```bash
WLOG=tmp/worklog.jsonl

python scripts/prompt_tuning/tools/agent_worklog.py start   --log $WLOG --title "variant screen run"
python scripts/prompt_tuning/tools/agent_worklog.py note    --log $WLOG --message "using split old_good_50"
python scripts/prompt_tuning/tools/agent_worklog.py run     --log $WLOG --label "screen-v2a" -- python -c "print('hello')"
python scripts/prompt_tuning/tools/agent_worklog.py finish  --log $WLOG --status "ok"
```

This produces `$WLOG` (JSONL) and `$WLOG.md` (human-readable summary) side by
side. Each `run` entry captures timestamp, argv, cwd, duration, exit code, and
tail snippets of stdout/stderr.
