# Code Team Handoff (2026-04-07)

## Current Goal

We are in the middle of two parallel tracks:

1. **Unify `webqsp_train_case_skills_v3`**
   - full rebuild with the latest single-core / trajectory-first / shortest-path-fallback logic
2. **Evaluate the new action/filter/runtime prompt flow**
   - especially on hard cases where `plan` is usually fine but `action` selection and post-action refinement are unstable

This document summarizes the current state, the latest fixes, and the concrete commands the code team can run next.


## What Has Already Changed

### 1. Skill generation / core relation selection

Relevant files:

- [case_skill.py](/zhaoshu/subgraph/src/subgraph_kgqa/skill_mining/case_skill.py)
- [extractor.py](/zhaoshu/subgraph/src/subgraph_kgqa/skill_mining/extractor.py)
- [rebuild_case_skills_v3.py](/zhaoshu/subgraph/scripts/rebuild_case_skills_v3.py)

Current logic:

- If original trajectory contains usable `plan` relations, `core_relation` should come from those trajectory relations.
- If trajectory plan relations are missing, fallback is:
  - build shortest connected paths from anchor to GT answer
  - let the LLM choose the winning path
  - derive the single `core_relation` from that path

Important principle:

- shortest path is **fallback only**
- it must not override a good trajectory plan relation


### 2. Action / filter stage refactor

Relevant files:

- [runtime.py](/zhaoshu/subgraph/src/subgraph_kgqa/inference/runtime.py)
- [state.py](/zhaoshu/subgraph/src/subgraph_kgqa/inference/state.py)
- [parser.py](/zhaoshu/subgraph/src/subgraph_kgqa/inference/parser.py)
- [plug_v12_feedback.py](/zhaoshu/subgraph/plug_v12_feedback.py)
- [sys_prompt.py](/zhaoshu/subgraph/sys_prompt.py)
- [action_filter_stage_refactor_20260407.md](/zhaoshu/subgraph/docs/experiments/prompt_tuning_v1/action_filter_stage_refactor_20260407.md)

Current runtime shape:

1. `plan`
2. `action`
3. `action-result judgment`
4. `filter stage` (if suggested filter relations exist)
5. `final reasoning`

Key decisions:

- after action execution, do **not** jump directly back to `plan`
- first decide:
  - `retry_action`
  - `proceed`
- if backend gives `Suggested Filter Relations`, enter filter stage before final reasoning
- final reasoning still owns:
  - keep whole action space
  - subset selection
  - collapse


### 3. Stage hints

Relevant files:

- [plug_v12_feedback.py](/zhaoshu/subgraph/plug_v12_feedback.py)
- [runtime.py](/zhaoshu/subgraph/src/subgraph_kgqa/inference/runtime.py)
- [skill_aggregator.py](/zhaoshu/subgraph/src/subgraph_kgqa/skill_mining/skill_aggregator.py)

Current status:

- `Stage 1→2` and `Stage 2→3` were shortened but still keep format-critical constraints
- action-related stages now tell the model to **check historical action-skill guidance**
- duplicated stage-5 hint blocks were reduced via runtime deduplication
- LLM-aggregated skill labels are sanitized so weak labels like `Merged` do not leak through as-is

Still under active observation:

- some experiments still appear to consume mixed old/new skill artifacts
- some action-stage aggregated skill blocks may still feel too shallow if upstream cards are stale


## Current Blocker

The main blocker is no longer "missing files".

Coverage check:

- train total: `2601`
- `webqsp_train_case_skills_v3` json count: `2601`
- `webqsp_train_case_skills_v3` md count: `2601`
- `index.json` count: `2601`

So the issue is:

- **v3 exists at full coverage, but the directory still contains stale cards generated under older logic**

Therefore the current task is:

- **full rebuild in place**


## Full Rebuild Task (Running Now)

Script:

- [rebuild_case_skills_v3.py](/zhaoshu/subgraph/scripts/rebuild_case_skills_v3.py)

Current run:

- PID: `2506072`
- Log: [rebuild_case_skills_v3_20260407T101442Z.log](/zhaoshu/subgraph/logs/rebuild_case_skills_v3_20260407T101442Z.log)
- Report: [rebuild_case_skills_v3_full_20260407T101442Z.json](/zhaoshu/subgraph/reports/skill_v3_test/rebuild_case_skills_v3_full_20260407T101442Z.json)

This run uses:

- `--concurrency 32`

Progress watcher:

- [show_rebuild_case_skills_v3_progress.py](/zhaoshu/subgraph/scripts/show_rebuild_case_skills_v3_progress.py)

Command:

```bash
python scripts/show_rebuild_case_skills_v3_progress.py \
  --log /zhaoshu/subgraph/logs/rebuild_case_skills_v3_20260407T101442Z.log \
  --report /zhaoshu/subgraph/reports/skill_v3_test/rebuild_case_skills_v3_full_20260407T101442Z.json \
  --tail 20
```


## Why We Rebuilt At Concurrency 32

The earlier rebuild run at lower concurrency did not saturate available GPU/throughput.

To speed up LLM-backed skill regeneration, we now use:

```bash
PYTHONPATH=/zhaoshu/subgraph/src python /zhaoshu/subgraph/scripts/rebuild_case_skills_v3.py \
  --concurrency 32 \
  --report-path /zhaoshu/subgraph/reports/skill_v3_test/rebuild_case_skills_v3_full_<timestamp>.json
```

Notes:

- the script writes incremental progress to the report every 25 completed cases
- the log may remain empty briefly at startup because all first-wave concurrent requests are still in-flight


## What To Test After Rebuild

### A. Verify stale skill issues are gone

Check that cases like:

- `WebQTrn-610`
- `WebQTrn-1834`
- `WebQTrn-3507`

no longer have old dual-core relation artifacts unless genuinely intended.

Suggested quick scan:

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path('skills/webqsp_train_case_skills_v3')
bad = []
for p in root.glob('WebQTrn-*.json'):
    data = json.loads(p.read_text())
    if len(data.get('core_relations', [])) > 1:
        bad.append((p.stem, data.get('question'), data.get('core_relations')))
print('multi_core_count=', len(bad))
for row in bad[:50]:
    print(row)
PY
```


### B. Run 100-sample quality checks on good/bad subsets

Existing prepared splits:

- `tmp/prompt_tuning/old_bad_100.jsonl`
- `tmp/prompt_tuning/old_good_50.jsonl`
- `tmp/prompt_tuning/unstable_all.jsonl`

`old_bad_100` already exists and is directly usable.

If the team wants a larger good-case sample, either:

1. use the current `old_good_50.jsonl`, or
2. create a temporary `old_good_100.jsonl` by sampling more from the same source logic in [prepare_splits.py](/zhaoshu/subgraph/scripts/prompt_tuning/prepare_splits.py)


## Recommended 100-Sample Test Commands

### 1. Bad-case test (already ready)

```bash
PYTHONPATH=/zhaoshu/subgraph/src:/zhaoshu/subgraph python scripts/run_skill_enhanced_test.py \
  --data-path /zhaoshu/subgraph/tmp/prompt_tuning/old_bad_100.jsonl \
  --train-data-path /zhaoshu/subgraph/data/webqsp/webqsp_train.jsonl \
  --skills-root /zhaoshu/subgraph/skills/webqsp_train_case_skills_v3 \
  --limit-cases 100 \
  --max-concurrency 32 \
  --skill-top-k 3 \
  --label skill_v3_old_bad_100
```


### 2. Good-case test (current available split = 50)

```bash
PYTHONPATH=/zhaoshu/subgraph/src:/zhaoshu/subgraph python scripts/run_skill_enhanced_test.py \
  --data-path /zhaoshu/subgraph/tmp/prompt_tuning/old_good_50.jsonl \
  --train-data-path /zhaoshu/subgraph/data/webqsp/webqsp_train.jsonl \
  --skills-root /zhaoshu/subgraph/skills/webqsp_train_case_skills_v3 \
  --limit-cases 50 \
  --max-concurrency 32 \
  --skill-top-k 3 \
  --label skill_v3_old_good_50
```


### 3. Unstable-case test

```bash
PYTHONPATH=/zhaoshu/subgraph/src:/zhaoshu/subgraph python scripts/run_skill_enhanced_test.py \
  --data-path /zhaoshu/subgraph/tmp/prompt_tuning/unstable_all.jsonl \
  --train-data-path /zhaoshu/subgraph/data/webqsp/webqsp_train.jsonl \
  --skills-root /zhaoshu/subgraph/skills/webqsp_train_case_skills_v3 \
  --max-concurrency 32 \
  --skill-top-k 3 \
  --label skill_v3_unstable_all
```


## If The Team Needs 100 Good Cases

Current repo only has `old_good_50.jsonl` prebuilt.

The fastest path is:

- duplicate the split generation logic in [prepare_splits.py](/zhaoshu/subgraph/scripts/prompt_tuning/prepare_splits.py)
- change the good sample cap from `50` to `100`
- regenerate

Until then, use:

- `old_bad_100` for hard-regression checks
- `old_good_50` for non-regression sanity


## Recommended Order Of Work

1. Let the full `v3` rebuild finish.
2. Run the multi-core scan on rebuilt skills.
3. Run `old_bad_100`.
4. Run `old_good_50` (or regenerated `old_good_100` if prepared).
5. Compare:
   - core relation quality
   - action/result/final stage behavior
   - regressions on historically good cases


## Bottom Line

The code team should assume:

- the current `v3` directory needs a full logical refresh, not just spot fixes
- concurrency `32` is now the preferred rebuild setting
- the next reliable evaluation step is subset-based:
  - bad / unstable first
  - good/non-regression second

The most important currently running job is the `v3` full rebuild at concurrency `32`.
