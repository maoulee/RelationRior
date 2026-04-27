# Prompt Tuning Tools

This folder stores reusable analysis and observability helpers for the
`prompt_tuning_v1` workflow so we do not keep rewriting one-off scripts.

## Tool inventory

- `agent_worklog.py`
  - Wraps shell steps and writes JSONL + Markdown progress logs.
- `consistency_ab_compare.py`
  - Compares case-level deltas between two consistency arms.
- `small_delta_inspector.py`
  - Buckets small-magnitude changes to separate likely noise from real effects.
- `summarize_decision_consistency.py`
  - Builds a summary report from decision-consistency experiment outputs.
- `trajectory_inspector.py`
  - Extracts per-case tool-call and trajectory comparisons.

## What stays outside this folder

Entry scripts remain in `scripts/prompt_tuning/`:

- `prepare_splits.py`
- `run_variant_screening.py`
- `compare_variants.py`
- `run_decision_consistency_suite.sh`

Those are the top-level workflows. The helpers in `tools/` support them.
