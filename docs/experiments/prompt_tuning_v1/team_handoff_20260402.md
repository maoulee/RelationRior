# Prompt Tuning / Skill / Consistency Team Handoff

Date: `2026-04-02`

This note is the current handoff artifact for moving the workflow into a larger
team-mode setup without losing the reasoning behind past decisions.

## 1. What is current best

The strongest broad configuration so far is:

- Stage-5 prompt variant: `v5_filter_then_answer`
- low temperature
- `conflict_only` skill audit
- `per_skill` skill reasoning injection
- **no global always-on decision consistency**

Reference summaries:

- `/zhaoshu/subgraph/reports/skill_enhanced_test/prompt_tuning_v1/archive_summary_20260401.md`
- `/zhaoshu/subgraph/reports/skill_enhanced_test/prompt_tuning_v1/full_decision_consistency_summary.md`

## 2. Version history and what changed

### Prompt branch evolution

- `v0_baseline`
  - Untuned Stage-5 baseline.
- `v1_filter_value_strict`
  - Interpreted filter output more aggressively at the value level.
  - Result: often over-pruned or over-constrained.
- `v2_no_unsupported_collapse`
  - Penalized unsupported single-answer collapse very strongly.
  - Result: hurt subset selection and several inventory/canonical splits.
- `v3_primary_action_semantics`
  - Increased emphasis on whole-path semantics during final selection.
  - Result: not enough net gain over baseline.
- `v4_concise_final_reasoning`
  - Compressed final reasoning heavily.
  - Result: too much useful signal was lost.
- `v5_filter_then_answer`
  - Enforced the sequence:
    - primary action
    - filter
    - subset/keep-all
    - answer
  - Result: best prompt-only branch and current default.

### Skill audit branch evolution

- `top10 -> audit -> keep3`
  - Good for a few semantic-family mismatch cases.
  - Net regression on broader samples.
  - Archived as a non-default branch.
- `lowtemp + conflict_only`
  - Near tie on broader mixed50.
  - Clear win on targeted unstable subsets.
  - Best practical audit mode so far.

### Consistency branch evolution

- Global turn-level consistency from early turns
  - Good on targeted unstable cases.
  - Negative on the full dataset.
  - Main failure mode: it changes real behavior, not just noise.
  - It reduces frontend errors, but hurts overall accuracy when always-on.

## 3. What the evidence says

### Prompt-only tuning has largely plateaued

The project is no longer bottlenecked on broad prompt proliferation.
`v5` gave the one real prompt-level gain; later prompt variants did not
reliably beat it.

### Broad skill audit is too expensive in accuracy

Using a wide retrieved pool and then auditing aggressively introduces too much
noise. The best default remains conservative:

- smaller skill set
- only conflict-aware intervention
- do not always summarize or rewrite skill experience

### Global consistency is not noise control; it is a real mechanism change

From the full consistency analysis:

- improvements: `122`
- regressions: `122`
- but regressions are larger in magnitude
- only a small fraction is attributable to simple sampling noise

Implication:

- do **not** enable consistency globally
- if consistency is used, it should be selective and likely final-stage only

## 4. The two most promising next directions

### Direction A: final-stage-only selective consistency

Use consistency only for the final answer-selection step, and only on cases
that exhibit real risk signals:

- frontend error already happened
- turn count is high
- final candidates remain ambiguous
- filter evidence exists but subset choice is unstable

Additional guard:

- filter hash-only signatures from voting

This is the shortest path to a deployable stability improvement.

### Direction B: skill construction from repeated supervised runs

For each training sample:

1. run the model `k` times
2. score the runs with the existing reward / GT-aware evaluation
3. identify instability patterns:
   - common misreadings
   - wrong answer family
   - wrong specificity level
   - inventory vs canonical confusion
   - temporal/current vs historical confusion
4. write those signals back into the skill

This direction is more ambitious but likely more valuable long-term.

The key shift is:

- not just storing relation hints
- but storing **intent clarification** and **common failure modes**

Examples of future skill fields:

- `intent_clarification`
- `common_misreadings`
- `instability_triggers`
- `wrong_but_related_answer_families`

## 5. What to avoid repeating

- Do not treat early `9002` zero-result runs as valid evidence.
- Do not keep broad always-on `top10 -> keep3` audit as a default.
- Do not use global always-on consistency as a default.
- Do not keep adding special-case prompt exceptions for individual failures.

## 6. Directory map

### Active references

- reports:
  - `/zhaoshu/subgraph/reports/skill_enhanced_test/prompt_tuning_v1`
- docs:
  - `/zhaoshu/subgraph/docs/experiments/prompt_tuning_v1`
- prompt variants:
  - `/zhaoshu/subgraph/src/subgraph_kgqa/prompt_variants/variants.py`
- reusable helper tools:
  - `/zhaoshu/subgraph/scripts/prompt_tuning/tools`

### Archived material

- invalid `9002` runs:
  - `/zhaoshu/subgraph/reports/skill_enhanced_test/prompt_tuning_v1/_archive/invalid_glm9002_runs_20260401`
- legacy prompt-only branches:
  - `/zhaoshu/subgraph/reports/skill_enhanced_test/prompt_tuning_v1/_archive/legacy_prompt_variants_20260402`
- legacy skill-audit branches:
  - `/zhaoshu/subgraph/reports/skill_enhanced_test/prompt_tuning_v1/_archive/legacy_skill_audit_20260402`
- obsolete misc / smoke:
  - `/zhaoshu/subgraph/reports/skill_enhanced_test/prompt_tuning_v1/_archive/obsolete_misc_20260402`
- raw execution logs:
  - `/zhaoshu/subgraph/reports/skill_enhanced_test/prompt_tuning_v1/_archive/raw_logs_20260402`

## 7. Recommended team split

### Team A: selective consistency

- implement final-stage-only gate
- add hash-signature filtering
- rerun on unstable/high-risk subsets first

### Team B: skill supervision

- design `k`-run training replay
- build instability-aware skill fields
- test whether improved skill semantics reduce answer-family mismatch

### Team C: tool-side fallback

- keep web-search work isolated
- use it only as a fallback for:
  - evidence insufficiency
  - repeated final-stage disagreement

## 8. Concrete next experiment order

1. selective final-stage consistency
2. hash-signature filtering
3. repeated-run supervised skill construction
4. optional search fallback after unresolved final disagreement

That order keeps the fast implementation path separate from the more ambitious
skill redesign path.
