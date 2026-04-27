# Skill Refactor Handoff (2026-04-06)

## Goal

This document records the current agreed direction for `case skill` generation and selection, based on the recent debugging round around:

- wrong `core_relation` selection
- over-reliance on shortest-path heuristics
- mismatch between question wording and graph-reachable answer paths
- poor fallback behavior when original trajectories do not contain usable `plan` relations

The intended audience is the code team implementing the next round of `skill` refactors.


## High-Level Decisions

### 1. `core_relation` should not be selected from oracle / shortest-path by default

The original bug was that `core_relation` could be selected from:

- `ground_truth.core_relations`
- `ground_truth.oracle_relations`
- shortest-path derived relations

even when those relations were not actually used by the model in its original `plan`.

That behavior is not faithful to the training-time meaning of the reward signal.

The accepted rule is now:

1. If the original trajectory already contains usable `plan` relations, choose `core_relation` only from those trajectory relations.
2. Only if trajectory `plan` relations are unavailable, fall back to shortest connected paths between anchor and answer.
3. Only in that fallback path-selection scenario should the LLM decide which path is best and which relation inside that path is the `core_relation`.


### 2. Reward is trajectory-conditioned, not shortest-path-conditioned

The reward signal should be interpreted as:

- model proposes `plan` relations
- those relations induce an action space
- the induced action space is scored

It should **not** be interpreted as:

- find a shortest path first
- then use that shortest path to define the main relation

This distinction matters because shortest paths can leak oracle structure and over-credit graph-near relations that the model never actually planned.


### 3. Shortest-path fallback is still useful, but only as fallback

When no usable trajectory `plan` relations are available, we still need a way to recover a meaningful `core_relation`.

The accepted fallback is:

1. Build shortest connected path candidates from anchor entity to answer entity.
2. Provide those paths to the LLM.
3. Ask the LLM to:
   - choose the winning path
   - choose the single `core_relation` inside that path

This is intentionally path-level, not relation-level, because relation choice depends on path semantics.


### 4. `answer_strategy` should be partly structured, partly LLM-authored

The current accepted direction is:

- keep `action_space_mode` fixed to one of:
  - `keep_whole_action_space`
  - `filter_within_action_space`
  - `collapse_within_action_space`

- but let the LLM write:
  - `filter_likely_attributes`
  - `selection_rule`

This gives enough structure for downstream use while still allowing case-level semantic variation.


## Current Implementation Status

### Files touched

- [case_skill.py](/zhaoshu/subgraph/src/subgraph_kgqa/skill_mining/case_skill.py)
- [extractor.py](/zhaoshu/subgraph/src/subgraph_kgqa/skill_mining/extractor.py)
- [build_case_skills.py](/zhaoshu/subgraph/scripts/build_case_skills.py)


### Already implemented in `case_skill.py`

#### A. Reward-first relation ranking helpers

Implemented:

- reward cache loading from retrieval training artifacts
- relation-level reward aggregation
- ranking helpers for candidate relations

Key functions:

- `_load_train_relation_reward_index(...)`
- `_rank_core_relations_by_reward(...)`
- `_select_core_relations_by_reward(...)`


#### B. Trajectory relation candidate loading

Implemented:

- load local `source_cards`
- load local `raw_materials`
- recover `planned_relations` candidates from those artifacts
- recover prompt-side relation candidates when available

Key functions:

- `_load_local_source_card(...)`
- `_load_local_raw_attempts_for_source(...)`
- `_trajectory_relation_candidates(...)`
- `_dataset_prompt_relation_candidates(...)`


#### C. Shortest-path fallback candidate generation

Implemented:

- build shortest connected path candidates from training graph object
- expose path candidates as ordered relation chains
- allow LLM to choose winning path + core relation

Key functions:

- `_shortest_path_candidates_for_case(...)`
- `_llm_pick_winning_shortest_path(...)`


#### D. Gating logic

Implemented:

- if trajectory candidates exist: choose only from trajectory candidates
- if trajectory candidates do not exist: allow shortest-path fallback
- if neither exists: use broader reward/rule fallback

This gating is now active in:

- `synthesize_case_skill(...)`
- `synthesize_case_skill_from_dataset_case(...)`


#### E. Parallel case-skill building

Implemented:

- `build_case_skill_outputs(...)` now supports async concurrency
- `scripts/build_case_skills.py` accepts `--concurrency`

This only parallelizes synthesis calls. File writes still happen in a stable sequential phase.


### Already implemented in `extractor.py`

The old extractor only recovered `planned_relations` from trajectory lines like:

- `[Relations]: rel1, rel2`

This missed many valid plans written directly as:

- `plan(... related=[...], maybe_related=[...])`

Implemented fix:

- `parse_smoke_report(...)` now also extracts relations from:
  - `related=[...]`
  - `maybe_related=[...]`

This is required so `source_cards/*.json` stop under-reporting `planned_relations_seen`.


## Observed Behavior From Debugging

### Case: `WebQTrn-0`

Question:

- `what is the name of justin bieber brother`

Important finding:

- the original trajectory did contain sibling relations
- the old extractor failed to capture them
- this incorrectly forced the case into fallback mode

After fixes:

- `core_relation` is now correctly stabilized as:
  - `people.person.sibling_s`


### Case: `WebQTrn-6`

Question:

- `who does joakim noah play for`

Important finding:

- raw/source trajectory already contained:
  - `sports.pro_athlete.teams`
  - `sports.sports_team_roster.team`

Therefore:

- fallback should not override these
- shortest-path candidates are only diagnostic here, not authoritative

Current correct behavior:

- `core_relation = sports.pro_athlete.teams`


### Cases: `WebQTrn-8`, `WebQTrn-72`

Questions look like:

- `where did saki live`
- `where does jackie french live`

Important finding:

- although the wording suggests a residence relation,
- the shortest graph-reachable answer path in the training data is actually `place_of_birth`

Therefore these should **not** automatically be treated as wrong simply because `lived` is more natural surface wording.

The correct interpretation is:

- graph semantics dominate if trajectory relations are absent
- shortest connected path can legitimately resolve to `place_of_birth`
  when that is the best answer-bearing graph path to the GT


## What The Code Team Should Preserve

### Preserve this invariant

If trajectory plan relations are present, the LLM must not freely replace them with unrelated graph-near relations.

More concretely:

1. trajectory candidates first
2. shortest-path fallback only if no trajectory candidates
3. broader fallback only if both of the above fail


### Preserve this invariant

`action_space_mode` remains structured.

Do not turn the whole answer strategy into unconstrained prose. We still want:

- stable downstream parsing
- controlled behavior categories


## What The Code Team Should Improve Next

### 1. Rebuild `source_cards` / `raw_materials`

This is the most important operational next step.

The current local `skills/source_cards/*.json` were generated before the extractor fix, so many still under-report:

- `planned_relations_seen`

Without a rebuild, some cases will continue to fall into shortest-path fallback unnecessarily.


### 2. Improve path-level explanation artifact

Right now shortest-path fallback chooses:

- winning path
- core relation

But the chosen path itself is not yet persistently written into the final case skill card.

Recommended next addition:

- store the chosen path in the card metadata or notes

Suggested field:

- `selected_core_path`

This will make later debugging much easier.


### 3. Improve strategy prompt quality

The current strategy prompt is better than before, but still somewhat generic.

Recommended refinement:

- keep `action_space_mode` fixed
- explicitly ask the LLM to explain:
  - whether the selected action space should be kept whole
  - or filtered within
  - and if filtered, which attribute usually distinguishes the answer

This is especially useful for:

- temporal holder questions
- role/title disambiguation
- multi-candidate entity sets


### 4. Optionally add per-case diagnostics

Recommended report additions:

- whether a case used:
  - trajectory plan selection
  - shortest-path fallback
  - relation-only fallback

- selected path text when fallback used
- reward ranking over candidate trajectory relations


## Suggested Execution Plan For Code Team

1. Rebuild source/raw artifacts with the improved extractor.
2. Re-run a 50-case skill build with concurrency enabled.
3. Produce a summary with:
   - count of cases using trajectory candidates
   - count of cases using shortest-path fallback
   - count of cases with empty candidates
4. Review fallback-heavy cases separately before expanding to larger scale.


## Useful Commands

Rebuild extraction artifacts:

```bash
PYTHONPATH=/zhaoshu/subgraph/src python scripts/extract_skill_cards.py \
  --batch-report reports/inference_runtime_batch/prompt_branch30/protocol_guard_action_id_experiment.md \
  --data-path data/webqsp/webqsp_train.jsonl \
  --skills-root skills \
  --smoke-reports-dir reports \
  --kg-api-url http://127.0.0.1:8001 \
  --limit 30
```

Build case skills concurrently:

```bash
PYTHONPATH=/zhaoshu/subgraph/src python scripts/build_case_skills.py \
  --skills-root skills \
  --sample 50 \
  --seed 0 \
  --concurrency 8
```


## Bottom Line

The current agreed architecture is:

- **use trajectory plan relations when available**
- **use shortest connected paths only as fallback**
- **let the LLM choose a winning fallback path and core relation**
- **keep answer strategy semi-structured**

The single highest-leverage operational task now is:

- **rebuild the extraction artifacts so trajectory relations are no longer missing**
