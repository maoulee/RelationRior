# Action / Filter Stage Refactor (2026-04-07)

## Purpose

This refactor implements the agreed workflow change:

1. `plan` is usually correct and should not be retried too early.
2. After action execution, the model should first judge whether the current action result is usable.
3. If backend exposes `Suggested Filter Relations`, the system should enter a dedicated filter stage before final reasoning.
4. Final reasoning still owns:
   - `keep_whole_action_space`
   - `filter-subset`
   - `collapse`


## New Runtime State Machine

The intended flow is now:

1. `plan`
2. `action` / `select_action`
3. `action-result judgment`
4. `filter stage` (only when suggested filter relations exist)
5. `final reasoning`

Important boundary:

- retry another **existing action** first
- only `re-plan` when the whole current action-space set is exhausted or clearly unusable


## Implemented Changes

### 1. Parser support for explicit stage decisions

File:

- [parser.py](/zhaoshu/subgraph/src/subgraph_kgqa/inference/parser.py)

Added parsing for:

- `<decision>retry_action</decision>`
- `<decision>proceed</decision>`
- `<decision>continue</decision>`

Also supports self-closing forms:

- `<retry_action/>`
- `<proceed/>`
- `<continue/>`

Parser output now includes:

- `stage_decision`


### 2. Runtime state flags

File:

- [state.py](/zhaoshu/subgraph/src/subgraph_kgqa/inference/state.py)

Added:

- `awaiting_action_judgment`
- `awaiting_filter_decision`
- `filter_stage_waived_action_id`

These are also included in state snapshots for debugging.


### 3. Action-result judgment handling

File:

- [runtime.py](/zhaoshu/subgraph/src/subgraph_kgqa/inference/runtime.py)

Added handler:

- `_handle_stage_decision(...)`

Behavior:

- If runtime is waiting for action judgment:
  - `retry_action`
    - clears current candidates
    - marks current action as failed
    - enters backtrack / action retry mode
  - `proceed`
    - requires `<candidates>`
    - advances to filter stage if suggested relations exist
    - otherwise advances directly to final reasoning


### 4. Dedicated filter stage handling

File:

- [runtime.py](/zhaoshu/subgraph/src/subgraph_kgqa/inference/runtime.py)

Behavior:

- If runtime is waiting for filter decision:
  - `filter(...)`
    - execute normally
  - `continue`
    - skip filter for the current action space
    - advance to final reasoning

Important detail:

- once `continue` is chosen for the current action, the legacy filter gate will not re-force filtering for the same action id


### 5. New phase hints

File:

- [plug_v12_feedback.py](/zhaoshu/subgraph/plug_v12_feedback.py)

Added:

- `_stage3_action_result_judgment(...)`
- `_stage4_filter_decision(...)`

And `PhaseHintGenerator.generate(...)` now prioritizes:

1. repair modes
2. `awaiting_filter_decision`
3. `awaiting_action_judgment`
4. normal stage detection


### 6. Prompt-level workflow updates

File:

- [sys_prompt.py](/zhaoshu/subgraph/sys_prompt.py)

Adjusted workflow language so that:

- action failure should first trigger retry of another existing action
- re-plan should only happen when the whole action-space set is insufficient
- suggested filter relations imply entry into a filter stage before final reasoning


## Current Expected Model Behavior

### After action execution

The model should output one of:

```xml
<reasoning>
- Current action result is clearly wrong because ...
</reasoning>
<decision>retry_action</decision>
```

or

```xml
<reasoning>
- Current action result is acceptable because ...
</reasoning>
<decision>proceed</decision>
<candidates>
  - Candidate A
  - Candidate B
</candidates>
```


### In filter stage

The model should output either:

```xml
<reasoning>
- These suggested filter relations are useful because ...
</reasoning>
<act>
  <query>filter(constraint_relations=["..."])</query>
</act>
```

or

```xml
<reasoning>
- Suggested filter relations are not useful for this action space.
</reasoning>
<decision>continue</decision>
```


## What This Refactor Does NOT Change

This refactor does **not** move final answer strategy out of reasoning.

Final reasoning still decides:

- whether to keep the whole selected action space
- whether to take a subset
- whether to collapse to one answer

The new filter stage only decides:

- inspect attributes via `filter(...)`
- or continue without doing so


## Validation Performed

Files compiled successfully:

- [state.py](/zhaoshu/subgraph/src/subgraph_kgqa/inference/state.py)
- [parser.py](/zhaoshu/subgraph/src/subgraph_kgqa/inference/parser.py)
- [runtime.py](/zhaoshu/subgraph/src/subgraph_kgqa/inference/runtime.py)
- [plug_v12_feedback.py](/zhaoshu/subgraph/plug_v12_feedback.py)
- [sys_prompt.py](/zhaoshu/subgraph/sys_prompt.py)

Parser smoke check confirmed:

- `retry_action`
- `proceed`
- `continue`

are all parsed correctly.

Phase hint smoke check confirmed:

- action-result judgment hint is emitted
- filter-stage hint is emitted


## Suggested Next Tests For Code Team

1. Replay `WebQTest-234`
   - verify the model retries another existing action before re-planning

2. Use a case with suggested filter relations
   - verify the system enters filter stage
   - verify `continue` suppresses repeated legacy filter forcing

3. Use a case with no suggested filter relations
   - verify `proceed` jumps directly to final reasoning

4. Verify a case with exhausted action ids
   - ensure `re-plan` is still allowed as fallback


## Summary

The system now supports the agreed separation:

- `plan` chooses action space
- action execution produces candidate evidence
- model explicitly judges whether to retry action or proceed
- backend-driven filter suggestions trigger a dedicated filter stage
- final reasoning remains responsible for answer-shaping decisions
