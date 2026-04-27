# Action-Space Prompt/Skill V1

This is the next experimental workspace after freezing the 2026-03-31 snapshot.

## Experimental goal

Shift the system from:

- relation-centric planning + global answer-count heuristics

to:

- **primary action-space selection**
- **case-level final-selection experience**

The target runtime pattern is:

1. choose the action space whose **whole path semantics** best matches the question
2. treat final answer selection as:
   - keep all answers from that action space, or
   - filter to the appropriate subset within that action space
3. only use cross-action merging as a rare exception

## Frozen source version

- Snapshot:
  - `backups/actionspace_prompt_skill_snapshot_20260331/`

## New experimental output locations

- New skill corpus:
  - `skills/webqsp_train_case_skills_actionspace_v1/`
- New reports:
  - `reports/skill_enhanced_test/actionspace_prompt_skill_v1/`

## Priority failure slices to retest

### 1. Single-action subset filtering

These are the most important:

- `WebQTest-14`
- `WebQTest-39`
- `WebQTest-149`

Question:
- once the correct action space is chosen, can the new skill experience tell the model what evidence to inspect before keeping all vs keeping a subset?

### 2. Action-space misselection / wrong semantic path

- `WebQTest-109`
- `WebQTest-1027`
- `WebQTest-1428`

Question:
- can the prompt steer the model to choose the most semantically relevant **action space**, not just a tempting relation token?

### 3. True multi-action exceptions

- `WebQTest-1480`
- `WebQTest-1822`
- `WebQTest-1840`

Question:
- after adopting `single-action first`, do these rare set-like questions remain solvable as explicit exceptions?

## Prompt direction

Keep these global rules hard-coded:

- exact graph strings only
- full entity names only
- separate multiple answers cleanly
- rank the most credible answer first
- do not mix different semantic families in the final answer

Move these out of the global state-machine and into skill experience:

- whether to keep all answers from the chosen action
- when to inspect temporal / role / second-entity / granularity evidence
- when to avoid forced single-answer collapse

## Skill direction

The next skill iteration should generate, for each case, a short English `final_selection_experience` describing:

- what information matters for filtering
- what typically distinguishes the correct subset
- when not to collapse candidates

This should replace the current over-broad global `single/multiple/current/latest` answer rules.
