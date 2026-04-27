# CWQ No-Skill Handoff: Entity/Relation Priors

## Current conclusion

The current bottleneck is not the tool API itself. The main failure mode is that we do not reliably expose the right **entity priors** and **relation priors** before planning.

The practical decomposition target should be:

1. identify the entities explicitly or implicitly referenced in the question
2. identify the verb-like / predicate-like relation intents in the question
3. use backend GTE retrieval to expand those entity and relation priors
4. let the model decide:
   - which entity should be the anchor
   - which relations are answer-bearing
   - the rough ordering between relations

At this stage we should **not** optimize relation-composition search yet.  
We should first test whether we can reliably recover the full prior set:

- prior entities
- prior relations

If this layer is wrong, downstream planning and action selection will keep drifting.

## What has already been changed

### 1. Stage pipeline prompt and flow changes

Files:
- `config/subagent_prompts.py`
- `scripts/run_stage_pipeline_test.py`

Changes already made:
- `Agent0` decomposition prompt was revised several times to support `MAIN/ATTR` style decomposition with examples.
- `Agent2` planning prompt was revised to bias anchor selection, related relations, and constraints.
- Stage pipeline now supports:
  - decomposition
  - automatic GTE retrieval
  - planning
  - action execution
  - automatic filter
- duplicate action-space text in Stage 3 input was removed
- action execution upper bound was increased from `2` to `5`
- `filter_mode` is now tracked in state

Important note:
- these changes improve mechanics, but they do **not** solve the core prior-selection problem

### 2. GTE retrieval pool fix

File:
- `scripts/run_stage_pipeline_test.py`

Bug fixed:
- entity ranking previously only ran inside a narrow `verified_entities` pool
- it now uses backend semantic retrieval over the full sample subgraph

Effect:
- bridge entities like `Forrest Gump` can now appear in candidates for cases such as `WebQTrn-962`

### 3. Backend CVT expansion adjustment

File:
- `/zhaoshu/SubgraphRAG-main/retrieve/graph_server.py`

Change:
- `expand_node_v2()` now expands outgoing neighbors first
- if no outgoing expansion exists, it falls back to incoming neighbors

Important limitation:
- this does **not** solve numeric literal decoding
- for cases like `WebQTrn-60`, `g.xxx` nodes are already CVT-like but often do not expose a readable numeric value such as `1.8`

### 4. Backend path enumeration debugging

File:
- `/zhaoshu/SubgraphRAG-main/retrieve/graph_server.py`

Experimental flags were added:
- `KGQA_PATH_EARLY_STOP`
- `KGQA_PATH_GLOBAL_VISITED`

Observation:
- disabling early stop and global visited significantly increases action-space recall
- example: `WebQTrn-962` grows from `4` hints to `22` hints and starts exposing GT-hitting paths

Important limitation:
- larger action space alone is not enough
- Stage 3 may still choose the wrong shortest-looking path

## What has been empirically confirmed

### A. Backend is not the main speed problem

Observed:
- direct backend API calls are fast
- earlier perceived slowness mainly came from local scripts repeatedly loading `pkl` data

### B. `find_paths_with_relation` recall was too aggressive

Confirmed:
- early stop and global visited pruning hide deeper but valid paths

But:
- fixing recall does not automatically fix selection

### C. The real architecture issue is prior extraction

Across recent CWQ debugging, the most stable interpretation is:

- anchor selection is usually recoverable if we surface the right entity candidates
- relation selection is harder and depends on exposing the right verb/predicate prior
- the current system still jumps too early into plan/action before building this prior layer cleanly

## The architecture that should be tested next

The intended architecture is:

### Step 1: Prior extraction from question text

The model should output two groups:

- `entity_priors`
  - noun phrases
  - quoted names
  - named entities
  - answer-type-supporting bridge entities when obvious

- `relation_priors`
  - verb phrases
  - predicate-like intents
  - answer-bearing relation hints
  - check/filter relation hints

This is **not** full planning yet.

The model is only asked:
- what entities are relevant
- what relation intents are relevant

### Step 2: Backend GTE expansion

For each prior:

- entity prior -> `semantic_retrieve` over full subgraph entities
- relation prior -> `semantic_retrieve` over full subgraph relations

The backend returns:
- ranked entity candidates per prior
- ranked relation candidates per prior

### Step 3: Model decides anchor and rough ordering

Only after prior expansion, ask the model:
- which entity should be anchor
- which relations look answer-bearing
- what rough order between relations is plausible

At this stage:
- do not yet search for optimal relation composition
- do not yet overfit action-space selection

First target:
- recover the full useful prior set

## What should be tested now

The immediate test objective is:

> Can we reliably recover all useful prior entities and prior relations from the question?

That means:

1. ask the model to list:
   - entity priors
   - relation priors
2. run GTE for each prior
3. inspect whether:
   - anchor candidate appears
   - bridge entity appears
   - answer-bearing relation appears
   - check/filter relation appears

### This is different from the previous diagnostic

The previous quick diagnostic asked the model to generate commands like:
- `find entities about ...`
- `find relations about ...`

That test was useful, but it is **not exactly the desired interface**.

The intended interface should be closer to:

```json
{
  "entity_priors": ["Lou Seal", "World Series", "team with mascot Lou Seal"],
  "relation_priors": ["mascot -> team", "team -> championships", "latest championship year"]
}
```

The backend should then turn each prior into a GTE retrieval call.

## Recommended prompt contract

The model should not plan yet.

It should produce something like:

```json
{
  "entity_priors": [
    "...",
    "..."
  ],
  "relation_priors": [
    "...",
    "..."
  ]
}
```

Guidelines:
- entity priors should be short noun/entity phrases
- relation priors should be short predicate/verb phrases
- preserve explicit entities from the question whenever possible
- include bridge entities only if they are strongly implied
- do not compose paths
- do not output actions
- do not choose final anchor yet

## Example interpretation

### Example: Lou Seal

Question:
- `Lou Seal is the mascot for the team that last won the World Series when?`

Desired output:

```json
{
  "entity_priors": [
    "Lou Seal",
    "World Series",
    "team with mascot Lou Seal"
  ],
  "relation_priors": [
    "mascot of team",
    "team championships",
    "latest championship year"
  ]
}
```

### Example: Libya anthem

Question:
- `Which man is the leader of the country that uses Libya, Libya, Libya as its national anthem?`

Desired output:

```json
{
  "entity_priors": [
    "Libya, Libya, Libya",
    "country using this anthem",
    "leader of that country"
  ],
  "relation_priors": [
    "anthem to country",
    "country to leader",
    "office holder"
  ]
}
```

### Example: Portuguese + child labor

Question:
- `In which countries do the people speak Portuguese, where the child labor percentage was once 1.8?`

Desired output:

```json
{
  "entity_priors": [
    "Portuguese",
    "countries speaking Portuguese",
    "countries with child labor percentage 1.8"
  ],
  "relation_priors": [
    "language spoken in country",
    "country child labor percentage",
    "filter by 1.8"
  ]
}
```

## Known case-level observations

### `WebQTrn-962`

The problem is not only decomposition.

Confirmed:
- deeper valid graph paths exist
- backend pruning had been hiding some of them
- after widening retrieval, `Forrest Gump` can appear

But:
- this case still shows that the system needs stronger bridge-entity and role/predicate priors

### `WebQTrn-60`

The problem is not backend speed.

Confirmed:
- backend is fast
- numeric constraint handling is weak
- `1.8` behaves more like a relation/value constraint than a stable entity candidate

This supports the prior-layer design:
- entities and relations should be treated separately

## Deliverables already available

Recent reports:
- `reports/stage_pipeline_test/cwq_main_attr_10_smoke_v2/report.md`
- `reports/stage_pipeline_test/cwq_main_attr_10_smoke_v3/report.md`
- `reports/stage_pipeline_test/direct_decomposition_10x5/report.md`
- `reports/stage_pipeline_test/find_check_decomposition_10cases/report.md`
- `reports/stage_pipeline_test/find_check_plan_pilot_10cases/results.json`
- `reports/stage_pipeline_test/962_action_space_growth_experiment.json`

Recent diagnostic script:
- `scripts/diag_llm_query_gte.py`

Note:
- `diag_llm_query_gte.py` was a quick command-style retrieval diagnostic
- it is useful as a reference, but it should probably be replaced by a cleaner `entity_priors / relation_priors` diagnostic

## Recommended next action for team

The team should implement a narrow experiment only:

1. input: raw question text
2. model outputs:
   - `entity_priors`
   - `relation_priors`
3. backend runs GTE retrieval for each prior
4. report:
   - which priors recover useful anchor candidates
   - which priors recover useful bridge entities
   - which priors recover answer-bearing relations
   - which priors recover check/filter relations

No action execution needed yet.
No plan composition search needed yet.

That experiment should answer whether the prior layer is learnable and stable.
