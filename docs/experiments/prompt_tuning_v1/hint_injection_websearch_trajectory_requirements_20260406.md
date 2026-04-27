# Hint Injection / WebSearch / Trajectory Requirements

Date: 2026-04-06  
Scope: `skill` injection, final-stage `web_search` prompting, case-level trajectory retention  
Audience: team implementers, experiment owners, analysis agents

## 1. Executive Summary

This document captures three concrete requirements for the current WebQSP runtime:

1. Fix what is currently wrong in stage-level hint injection.
2. Make final-stage `web_search` invocation an explicit, visible part of the reasoning protocol.
3. Preserve skill-related context in case-level trajectories so later analysis can reliably explain wins/regressions.

This document does **not** propose a rewrite of the main system prompt.  
The current direction is:

- keep the broad protocol stable,
- move optimization focus to `skill` usage and scaffold/runtime logic,
- inject better hints at the stages that actually decide success,
- improve trajectory observability so future tuning is evidence-based.

## 2. Current Diagnosis

### 2.1 What is wrong with current hint injection

The current injection design is stage-aware in code, but the effective guidance is misplaced:

- Stage 1 hinting is too early and too narrow.
- Stage 2-3 action-stage hinting is the real missing leverage.
- Stage 4/5 final-stage hinting is too generic and too biased toward "do not collapse".
- The runtime has `web_search` hinting/gating code, but that guidance is not currently strong, explicit, or well preserved in trajectories.

### 2.2 Relevant code points

- Runtime stage hint injection:
  - [runtime.py](/zhaoshu/subgraph/src/subgraph_kgqa/inference/runtime.py)
- Skill hint builders:
  - [retriever.py](/zhaoshu/subgraph/src/subgraph_kgqa/skill_mining/retriever.py)
- Experiment runner and trajectory markdown writer:
  - [run_skill_enhanced_test.py](/zhaoshu/subgraph/scripts/run_skill_enhanced_test.py)
- System protocol / optional web search text:
  - [sys_prompt.py](/zhaoshu/subgraph/sys_prompt.py)
- Frontend validation and gate wording:
  - [plug_v12_feedback.py](/zhaoshu/subgraph/plug_v12_feedback.py)

### 2.3 Evidence from current trajectories

Representative examples:

- [WebQTest-110 new](/zhaoshu/subgraph/reports/skill_enhanced_test/v12_protocol_guard_v5_full1494/trajectories/WebQTest-110.md)
- [WebQTest-111 new](/zhaoshu/subgraph/reports/skill_enhanced_test/v12_protocol_guard_v5_full1494/trajectories/WebQTest-111.md)
- [WebQTest-111 old](/zhaoshu/subgraph/reports/skill_enhanced_test/fulltest_skill_top3_protocol_guard_20260330/trajectories/WebQTest-111.md)

Observed pattern:

- relation-stage skill hints are visible,
- final-selection hints are visible,
- meaningful action-stage skill guidance is weak or absent,
- final-stage hints often degenerate into generic anti-collapse text,
- `web_search(...)` appears as a weak textual reminder instead of a first-class decision protocol,
- trajectories do not retain enough structured information about what skill bundle and stage hints were actually used.

## 3. Requirement A: Correct the Hint Injection Design

### 3.1 Current incorrect behavior

#### A1. Stage 1 is overused

Current Stage 1 relation hints can narrow the search space too early.

Problem:

- they can bias the model before planning has stabilized,
- in `top3`-style settings they may remove useful bridge relations too early,
- they risk replacing model planning with a narrow skill prior.

Example:

- In older successful runs, wider relation evidence helped preserve broader path semantics.
- In newer weaker runs, Stage 1 hinting often collapses to a small set of tempting relation tokens.

#### A2. Stage 2-3 is underused

Current action-stage hint injection exists in code but is not the effective center of guidance.

Problem:

- this is the stage where the model chooses among candidate actions / paths,
- but stage-specific skill experience often does not land there strongly enough,
- generic protocol text is doing more work than skill guidance.

This is a mismatch with the actual problem:

- many remaining errors are not pure final-answer errors,
- they are wrong path/action choices caused by weak semantic comparison between candidate chains.

#### A3. Stage 4/5 final hints are too generic

Current final-selection guidance often reduces to generic messages like:

- inspect graph-visible evidence,
- do not force collapse,
- do not force a single answer.

This is helpful for avoiding some bad single-answer collapses, but too weak for:

- choosing higher-level answers over member lists,
- deciding whether the question type is usually singular vs plural,
- deciding whether a no-time-scope role-holder question tends toward current/latest,
- clarifying what the question is actually asking for.

### 3.2 Target design

#### A1. Plan remains mostly free

Do not make `skill` the primary source of planning priors.

Target:

- Stage 1 should not strongly constrain plan formation.
- Stage 1 relation hints, if kept, should be weak and non-binding.
- The main role of `skill` should move later, after plan information exists.

#### A2. Action-stage is the main `skill` injection point

After plan is produced, `skill` should help action selection.

The action-stage experience should be soft guidance such as:

- focus on the most relevant core relations,
- do not pick a path only because one relation token looks attractive,
- judge the whole relation-chain logic when choosing the most suitable action.

Important:

- this is **not** a hard rule,
- it is a soft preference that helps the model compare candidate paths better.

#### A3. Final-stage hint should reflect question answering tendency

The final-stage experience should not be framed only as "collapse vs no collapse".

Instead it should help the model reason in this order:

1. What kind of answer is this question asking for?
2. For this kind of answer, is the usual output single or multiple?
3. If temporal scope is unspecified, is there a soft tendency such as current/latest?
4. What are common misreadings?

This should remain soft guidance, not hard constraints.

### 3.3 Concrete target structure for aggregated skill guidance

After weak filtering, the LLM should aggregate retained skills into **one markdown artifact** with multiple natural-language sections.

Each section should be written in natural language, not rigid taxonomy labels.

Each section should have content shaped like:

- what this kind of question is asking for,
- action selection cues,
- answering tendency,
- common misreadings.

Example tone:

- "When the question is asking for the role holder rather than the office or organization itself..."
- "When the question is asking for the broader containing region rather than member entities..."

This artifact is not injected all at once:

- action stage gets only action-oriented excerpts,
- final stage gets only answer-selection-oriented excerpts.

## 4. Requirement B: Add Explicit Final-Stage WebSearch Hinting

### 4.1 Current problem

The system already contains `web_search` prompt text and gate logic, but this capability is not sufficiently explicit in the final reasoning workflow.

Current behavior is too subtle because:

- the model is not clearly taught that it may request external verification as part of final reasoning,
- trajectories do not make the search decision process obvious,
- the search reminder often appears as one more feedback line rather than a visible reasoning protocol.

### 4.2 Design goal

At final stage, the model should clearly understand:

- `web_search(...)` is available,
- it is not a default step,
- it is appropriate when graph candidates exist but graph-visible evidence is insufficient for disambiguation,
- it should be used to compare current graph candidates only,
- the final answer must still come from the current graph candidate set.

### 4.3 Preferred behavioral design

The preferred behavior is:

1. The model reaches final-stage reasoning.
2. The model sees graph candidates.
3. If graph-visible evidence is enough, it answers directly.
4. If graph-visible evidence is not enough, it may request `web_search(...)`.
5. Search is used only to disambiguate current graph candidates.
6. The final answer remains graph-grounded.

### 4.4 What must change

#### B1. Final-stage hint must explicitly mention the option

The final-stage hint should explicitly state:

- if graph-visible evidence cannot distinguish current candidates,
- and a narrower choice is needed,
- the model may or must use `web_search(...)` before answering.

#### B2. This must be visible in trajectory

Case-level trajectories must show:

- whether a final-stage `web_search` hint was injected,
- whether the model chose to invoke `web_search`,
- whether the result was used for disambiguation.

#### B3. Search request should be framed as final-stage disambiguation

The model should not treat search as a replacement for graph reasoning.

The language should make clear:

- first use graph evidence,
- then request search only when needed,
- search is for verifying candidates, not inventing new ones.

### 4.5 Acceptance criteria

- At least one explicit final-stage section in prompt/feedback references `web_search(...)` as a valid disambiguation tool.
- The wording appears in case trajectories when the condition is met.
- The trajectory makes it possible to distinguish:
  - no search available,
  - search available but not needed,
  - search suggested,
  - search actually used.

## 5. Requirement C: Preserve Skill Context in Case-Level Trajectories

### 5.1 Current problem

Current trajectory markdowns only preserve partial skill information:

- retrieved case IDs,
- retrieval note,
- regex-extracted skill blocks from feedback.

What is missing or incomplete:

- the actual stage-by-stage skill hints used,
- the retained skill bundle metadata,
- whether skills were post-plan filtered,
- whether aggregated skill markdown was built,
- what aggregated stage hints were injected,
- what negative hints were injected,
- whether intent analysis or audit/conflict logic modified the bundle.

This makes later analysis too fragile.

### 5.2 What needs to be preserved

Each case trajectory should preserve enough information to answer:

1. What skill cards were initially retrieved?
2. What subset survived filtering / audit / conflict handling?
3. Was an aggregated skill artifact built?
4. What action-stage hint text was injected?
5. What final-stage hint text was injected?
6. Did negative plan/action hints exist?
7. Did intent analysis alter the final stage hint?
8. Did web search hinting appear?

### 5.3 Minimum trajectory additions

Add a structured "Skill Context" section per case:

- retrieved case IDs
- shortlisted case IDs
- selected case IDs
- retrieval note
- audit mode
- audit kept / dropped IDs
- conflict-detection note
- aggregation enabled or not
- aggregation summary
- stage hint availability summary:
  - relation-stage hint present?
  - action-stage hint present?
  - final-stage hint present?
  - web-search hint present?

Add turn-level retention:

- which stage hint key fired,
- exact stage hint text or truncated excerpt,
- whether it came from:
  - static stage hint,
  - dynamic skill bundle,
  - aggregated stage hint,
  - negative experience hint,
  - web-search hint,
  - stable-state repair hint.

### 5.4 Why this matters

Without this retention, later analysis cannot distinguish:

- skill truly had no effect,
- skill was present but too generic,
- action-stage hint was never injected,
- final-stage hint overrode action-stage signal,
- web-search was available but never surfaced,
- aggregated skill existed but was not what the model actually saw.

## 6. Non-Goals

The following are **not** part of this change set:

- rewriting the entire main system prompt,
- introducing hard rule-based answer constraints,
- enabling global always-on consistency,
- turning search into default behavior,
- making scaffold logic fully decide which semantic direction is valid.

## 7. Implementation Guidance

### 7.1 Hint injection refactor

Recommended direction:

- keep Stage 1 weak,
- strengthen Stage 2-3 action-stage guidance,
- redesign Stage 4/5 final-stage guidance around answer intention and answering tendency.

### 7.2 Aggregated skill artifact

The aggregated artifact should be one markdown object with multiple natural-language sections.  
It should support stage-specific extraction:

- action-stage excerpt builder
- final-stage excerpt builder

### 7.3 Trajectory writer

Update trajectory writing so it no longer relies only on regex extraction from feedback text.

The writer should consume structured runtime/result metadata whenever available.

Recommended source fields:

- `skill_bundle`
- `consistency`
- `trajectory[*].state_snapshot`
- stage hint metadata added at runtime

## 8. Suggested Acceptance Checklist

- [ ] Stage 1 no longer acts as the main narrowing point for skill-guided relation priors.
- [ ] Stage 2-3 action-stage hints are visibly present when action guidance is enabled.
- [ ] Stage 5 hints explicitly help with:
  - what kind of answer is being asked,
  - whether the answer type is usually single or multiple,
  - whether there is a soft current/latest tendency,
  - common misreadings.
- [ ] Final-stage `web_search(...)` capability is explicitly described in the reasoning flow.
- [ ] Case trajectories preserve structured skill-bundle context.
- [ ] Case trajectories preserve per-turn hint provenance and stage.

## 9. Recommended Next Work Split for Team

### Workstream 1: Runtime / hint injection

- adjust stage-role boundaries,
- reduce Stage 1 narrowing,
- strengthen Stage 2-3 guidance,
- redesign Stage 4/5 guidance semantics.

### Workstream 2: Aggregated skill artifact

- define the single-markdown multi-section format,
- define stage-specific extraction from the aggregated markdown,
- ensure generated content stays soft and natural-language.

### Workstream 3: Trajectory observability

- add structured hint provenance into turn snapshots,
- upgrade markdown trajectory writer,
- expose whether `web_search` hinting was available and used.

## 10. Short Team Message

The current problem is not that the runtime lacks hint infrastructure.  
The problem is that hints are landing at the wrong stages, the final-stage skill text is too generic, `web_search` is not visible enough as a final disambiguation option, and trajectories do not preserve enough skill context to debug regressions.

The next iteration should therefore optimize:

- stage placement of hints,
- quality and role of action-stage vs final-stage skill guidance,
- explicit final-stage web-search prompting,
- structured skill retention in per-case trajectories.
