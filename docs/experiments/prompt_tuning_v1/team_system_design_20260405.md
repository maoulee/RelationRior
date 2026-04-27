# KGQA System Redesign Memo

Date: `2026-04-05`

Owner context: this memo is the current detailed handoff for the WebQSP-focused
KGQA system after multiple rounds of prompt tuning, skill-injection changes,
consistency experiments, and harness stabilization work.

This document is intentionally detailed. It is written for a team that needs:

- the current best-known broad configuration,
- the experimental lessons that should not be relearned,
- the current diagnosis of why recent versions regressed,
- the target architecture for the next iteration,
- a concrete implementation scope that avoids disturbing the main prompt layer.

---

## 1. Executive Summary

The project has already extracted most of the easy gains from Stage-5 prompt
tuning. The best broad-performing configuration remains:

- `variant = protocol_guard_action_id_experiment`
- Stage-5 prompt variant = `v5_filter_then_answer`
- low temperature
- conservative `conflict_only` skill audit
- no global always-on consistency

Reference report:

- [decision_consistency_full_armA/report.md](/zhaoshu/subgraph/reports/skill_enhanced_test/prompt_tuning_v1/decision_consistency_full_armA/report.md)

Recent regressions are **not** best explained by the harness fixes themselves.
The larger drop came from changing multiple core behaviors at once:

- reverting to looser prompt/protocol settings,
- weakening skill coverage and skill usefulness,
- letting pre-plan skill priors interfere with planning,
- and allowing answer shaping to become too generic.

The current redesign direction should therefore be:

1. Freeze the main prompt layer for now.
2. Continue improving the harness/scaffold layer to reduce unnecessary
   instability.
3. Redesign skill usage so that:
   - planning stays mostly free,
   - skills are consumed **after plan generation**,
   - skills are aggregated into a single markdown artifact,
   - the artifact is split into natural-language semantic directions,
   - action-stage and final-stage experience are injected separately.
4. Treat external web search as a **final-stage verification fallback**, not a
   default reasoning tool.

This is the main message to carry into team mode:

- **Do not keep optimizing broad prompt wording.**
- **Do not hard-code more rules.**
- **Do not restore aggressive pre-plan skill priors.**
- **Do improve harness stability and post-plan skill aggregation.**

---

## 2. Best Known Broad Configuration

The strongest broad configuration we have confirmed is:

- report:
  - [decision_consistency_full_armA/report.md](/zhaoshu/subgraph/reports/skill_enhanced_test/prompt_tuning_v1/decision_consistency_full_armA/report.md)
- settings:
  - `protocol_guard_action_id_experiment`
  - `v5_filter_then_answer`
  - low temperature
  - `conflict_only`
  - no global always-on decision consistency

Observed broad metrics on full `1494`:

- `Avg F1 = 0.7918`
- `Hit@1 = 0.8213`
- `Hit@0.8 = 0.7222`
- `EM = 0.6948`

Important interpretation:

- this is the current broad “best known baseline”,
- not every sub-experiment beats it,
- any new branch should be compared back to this configuration unless the
  experiment is explicitly diagnostic-only.

---

## 3. What We Now Believe About the System

### 3.1 Prompt tuning has mostly plateaued

Prompt-only tuning no longer appears to be the main bottleneck.

What we learned:

- `v5_filter_then_answer` gave the one durable prompt-level gain.
- Later prompt variants mostly shuffled failure modes around.
- Overly strict or overly compressed prompt variants often caused regression.

Implication:

- the main prompt layer should be treated as **stable for now**,
- further gains should come from skill usage, harness behavior, and final-stage
  verification.

### 3.2 The real bottleneck is no longer broad planning text

Historically:

- skills helped most on action-space / relation selection,
- not much on broad planning itself,
- then more cases started reaching plausible candidates,
- and final selection became the new visible bottleneck.

However, more recent re-analysis suggests:

- on subsets of unstable or fully wrong cases,
- the dominant failure bucket is still often “GT never enters the candidate
  pool”.

Important terminology clarification:

- some newer analyses label “GT not in candidate set” as `plan` error,
- older analyses labeled the same phenomenon mostly as `action` or
  action-space miss.

These are not true contradictions. They are largely labeling differences.

The practical conclusion stays the same:

- the most important upstream decision is **which action-space / path family the
  model commits to**,
- and the most important downstream decision is **how it filters or preserves
  the candidates inside that chosen action-space**.

### 3.3 Global always-on consistency is not the right default

Global turn-level consistency:

- helps on some unstable/high-risk subsets,
- but hurts full-distribution accuracy when always enabled,
- changes real behavior, not just random noise.

Implication:

- do not enable global consistency by default,
- if consistency returns later, it should likely be:
  - selective,
  - final-stage only,
  - and probably triggered only on high-risk cases.

### 3.4 Broad skill audit is too blunt

Aggressive skill auditing such as `top10 -> audit -> keep3` was not robust on
broader samples.

The currently better idea is:

- keep retrieval conservative,
- only apply light filtering,
- and let the LLM use a richer post-plan skill aggregation artifact rather than
  hard-pruning the skill set too early.

---

## 4. What Went Wrong In Recent Regressed Versions

Recent lower-performing branches did **not** prove that harness work is bad.
Instead, they typically changed too many things at once.

### 4.1 Broad configuration drift

Compared with the best broad branch, some regressed versions simultaneously:

- switched from `protocol_guard_action_id_experiment` back to looser variants,
- reverted from `v5_filter_then_answer`,
- loosened temperature / top-p,
- disabled or weakened skill conflict handling,
- reduced skill coverage,
- and weakened the usefulness of skill injection content.

This makes it hard to interpret raw regression as “the harness patch caused the
drop”.

### 4.2 Skills became weaker, flatter, and more generic

Recent traces strongly suggest:

- relation priors still sometimes help,
- but final-selection skill injection often degenerates into generic advice like
  “inspect evidence” and “do not force collapse”,
- which is too weak to help in semantically subtle cases.

Representative pattern:

- [WebQTest-111 old success](/zhaoshu/subgraph/reports/skill_enhanced_test/fulltest_skill_top3_protocol_guard_20260330/trajectories/WebQTest-111.md)
- [WebQTest-111 newer failure](/zhaoshu/subgraph/reports/skill_enhanced_test/v2.1.3_hit1_fix_full1494/trajectories/WebQTest-111.md)

Observed failure mode:

- the old version had broader and more useful skill coverage,
- the new version retained only generic final-selection guidance,
- so the model was no longer really guided on semantic level choice.

### 4.3 Pre-plan skill injection drifted away from original design

The original desired design was:

1. let the model produce a plan mostly on its own,
2. then use the plan to filter and aggregate relevant skills.

The current implementation drifted toward:

1. retrieve skills first,
2. build relation priors first,
3. inject relation-stage hints before planning.

This is a major design drift.

Why this matters:

- it makes skill act like a hidden planner,
- it can bias planning even when planning was already mostly correct,
- and it makes debugging much harder because planning and skill influence are
  entangled too early.

### 4.4 Skill generation itself is not “bad”, but the current schema is too overlapping

Current skill cards still carry overlapping fields:

- `answer_strategy`
- `action_space_experience`
- `final_selection_experience`
- `common_pitfalls`
- plus some newer instability-aware fields

Problem:

- these fields are not fully aligned,
- they may partially repeat each other,
- and injection code sometimes mixes them again into one merged hint.

So the deeper problem is:

- not pure duplication,
- but **schema overlap + weak consistency control + over-merged injection**.

---

## 5. Harness / Scaffold State: What Is Good And What Still Needs Work

### 5.1 Improvements already made

Recent scaffold improvements already moved in the right direction:

- stable intermediate state tracking was introduced,
- repair can now reference prior valid intermediate state,
- consistency got better access to real candidates,
- some frontend error handling improved,
- some search truncation and reporting issues were patched.

These are real engineering improvements and should be preserved.

### 5.2 Remaining harness weaknesses

The current harness still needs more work in several places:

1. failed cases are still too easy to lose from evaluation accounting in some
   runners,
2. configuration is not always fully persisted into machine-readable manifests,
3. parser robustness is still weaker than it should be,
4. frontend repair still behaves too much like regeneration rather than patching,
5. branch answer aggregation is too permissive for WebQSP-like single-plan
   questions,
6. search triggering is still too implicit and too narrow.

### 5.3 Main harness design principle going forward

The harness should do one thing extremely well:

**preserve verified useful intermediate structure and avoid unnecessary
behavioral drift after errors**.

This means:

- if the model already selected a good action-space, preserve it,
- if candidates were already correctly extracted, preserve them,
- if filter evidence already exists, preserve it,
- if only one argument is missing or malformed, patch that part instead of
  regenerating the whole turn.

This is more important right now than further prompt changes.

---

## 6. Updated Architecture Direction

This is the current recommended target architecture.

### 6.1 Main prompt layer

For now:

- do **not** do large redesigns of the main prompt,
- keep the existing broad protocol stable,
- treat main prompt changes as low priority unless they are required for a new
  tool or stage protocol.

We may later modularize the prompt into protocol markdown files, but that is not
the current top priority.

### 6.2 Planning

Planning should remain mostly free.

Rules:

- do not inject strong skill relation priors before planning,
- do not let skill pre-commit the model to one semantic direction before the
  model has planned,
- planning should primarily use graph evidence, discovered domains, verified
  entities, and the current core protocol.

### 6.3 Post-plan skill filtering

After a plan is produced:

- use `plan.related` and `plan.maybe_related` as the first weak filter,
- remove clearly irrelevant skills,
- but do **not** prematurely reduce everything to a single intent direction.

Important:

- relation overlap is enough for weak filtering,
- do not add a separate answer-type filter here,
- because relation/path family already carries most of that answer-type signal.

### 6.4 Post-plan skill aggregation

This is the major next-step feature.

The system should take the filtered skills and ask the LLM to aggregate them
into **one markdown artifact**.

That markdown should:

- not choose one final interpretation for the question,
- not collapse everything into one summary paragraph,
- instead organize experience into a few **natural-language semantic directions**.

These are not rigid taxonomy labels.
They should read like:

- “When the question is asking for the broader containing region of a place...”
- “When the question is asking for the current or latest holder of a role...”
- “When the question is asking for the language actually used by a people or a country...”
- “When the question is asking for a character or role inside a named work...”

The point is:

- the LLM aggregator should surface several plausible semantic directions,
- the main reasoning model should later decide which one fits the current graph
  evidence.

### 6.5 Stage-separated skill injection

The aggregated markdown should **not** be dumped into the model all at once.

Instead:

- action-related content is injected only at action selection stage,
- final-selection content is injected only at final reasoning stage.

That is now the intended behavior.

#### Action-stage injection

This should tell the model things like:

- which relations are usually the meaningful ones to pay attention to,
- that a relevant relation token alone is not enough,
- that the model should consider the logic of the whole relation chain,
- and select the path whose overall logic best matches the question.

Important nuance:

- this should **not** become a hard “path semantics rule”,
- it should be phrased as a selection preference,
- for example:
  - focus on the relevant core relations,
  - but judge them within the logic of the whole relation chain,
  - and choose the most suitable path.

#### Final-stage injection

This should tell the model:

1. what kind of answer the question is really asking for,
2. whether this kind of answer is usually single or multiple,
3. whether in the absence of explicit temporal scope the answer usually leans
   current/latest,
4. and what common answer-family mismatches to avoid.

This is more useful than generic “do not force collapse” reminders.

### 6.6 Search agent

Search should not be a default generic reasoning tool.

The desired future design is:

- the main model reaches final stage with a selected action-space and candidates,
- if graph evidence is still insufficient for disambiguation,
- the model explicitly requests external verification,
- the scaffold invokes a dedicated search agent,
- the search agent receives:
  - the original question,
  - the current selected action-space summary,
  - the current graph candidates,
  - what needs to be verified,
- the search agent performs search and returns structured evidence,
- the final answer must still come from the graph candidates.

This is better than the current implicit “information insufficient” heuristic.

---

## 7. Proposed Shape of the Aggregated Skill Markdown

The aggregated artifact should be one markdown file, generated after weak
post-plan filtering, and organized by natural-language semantic directions.

It should not use abstract labels like `Direction A`.

It should not overfit to rigid taxonomy names.

It should not pick one direction for the model.

Instead, each section should look roughly like this:

### When the question is really asking for ...

Then include two subsections:

#### Action Experience

This section should:

- mention which relation families tend to matter most,
- remind the model to consider the logic of the whole relation chain,
- avoid hard-coded mandatory rules,
- help the model choose the most suitable path among current action options.

#### Final Selection Experience

This section should:

- say what the question is truly asking for,
- say whether this answer family usually converges to one answer or often keeps
  multiple,
- say whether it usually leans current/latest when temporal scope is implicit,
- warn against typical misreadings.

Example framing:

- “This question is asking for the office holder role entity rather than a
  biography-related person.”
- “This kind of question usually converges to a single answer.”
- “If explicit temporal scope is absent, it often leans toward the latest valid
  holder.”

This is much closer to the intended direction than rigid labels or hard-coded
rules.

---

## 8. Why This Design Is Better Than The Current One

### 8.1 It avoids pre-plan oversteering

The model gets to form its own plan first.

This matches the empirical observation that:

- most WebQSP cases already have a mostly-correct plan shape,
- the largest gains come later, when choosing the right action-space or the
  right subset of answers.

### 8.2 It avoids premature semantic collapse

If multiple semantic directions remain plausible:

- we do not want the aggregator to commit to exactly one,
- we want it to surface a few plausible direction chapters,
- and let the main model decide using current graph evidence.

### 8.3 It keeps skill useful without turning it into a hard controller

Skill should not become:

- a hidden planner,
- a rigid controller,
- or a global state machine.

Skill should become:

- structured reusable guidance,
- introduced after planning,
- staged according to the current decision point.

---

## 9. Concrete Implementation Scope For The Next Iteration

This is the recommended scope for the next implementation wave.

### In scope

1. Keep the main prompt broadly unchanged.
2. Refactor skill usage to be post-plan.
3. Add weak post-plan skill filtering based on plan relations.
4. Build an LLM skill aggregator that outputs one markdown artifact with
   multi-direction sections.
5. Separate action-stage injection from final-stage injection.
6. Continue harness stabilization:
   - preserve stable intermediate state,
   - reduce full-turn regeneration,
   - keep patch-style repair behavior.

### Out of scope for this wave

1. A full main-prompt rewrite.
2. Broad new prompt-variant sweeps.
3. Restoring global always-on consistency.
4. Reintroducing strong pre-plan skill priors.
5. A full production-grade web search integration.

---

## 10. Suggested Module Responsibilities

### Main runtime / scaffold

Responsible for:

- staging,
- preserving stable intermediate state,
- invoking skills at the right time,
- invoking search agent when requested,
- keeping final answer space tied to graph candidates.

### Skill retrieval

Responsible for:

- retrieving a shortlist,
- weakly filtering by current plan relation family,
- not making final semantic commitments.

### Skill aggregation LLM

Responsible for:

- turning filtered skills into one markdown artifact,
- organizing experience into natural-language semantic directions,
- not inventing hard rules,
- not deciding the single correct direction in advance.

### Main inference LLM

Responsible for:

- producing plan,
- selecting action based on current graph options and action-stage skill
  guidance,
- selecting final answer based on current graph evidence and final-stage skill
  guidance,
- requesting external verification when graph evidence is insufficient.

### Search agent

Responsible for:

- external verification only,
- comparing current graph candidates,
- returning structured evidence,
- not introducing new non-graph answer strings as final answers.

---

## 11. Known Risks And How To Avoid Them

### Risk 1: skill aggregator becomes another oversteering layer

Mitigation:

- do not let the aggregator choose a single direction,
- require multi-direction output,
- keep the language advisory rather than mandatory.

### Risk 2: direction chapters become too generic

Mitigation:

- require the final-stage section to say:
  - what the question is really asking for,
  - whether answers are usually single or multiple,
  - whether there is a current/latest tendency,
  - what common misreadings happen.

### Risk 3: action-stage advice becomes too rigid about path semantics

Mitigation:

- phrase it as:
  - focus on relevant core relations,
  - but consider the logic of the whole chain,
  - choose the path whose overall logic best matches the question.

Do not phrase it as a hard rejection rule.

### Risk 4: search becomes a noisy default crutch

Mitigation:

- search is only available after a selected action-space and current graph
  candidates already exist,
- the main model must explicitly request it,
- final answers must still be chosen from graph candidates.

---

## 12. Team Task Breakdown

### Team A — Harness / Scaffold

Focus:

- stable state preservation,
- patch-style repair,
- better run manifests and failure accounting,
- cleaner branch aggregation behavior for WebQSP.

Key deliverables:

- runtime state preservation improvements,
- explicit result accounting for failed cases,
- better repair prompts built from stable state,
- safer parser / execution handling.

### Team B — Skill Refactor

Focus:

- redesign skill generation usage,
- remove pre-plan overreach,
- add post-plan aggregation,
- stage-separated injection.

Key deliverables:

- post-plan weak filter,
- skill aggregation prompt,
- aggregated markdown schema,
- action-stage extraction,
- final-stage extraction.

### Team C — Search Agent

Focus:

- define the main-model-to-search-agent contract,
- add a minimal search-request path,
- return structured evidence,
- keep search as final-stage fallback only.

Key deliverables:

- search request schema,
- search result schema,
- scaffold invocation path,
- final-stage evidence handoff.

---

## 13. Recommended Validation Order

The next iterations should be validated in this order:

1. Preserve current best broad baseline as reference.
2. Introduce post-plan weak skill filtering only.
3. Add multi-direction aggregated skill markdown.
4. Split action-stage and final-stage skill injection.
5. Measure whether this beats the current best broad baseline.
6. Only then add a minimal search-agent fallback on selected difficult subsets.

Reason:

- this isolates the skill redesign,
- avoids mixing search effects too early,
- and keeps the comparison against the best-known branch fair.

---

## 14. Practical Summary For Team Mode

If the team needs the shortest possible instruction:

- Treat the main prompt as frozen for now.
- The next main redesign is **post-plan skill aggregation**, not prompt tuning.
- Skills should no longer heavily steer planning before plan exists.
- After plan, weakly filter skills by relation family.
- Aggregate the remaining skills into **one markdown artifact** with several
  natural-language semantic directions.
- Inject only action-stage guidance at action time.
- Inject only final-selection guidance at final-reasoning time.
- Keep search as a later final-stage fallback, not a default tool.
- Continue improving harness stability so valid intermediate structure is not
  lost during repair.

That is the current best synthesis of the project state.

