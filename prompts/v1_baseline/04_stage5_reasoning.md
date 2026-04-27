# Stage 5: Final Reasoning
# Source: plug_v12_feedback.py _stage5_reasoning() (full v5_filter_then_answer variant)

```
[STAGE 5: FINAL REASONING]

Candidates collected. Now perform final verification and output the answer.

CURRENT GRAPH CANDIDATES:
- {candidate_list}
If graph-visible attributes cannot justify a narrower answer subset, use web_search(...) before guessing.

FINAL ORDERING CONTRACT:
1. choose the primary action space
2. apply filter analysis if available
3. decide keep-all vs subset
4. output the answer
Do not skip directly from candidates to answer.

━━━ CORE PROTOCOL ━━━
Treat final reasoning as PRIMARY-ACTION-SPACE selection plus evidence-based filtering.
The knowledge graph remains the source of truth for final answer strings.

1. PRIMARY ACTION SPACE FIRST                                    ← REDUNDANT with Stage 3→4
   - Stay inside ONE primary action space whenever possible.     ← REDUNDANT
   - Prefer the action space whose WHOLE path semantics best matches the question.  ← REDUNDANT
   - Do NOT merge multiple action spaces by default.             ← REDUNDANT

2. FILTER INSIDE THE PRIMARY ACTION SPACE                        ← REDUNDANT with Stage 3→4
   - Look at graph-visible evidence inside the chosen action space first.
   - If all surviving candidates are non-conflicting and still answer the question, keep them.
   - If the question implies a narrower subset, filter to the supported subset.
   - Do NOT force a collapse just because the question sounds singular.  ← REDUNDANT
   - If `filter()` was executed, analyze the displayed `[Per-Candidate Matches]` values candidate by candidate.
   - A candidate "passing" filter only means the relation/value is present; it does NOT automatically mean the candidate fits the question.
   - Compare the shown filter values against the question target / named entity / time / role, and keep only the subset whose values actually match.

3. KNOWLEDGE VS GRAPH                                            ← HOME (keep here)
   - You may use parametric knowledge only as a weak aid for interpretation.
   - If graph evidence conflicts with parametric knowledge, graph evidence wins.
   - If graph-visible evidence does not distinguish candidates, do not invent a discriminator.

4. SPELLING & FORMAT CHECK (STRICT)                              ← HOME (keep here)
   - Every final answer must be an EXACT graph string from current candidates or current node details.
   - Use FULL entity names only. No truncation, paraphrase, or normalization.
   - Separate multiple answers cleanly.
   - Order multiple answers by current graph-supported credibility, with the strongest answer first.

━━━ REFLECTION CHECKPOINT (CRITICAL) ━━━
Before outputting:
  □ Did I stay inside one primary action space unless a true exception is necessary?  ← REDUNDANT
  □ Did I decide keep-all vs filter-subset using CURRENT graph-visible evidence?       ← REDUNDANT
  □ If graph evidence cannot distinguish candidates, did I avoid unsupported single-answer collapse?
  □ Is EACH answer an EXACT string from tool output? (Case-sensitive!)
  □ For multiple answers, is each in a SEPARATE \boxed{}?

━━━ OUTPUT FORMAT (STRICT) ━━━
- Single answer: <answer>\boxed{Exact Graph String}</answer>      ← BUG: model copies literal
- Multiple answers: <answer>\boxed{Entity1} \boxed{Entity2}</answer>
- Each answer MUST be exact string from graph
- NO text after </answer> tag

<reasoning>
  [STEP 1: PRIMARY ACTION SPACE]                                  ← REDUNDANT with Stage 3→4
  - ...

  [STEP 2: FILTER ANALYSIS]                                       ← REDUNDANT with Stage 3→4
  - ...

  [STEP 3: KEEP-ALL OR SUBSET]                                    ← REDUNDANT with Stage 3→4
  - ...

  [SPELLING VERIFICATION]
  - Final string: ...
</reasoning>
<answer>\boxed{Exact Graph String}</answer>                       ← BUG: model copies literal
```

## Injected Skill Hint: build_reasoning_stage_hint()
# Source: retriever.py:881-950
# Injected at Stage 4 (turns 4-5) via runtime.py:331-335

```
[RETRIEVED SKILL EXPERIENCE: FINAL SELECTION]
Below are final-selection experiences aggregated from similar solved questions.
Use them as soft priors only. Final answers must still follow CURRENT candidates and CURRENT graph evidence.

- Primary action-space policy:                                    ← REDUNDANT with Stage 3→4 + Stage 5
  - First stay inside one primary action space whose whole semantics best matches the question.  ← REDUNDANT
  - Inside that action space, decide whether to keep all supported answers or filter to the supported subset.  ← REDUNDANT
  - Do not merge different action spaces unless one action clearly cannot cover the required answer set and the extra action adds the same semantic family.  ← REDUNDANT

- Final-selection experience from similar questions:              ← KEEP (concrete data)
  - {actual experience lines from cards}

- Fixed output protocol:                                          ← REDUNDANT with Stage 5
  - Use exact graph strings only.                                 ← REDUNDANT
  - Use full entity names only.                                   ← REDUNDANT
  - Separate multiple answers cleanly.                            ← REDUNDANT
  - Rank the most credible graph-supported answer first.           ← REDUNDANT
  - Skill experience does NOT override current graph evidence.     ← REDUNDANT
```
