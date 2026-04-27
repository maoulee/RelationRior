# Stage 5: Final Reasoning (V2.1 — with backtrack + filter clarification)
# Changes from V2:
#   - Clarify filter() is an attribute display tool, NOT a filter-out tool
#   - Allow backtrack to Stage 3→4 with state checkpoint
#   - Candidates = single action space results (no cross-workspace merging)
#   - Remove over-conservative "do not reopen" language

```
[STAGE 5: FINAL REASONING]

Candidates collected from your selected primary action space. Now perform final verification and output the answer.

CURRENT GRAPH CANDIDATES:
- {candidate_list}

━━━ CORE PROTOCOL ━━━

1. CANDIDATE EVALUATION
   - Your candidates come from the single action space selected at Stage 3→4.
   - If `filter()` was executed, analyze `[Per-Candidate Matches]` as attribute information only.
   - `filter()` returning 0 matches means the requested attribute is not available — it does NOT mean candidates should be eliminated.
   - Keep all candidates that plausibly answer the question. If you are unsure, keep them.

2. KNOWLEDGE VS GRAPH
   - You may use parametric knowledge only as a weak aid for interpretation.
   - If graph evidence conflicts with parametric knowledge, graph evidence wins.
   - If graph-visible evidence does not distinguish candidates, do not invent a discriminator.

3. BACKTRACK (ONLY IF NEEDED)
   - If ALL current candidates are clearly irrelevant to the question, you may backtrack to Stage 2→3 to select a different action space.
   - When backtracking: state the reason (why current candidates fail), and which alternative action you want to try.
   - Do NOT backtrack just because candidates are imperfect — only when they are clearly wrong.

4. SPELLING & FORMAT CHECK (STRICT)
   - Every final answer must be an EXACT graph string from current candidates or current node details.
   - Use FULL entity names only. No truncation, paraphrase, or normalization.
   - Separate multiple answers cleanly.
   - Order multiple answers by current graph-supported credibility, strongest first.

━━━ REFLECTION CHECKPOINT ━━━
Before outputting:
  □ Did I evaluate candidates based on available evidence (not absence of filter results)?
  □ If evidence cannot distinguish candidates, did I keep them all?
  □ Is EACH answer an EXACT string from tool output? (Case-sensitive!)
  □ For multiple answers, is each in a SEPARATE \boxed{}?

━━━ OUTPUT FORMAT (STRICT) ━━━
- Single answer: <answer>\boxed{ETH Zurich}</answer>
- Multiple answers: <answer>\boxed{ETH Zurich} \boxed{University of Zurich}</answer>
- Each answer MUST be an exact string from the graph candidates or node details.
- NO text after </answer> tag.

<reasoning>
  [CANDIDATE EVALUATION]
  - Candidate 1: [Keep/Eliminate] because ...
  - Candidate 2: [Keep/Eliminate] because ...

  [BACKTRACK CHECK]
  - Need backtrack? [Yes/No] — If yes: reason and target action ...

  [SPELLING VERIFICATION]
  - Final string: "exact string from tool output"
</reasoning>
<answer>\boxed{exact graph string here}</answer>
```
