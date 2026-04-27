# Stage 5: Final Reasoning (V2 Consolidated)
# Changes from V1:
#   - Removed Core Protocol sections 1-2 (action space selection + filter analysis)
#     → already decided at Stage 3→4
#   - Added short handoff line referencing previous selection
#   - Simplified Reflection: removed action space / filter items
#   - Fixed \boxed{} template: removed duplicate placeholder, use instantiated example
#   - Kept: Knowledge vs Graph, Spelling & Format, anti-collapse

```
[STAGE 5: FINAL REASONING]

Candidates collected. Now perform final verification and output the answer.

CURRENT GRAPH CANDIDATES:
- {candidate_list}
If graph-visible attributes cannot justify a narrower answer subset, use web_search(...) before guessing.

FINAL ORDERING CONTRACT:
1. Use the primary action space selected at Stage 3→4; do not reopen action-space selection.
2. Apply filter analysis if filter() was executed earlier.
3. Decide keep-all vs subset based on graph-visible evidence.
4. Output the answer.
Do not skip directly from candidates to answer.

━━━ CORE PROTOCOL ━━━
The knowledge graph remains the source of truth for final answer strings.

1. EVIDENCE EVALUATION
   - Use the primary action space already selected at Stage 3→4.
   - If `filter()` was executed, analyze `[Per-Candidate Matches]` values candidate by candidate.
   - A candidate "passing" filter only means the relation/value is present; it does NOT automatically mean the candidate fits the question.
   - Compare filter values against the question target / named entity / time / role, and keep only matching candidates.

2. KNOWLEDGE VS GRAPH
   - You may use parametric knowledge only as a weak aid for interpretation.
   - If graph evidence conflicts with parametric knowledge, graph evidence wins.
   - If graph-visible evidence does not distinguish candidates, do not invent a discriminator.
   - Do NOT force a single-answer collapse when evidence supports multiple valid answers.

3. SPELLING & FORMAT CHECK (STRICT)
   - Every final answer must be an EXACT graph string from current candidates or current node details.
   - Use FULL entity names only. No truncation, paraphrase, or normalization.
   - Separate multiple answers cleanly.
   - Order multiple answers by current graph-supported credibility, with the strongest answer first.

━━━ REFLECTION CHECKPOINT ━━━
Before outputting:
  □ Did I use graph-visible evidence to justify each kept/eliminated candidate?
  □ If evidence cannot distinguish candidates, did I avoid unsupported single-answer collapse?
  □ Is EACH answer an EXACT string from tool output? (Case-sensitive!)
  □ For multiple answers, is each in a SEPARATE \boxed{}?

━━━ OUTPUT FORMAT (STRICT) ━━━
- Single answer: <answer>\boxed{ETH Zurich}</answer>
- Multiple answers: <answer>\boxed{ETH Zurich} \boxed{University of Zurich}</answer>
- Each answer MUST be an exact string from the graph candidates or node details.
- NO text after </answer> tag.

<reasoning>
  [EVIDENCE EVALUATION]
  - Primary action space: {from Stage 3→4}
  - Candidate 1: [Keep/Eliminate] because ...
  - Candidate 2: [Keep/Eliminate] because ...

  [SPELLING VERIFICATION]
  - Final string: "exact string from tool output"
</reasoning>
<answer>\boxed{exact graph string here}</answer>
```
