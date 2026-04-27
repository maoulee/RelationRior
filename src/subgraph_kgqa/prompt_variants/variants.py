"""
Stage-5 prompt variants for action-space style prompt tuning.

These variants only target the final reasoning / answer selection phase.
They are intentionally small so experiments can switch prompt behavior
without rewriting the rest of the inference pipeline.
"""

from __future__ import annotations

from typing import Dict


DEFAULT_NAME = "v0_baseline"


def _variant(
    *,
    description: str,
    core_protocol_override: str = "",
    reflection_override: str = "",
    output_format_override: str = "",
    reasoning_template_override: str = "",
    extra_reminder: str = "",
) -> Dict[str, str]:
    return {
        "description": description,
        "core_protocol_override": core_protocol_override,
        "reflection_override": reflection_override,
        "output_format_override": output_format_override,
        "reasoning_template_override": reasoning_template_override,
        "extra_reminder": extra_reminder,
    }


VARIANTS: Dict[str, Dict[str, str]] = {
    "v0_baseline": _variant(
        description="Current actionspace_v1 + per_skill + fix3 baseline.",
    ),
    "v1_filter_value_strict": _variant(
        description="Treat filter as value-level evidence; do not interpret all-pass as keep-all.",
        core_protocol_override="""
━━━ CORE PROTOCOL ━━━
Treat final reasoning as PRIMARY-ACTION-SPACE selection plus evidence-based filtering.
The knowledge graph remains the source of truth for final answer strings.

1. PRIMARY ACTION SPACE FIRST
   - Stay inside ONE primary action space whenever possible.
   - Prefer the action space whose WHOLE path semantics best matches the question.
   - Do NOT merge multiple action spaces by default.

2. FILTER INSIDE THE PRIMARY ACTION SPACE
   - Look at graph-visible evidence inside the chosen action space first.
   - If the question implies a narrower subset, prefer an explicit filter step before free-form collapse.
   - If `filter()` was executed, treat `[Per-Candidate Matches]` as the main evidence for subset selection.
   - A candidate "passing" filter only means the relation/value exists; it does NOT mean the candidate matches the question target.
   - Compare each shown value against the question target / named entity / time / role.
   - If only some candidates' values match the question target, keep only that subset.
   - Keep all candidates only when their shown values are all compatible with the same question target.

3. KNOWLEDGE VS GRAPH
   - Use parametric knowledge only as weak interpretation support.
   - If graph evidence conflicts with parametric knowledge, graph evidence wins.
   - If graph-visible evidence does not distinguish candidates, do not invent a discriminator.

4. SPELLING & FORMAT CHECK (STRICT)
   - Every final answer must be an EXACT graph string from current candidates or current node details.
   - Use FULL entity names only. No truncation, paraphrase, or normalization.
   - Separate multiple answers cleanly.
   - Order multiple answers by current graph-supported credibility, with the strongest answer first.
""".strip(),
        reflection_override="""
━━━ REFLECTION CHECKPOINT (CRITICAL) ━━━
Before outputting:
  □ Did I stay inside one primary action space unless a true exception is necessary?
  □ If filter values were shown, did I compare the VALUES themselves against the question target?
  □ Did I avoid treating "all passed" as "all keep"?
  □ Is EACH answer an EXACT string from tool output? (Case-sensitive!)
  □ For multiple answers, is each in a SEPARATE \\boxed{}?
""".strip(),
        reasoning_template_override="""
<reasoning>
  [PRIMARY ACTION SPACE]
  - Why this action space best matches the question: ...

  [FILTER VALUE ANALYSIS]
  - Which filter values actually match the question target: ...
  - Candidate 1: [Keep/Eliminate] because value ... [matches / does not match]
  - Candidate 2: [Keep/Eliminate] because value ... [matches / does not match]

  [SPELLING VERIFICATION]
  - Final string: "exact string from tool output"
</reasoning>
<answer>\\boxed{Exact Graph String}</answer>
""".strip(),
    ),
    "v2_no_unsupported_collapse": _variant(
        description="Strongly penalize unsupported single-answer collapse.",
        core_protocol_override="""
━━━ CORE PROTOCOL ━━━
Treat final reasoning as PRIMARY-ACTION-SPACE selection plus evidence-based filtering.
The knowledge graph remains the source of truth for final answer strings.

1. PRIMARY ACTION SPACE FIRST
   - Stay inside ONE primary action space whenever possible.
   - Prefer the action space whose WHOLE path semantics best matches the question.
   - Do NOT merge multiple action spaces by default.

2. FILTER INSIDE THE PRIMARY ACTION SPACE
   - Look at graph-visible evidence inside the chosen action space first.
   - If all surviving candidates are non-conflicting and still answer the question, keep them.
   - Do NOT collapse to a single answer unless current graph-visible evidence actually distinguishes one candidate from the others.
   - The question sounding singular is NOT enough by itself.
   - If `filter()` was executed, use its displayed values as the main justification for any collapse.
   - If evidence does not distinguish candidates, preserve the supported subset instead of guessing one "best" answer.

3. KNOWLEDGE VS GRAPH
   - Use parametric knowledge only as weak interpretation support.
   - If graph evidence conflicts with parametric knowledge, graph evidence wins.
   - If graph-visible evidence does not distinguish candidates, do not invent a discriminator.

4. SPELLING & FORMAT CHECK (STRICT)
   - Every final answer must be an EXACT graph string from current candidates or current node details.
   - Use FULL entity names only. No truncation, paraphrase, or normalization.
   - Separate multiple answers cleanly.
   - Order multiple answers by current graph-supported credibility, with the strongest answer first.
""".strip(),
        reflection_override="""
━━━ REFLECTION CHECKPOINT (CRITICAL) ━━━
Before outputting:
  □ Did I stay inside one primary action space unless a true exception is necessary?
  □ If I reduced multiple candidates to one, what explicit graph-visible evidence justified that collapse?
  □ If graph evidence cannot distinguish candidates, did I preserve the supported subset?
  □ Is EACH answer an EXACT string from tool output? (Case-sensitive!)
  □ For multiple answers, is each in a SEPARATE \\boxed{}?
""".strip(),
        reasoning_template_override="""
<reasoning>
  [PRIMARY ACTION SPACE]
  - Why this action space best matches the question: ...

  [COLLAPSE CHECK]
  - Can graph-visible evidence distinguish candidates? [Yes/No]
  - If yes: what evidence distinguishes them?
  - If no: keep the supported subset; do not force single-answer collapse.

  [SPELLING VERIFICATION]
  - Final string: "exact string from tool output"
</reasoning>
<answer>\\boxed{Exact Graph String}</answer>
""".strip(),
    ),
    "v3_primary_action_semantics": _variant(
        description="Bias action-space choice toward whole-path semantics over single relation tokens.",
        core_protocol_override="""
━━━ CORE PROTOCOL ━━━
Treat final reasoning as PRIMARY-ACTION-SPACE selection plus evidence-based filtering.
The knowledge graph remains the source of truth for final answer strings.

1. PRIMARY ACTION SPACE FIRST
   - First choose the ONE action space whose WHOLE path semantics best answers the question.
   - Judge the start entity, intermediate meaning, and returned answer type together.
   - Do NOT prefer an action just because one relation token looks familiar in isolation.
   - Use another action space only if the primary one clearly fails to answer the same semantic intent.

2. FILTER INSIDE THE PRIMARY ACTION SPACE
   - After choosing the primary action space, stay inside it whenever possible.
   - If all surviving candidates are non-conflicting and still answer the question, keep them.
   - If the question implies a narrower subset, filter to the supported subset.
   - If `filter()` was executed, use the displayed values to justify subset selection.

3. KNOWLEDGE VS GRAPH
   - Use parametric knowledge only as weak interpretation support.
   - If graph evidence conflicts with parametric knowledge, graph evidence wins.
   - If graph-visible evidence does not distinguish candidates, do not invent a discriminator.

4. SPELLING & FORMAT CHECK (STRICT)
   - Every final answer must be an EXACT graph string from current candidates or current node details.
   - Use FULL entity names only. No truncation, paraphrase, or normalization.
   - Separate multiple answers cleanly.
   - Order multiple answers by current graph-supported credibility, with the strongest answer first.
""".strip(),
        reflection_override="""
━━━ REFLECTION CHECKPOINT (CRITICAL) ━━━
Before outputting:
  □ Did I choose the primary action by WHOLE path semantics, not one relation token?
  □ Did I avoid merging a second action space unless it was truly necessary?
  □ Did I decide keep-all vs filter-subset using CURRENT graph-visible evidence?
  □ Is EACH answer an EXACT string from tool output? (Case-sensitive!)
  □ For multiple answers, is each in a SEPARATE \\boxed{}?
""".strip(),
        reasoning_template_override="""
<reasoning>
  [PRIMARY ACTION SPACE]
  - Whole-path semantics: start -> path meaning -> answer type
  - Why this action space best matches the question: ...

  [FILTER INSIDE ACTION SPACE]
  - Keep-all or filter-subset: ...
  - Candidate 1: [Keep/Eliminate] because ...
  - Candidate 2: [Keep/Eliminate] because ...

  [SPELLING VERIFICATION]
  - Final string: "exact string from tool output"
</reasoning>
<answer>\\boxed{Exact Graph String}</answer>
""".strip(),
    ),
    "v4_concise_final_reasoning": _variant(
        description="Reduce free-form drift with a shorter final-reasoning structure.",
        extra_reminder=(
            "FINAL-REASONING STYLE:\n"
            "- Keep the final reasoning extremely short.\n"
            "- No long explanations, no background knowledge dump, no free-form debate.\n"
            "- Make one primary-action decision, one keep/filter decision, then answer."
        ),
        reflection_override="""
━━━ REFLECTION CHECKPOINT (CRITICAL) ━━━
Before outputting:
  □ One primary action?
  □ One keep/filter decision grounded in current evidence?
  □ No unsupported collapse?
  □ Exact graph strings only?
""".strip(),
        reasoning_template_override="""
<reasoning>
  [PRIMARY ACTION] ...
  [KEEP/FILTER] ...
  [SPELLING VERIFICATION]
  - Final string: ...
</reasoning>
<answer>\\boxed{Exact Graph String}</answer>
""".strip(),
    ),
    "v5_filter_then_answer": _variant(
        description="Enforce a strict final sequence: action -> optional filter -> reasoning with skill reference -> answer.",
        extra_reminder=(
            "FINAL ORDERING CONTRACT:\n"
            "1. Choose the primary action space\n"
            "2. Disambiguation (when filter gate triggers):\n"
            "   - You MUST execute filter() with your chosen relation(s)\n"
            "   - You decide which relations are most relevant for disambiguation\n"
            "   - filter() retrieves attribute values; analysis happens in reasoning (step 3)\n"
            "3. Reference historical case experience (advisory, not mandatory) — combine with current context\n"
            "4. Output the answer"
        ),
        reasoning_template_override="""
<reasoning>
  [STEP 1: PRIMARY ACTION SPACE]
  - Which action space was selected and why.

  [STEP 2: REFERENCE EXPERIENCE]
  - Historical case experience for reference (advisory, not mandatory).
  - Combine this reference with the current question's specifics to form your analysis.

  [STEP 3: CANDIDATE EVALUATION]
  - If filter() was executed: analyze per-candidate attribute values.
  - If filter() was not triggered: evaluate candidates directly.
  - Based on current information and reference experience, decide: keep all or select subset.

  [SPELLING VERIFICATION]
  - Final string: ...
</reasoning>
<answer>\\boxed{Exact Graph String}</answer>
""".strip(),
    ),
}
