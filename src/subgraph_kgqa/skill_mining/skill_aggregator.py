"""Skill aggregation mini-agent for KGQA question-answering.

Before reasoning starts, the LLM reviews the question + retrieved skill cards
and synthesizes them into a semi-structured markdown narrative grouped by
intent directions.  This replaces rigid JSON fields with emergent,
experience-based guidance.

Feature-gated by KGQA_ENABLE_SKILL_AGGREGATION env var (default: off).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature gate
# ---------------------------------------------------------------------------

def is_skill_aggregation_enabled() -> bool:
    """Return True when KGQA_ENABLE_SKILL_AGGREGATION is set to a truthy value."""
    return os.getenv("KGQA_ENABLE_SKILL_AGGREGATION", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class AggregatedSkill:
    """Synthesized skill guidance produced by the aggregation LLM call.

    The core output is `markdown_narrative` — a semi-structured markdown
    document grouped by intent directions, containing preferred relations,
    error experience narratives, and evidence-driven answer strategies.

    Legacy structured fields are retained for backward compatibility and
    downstream consumers that may still reference them.
    """

    # --- Core output (new design) ---
    markdown_narrative: str = ""          # Semi-structured markdown grouped by intent

    # --- Legacy structured fields (kept for compat, populated from markdown) ---
    question_analysis: str = ""
    answer_type_guidance: str = ""
    temporal_guidance: str = ""
    scope_guidance: str = ""
    pitfalls: List[str] = field(default_factory=list)
    conflict_notes: str = ""
    combined_reasoning_hint: str = ""
    negative_experiences: List[str] = field(default_factory=list)
    plan_guidance: str = ""
    action_guidance: str = ""
    reasoning_guidance: str = ""
    error_type_distribution: Dict[str, int] = field(default_factory=dict)

    # --- Meta ---
    raw_llm_response: str = ""


def _empty_aggregated_skill(raw: str = "") -> AggregatedSkill:
    """Return an empty AggregatedSkill (graceful-degradation sentinel)."""
    return AggregatedSkill(raw_llm_response=raw, error_type_distribution={})


# ---------------------------------------------------------------------------
# Helpers for normalising skill-card field values
# ---------------------------------------------------------------------------

def _normalize_field(value: Any) -> str:
    """Coerce a skill-card field value to a plain string for the prompt."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "; ".join(str(v).strip() for v in value if str(v).strip())
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value).strip()


def _parse_pitfalls(value: Any) -> List[str]:
    """Parse the *pitfalls* field from the LLM JSON response."""
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [
            line.strip("-* ").strip()
            for line in value.splitlines()
            if line.strip("-* ").strip()
        ]
    return []


# ---------------------------------------------------------------------------
# Prompt construction — NEW: asks for semi-structured markdown output
# ---------------------------------------------------------------------------

_SKILL_FIELDS = [
    "case_id",
    "question",
    "core_relations",
    "action_space_experience",
    "final_selection_experience",
    "constraint_guidance",
    "answer_strategy",
    "common_pitfalls",
    "intent_clarification",
    "common_misreadings",
    "wrong_but_related_answer_families",
]


def _build_aggregation_prompt(
    question_text: str,
    skills: List[Any],
    negative_skills: Optional[List[Any]] = None,
) -> str:
    """Build the LLM prompt that asks for semi-structured markdown synthesis."""
    skill_blocks: list[str] = []
    for idx, skill in enumerate(skills, start=1):
        lines = [f"Skill Card {idx}:"]
        for fld in _SKILL_FIELDS:
            val = _normalize_field(getattr(skill, fld, ""))
            if val:
                lines.append(f"- {fld}: {val}")
        skill_blocks.append("\n".join(lines))

    parts = [
        "You are synthesizing multiple KGQA skill cards into a semi-structured experience document.",
        f"Target question:\n{question_text.strip()}",
        "Retrieved skill cards (successful cases):\n" + "\n\n".join(skill_blocks),
    ]

    # Negative experience section
    if negative_skills:
        neg_blocks: list[str] = []
        error_counts: Dict[str, int] = {}
        for idx, neg in enumerate(negative_skills, start=1):
            neg_lines = [f"Failed Case {idx}:"]
            neg_lines.append(f"- question: {_normalize_field(getattr(neg, 'question', ''))}")
            etype = getattr(neg, 'error_type', '') or 'unknown'
            neg_lines.append(f"- error_type: {etype}")
            error_counts[etype] = error_counts.get(etype, 0) + 1
            wrong_rels = getattr(neg, 'wrong_plan_relations', []) or []
            if wrong_rels:
                neg_lines.append(f"- wrong_relations: {'; '.join(wrong_rels[:5])}")
            correct_rels = getattr(neg, 'correct_plan_relations', []) or []
            if correct_rels:
                neg_lines.append(f"- correct_relations: {'; '.join(correct_rels[:5])}")
            error_pattern = getattr(neg, 'error_pattern', '')
            if error_pattern:
                neg_lines.append(f"- what_went_wrong: {_normalize_field(error_pattern)}")
            correct_approach = getattr(neg, 'correct_approach', '')
            if correct_approach:
                neg_lines.append(f"- correct_approach: {_normalize_field(correct_approach)}")
            neg_blocks.append("\n".join(neg_lines))

        parts.append(
            "Failed similar cases (what went wrong):\n" + "\n\n".join(neg_blocks)
        )

        if error_counts:
            dist_lines = [f"  - {k}: {v}" for k, v in sorted(error_counts.items(), key=lambda x: -x[1])]
            parts.append(
                "Error type summary:\n" + "\n".join(dist_lines)
            )

    # --- Output format instruction ---
    parts.append(
        "## Task\n"
        "Analyze the skill cards and failed cases, then produce a natural-language "
        "experience document grouped by **intent directions** (derived from "
        "the retrieved skills, not predefined categories).\n\n"
        "Each direction chapter describes one recognizable question intent, written "
        "as advisory prose — not rigid rules or statistical counts.\n\n"
        "Keep two layers of information at the same time:\n"
        "1. A high-level direction summary.\n"
        "2. Case-grounded relation evidence showing which concrete questions in this direction "
        "were actually answered by which hard relations, and which nearby relations were tempting but wrong.\n\n"
        "## Output Format\n"
        "Return a JSON object with a single key `markdown_narrative` containing the markdown. "
        "Each direction chapter must follow this structure:\n\n"
        "```markdown\n"
        "## Direction: <short intent description>\n"
        "\n"
        "**Question pattern:** <what this kind of question is really asking for>\n"
        "**Action-space tendency:** <whether this kind of question usually keeps the selected action space, filters within it, or collapses within it>\n"
        "**Current/latest tendency:** <whether there is a temporal aspect, or none>\n"
        "**Common misreadings:**\n"
        "- <specific misreading 1>\n"
        "- <specific misreading 2>\n"
        "\n"
        "### Action guidance\n"
        "- Core relations to focus on: `relation1`, `relation2`\n"
        "- Relation-chain logic to prefer: <which whole relation-chain logic best matches the question>\n"
        "\n"
        "### Case-grounded relation evidence\n"
        "- Example question: <short question text>\n"
        "  - Hard relations that actually answered it: `relationA`, `relationB`\n"
        "  - Near but wrong relations: `relationC`\n"
        "- Example question: <short question text>\n"
        "  - Hard relations that actually answered it: `relationD`\n"
        "\n"
        "### Final reasoning guidance\n"
        "- <how to decide whether to keep, filter, or collapse within the selected action space>\n"
        "- <what evidence to check before narrowing or keeping candidates>\n"
        "- Wrong but related answer families to consider distinguishing from:\n"
        "  - <family description>\n"
        "```\n\n"
        "If multiple intent directions exist, create a chapter for each. "
        "If skills are homogeneous, a single direction is fine.\n\n"
        "Important:\n"
        "- Write in natural advisory language: describe what tends to happen, not what must happen\n"
        "- Do NOT reduce the advice to single vs multiple alone; describe whether the selected action space is usually kept, filtered within, or collapsed within\n"
        "- Use markdown formatting with backticks for relation names\n"
        "- Group by emergent intent, not by error type\n"
        "- Do NOT let the high-level direction summary erase question-specific hard relation differences\n"
        "- If two nearby relations lead to different answer slots in similar questions, preserve that distinction explicitly in case-grounded relation evidence\n"
        "- For each direction, list 2-4 concrete example questions when possible\n"
        "- 'Hard relations that actually answered it' should come from successful skill cards' core_relations, not from broad neighboring relations\n"
        "- 'Near but wrong relations' should only be listed when a plausible nearby relation family appears in the evidence and could mislead the current question\n"
        "- Successful skill cards are the primary evidence; failed similar cases are weak cautionary references only\n"
        "- Error experiences should be case-based reasoning, not commands\n"
        "- Return JSON only: `{\"markdown_narrative\": \"...\"}`"
    )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _extract_json_text(raw: str) -> str:
    """Pull the JSON object out of a possibly markdown-fenced response."""
    text = raw.strip()
    if not text:
        return text

    # Try to strip ```json ... ``` fences
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                return candidate

    # Fallback: outermost braces
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text


def _sanitize_markdown_narrative(markdown: str) -> str:
    text = _normalize_field(markdown)
    if not text:
        return text
    # Safety net: old / weak generations may still emit internal labels like "Merged".
    text = re.sub(
        r"^## Direction:\s*Merged\s*$",
        "## Direction: When similar solved questions point to the same answer direction",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^## Direction:\s*Unknown\s*$",
        "## Direction: When the retrieved skills suggest a similar answer direction",
        text,
        flags=re.MULTILINE,
    )
    return text


def _parse_aggregated_skill(raw: str) -> AggregatedSkill:
    """Parse the LLM response into an AggregatedSkill.

    Supports two formats:
    1. New format: {"markdown_narrative": "..."}
    2. Legacy format: {"question_analysis": "...", ...}
    3. Hybrid: both markdown_narrative and legacy fields present
    """
    payload = json.loads(_extract_json_text(raw))

    markdown_narrative = _sanitize_markdown_narrative(payload.get("markdown_narrative", ""))

    result = AggregatedSkill(
        markdown_narrative=markdown_narrative,
        question_analysis=_normalize_field(payload.get("question_analysis", "")),
        answer_type_guidance=_normalize_field(payload.get("answer_type_guidance", "")),
        temporal_guidance=_normalize_field(payload.get("temporal_guidance", "")),
        scope_guidance=_normalize_field(payload.get("scope_guidance", "")),
        pitfalls=_parse_pitfalls(payload.get("pitfalls", [])),
        conflict_notes=_normalize_field(payload.get("conflict_notes", "")),
        combined_reasoning_hint=_normalize_field(payload.get("combined_reasoning_hint", "")),
        negative_experiences=_parse_pitfalls(payload.get("negative_experiences", [])),
        plan_guidance=_normalize_field(payload.get("plan_guidance", "")),
        action_guidance=_normalize_field(payload.get("action_guidance", "")),
        reasoning_guidance=_normalize_field(payload.get("reasoning_guidance", "")),
        error_type_distribution=(
            {k: int(v) for k, v in payload.get("error_type_distribution", {}).items()}
            if isinstance(payload.get("error_type_distribution"), dict)
            else {}
        ),
        raw_llm_response=raw,
    )

    # Backward compatibility: sync reasoning_guidance and combined_reasoning_hint
    if not result.combined_reasoning_hint and result.reasoning_guidance:
        result.combined_reasoning_hint = result.reasoning_guidance
    if not result.reasoning_guidance and result.combined_reasoning_hint:
        result.reasoning_guidance = result.combined_reasoning_hint

    return result


# ---------------------------------------------------------------------------
# Default LLM caller  (mirrors intent_analyzer._default_call_llm exactly)
# ---------------------------------------------------------------------------

async def _default_call_llm(messages: List[Dict[str, str]]) -> str:
    """Default LLM call using the shared client."""
    from ..llm_client import call_llm
    return await call_llm(messages, max_tokens=2048, temperature=0.1, timeout_seconds=30.0, retries=2)


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

async def aggregate_skills(
    question_text: str,
    skills: List[Any],
    call_llm: Optional[Callable[[List[Dict[str, str]]], Awaitable[str]]] = None,
    negative_skills: Optional[List[Any]] = None,
) -> AggregatedSkill:
    """Synthesize retrieved skill cards into a semi-structured markdown narrative.

    Parameters
    ----------
    question_text : str
        The natural language question being answered.
    skills : list of CaseSkillCard
        Retrieved skill cards to synthesize.
    call_llm : async callable, optional
        LLM invocation function accepting ``messages`` list.
    negative_skills : list of NegativeSkillCard, optional
        Negative experience cards to include in synthesis.

    Returns
    -------
    AggregatedSkill
        Synthesized skill guidance with markdown_narrative as the core output.
    """
    if not is_skill_aggregation_enabled():
        return _empty_aggregated_skill()

    if not skills:
        return _empty_aggregated_skill()

    if call_llm is None:
        call_llm = _default_call_llm

    prompt = _build_aggregation_prompt(question_text, skills, negative_skills=negative_skills)
    messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]

    try:
        raw = await call_llm(messages)
    except Exception as exc:
        logger.warning("Skill aggregation LLM call failed: %s", exc)
        return _empty_aggregated_skill()

    try:
        return _parse_aggregated_skill(raw)
    except Exception as exc:
        logger.warning("Skill aggregation JSON parse failed: %s", exc)
        return _empty_aggregated_skill(raw)


# ---------------------------------------------------------------------------
# Prompt formatting helpers
# ---------------------------------------------------------------------------

def format_aggregated_skill_for_prompt(skill: AggregatedSkill) -> str:
    """Format an AggregatedSkill as a text block for injection.

    Prioritizes markdown_narrative when available, falls back to legacy format.
    Returns empty string when no useful content exists.
    """
    # New design: use markdown narrative directly
    if skill.markdown_narrative:
        return f"[RETRIEVED SKILL EXPERIENCE]\n{skill.markdown_narrative}"

    # Legacy fallback
    has_content = any(
        [
            skill.question_analysis,
            skill.answer_type_guidance,
            skill.temporal_guidance,
            skill.scope_guidance,
            skill.pitfalls,
            skill.conflict_notes,
            skill.combined_reasoning_hint,
            skill.negative_experiences,
        ]
    )
    if not has_content:
        return ""

    lines = [
        "Synthesized Skill Guidance:",
        f"- Question analysis: {skill.question_analysis or 'N/A'}",
        f"- Answer type guidance: {skill.answer_type_guidance or 'N/A'}",
        f"- Temporal guidance: {skill.temporal_guidance or 'N/A'}",
        f"- Scope guidance: {skill.scope_guidance or 'N/A'}",
        f"- Plan guidance: {skill.plan_guidance or 'N/A'}",
        f"- Action guidance: {skill.action_guidance or 'N/A'}",
    ]

    if skill.pitfalls:
        lines.append("- Pitfalls: " + "; ".join(skill.pitfalls))
    else:
        lines.append("- Pitfalls: N/A")

    if skill.negative_experiences:
        lines.append("- Negative experience (what to avoid): " + "; ".join(skill.negative_experiences))

    lines.extend(
        [
            f"- Conflict notes: {skill.conflict_notes or 'N/A'}",
            f"- Combined reasoning hint: {skill.combined_reasoning_hint or 'N/A'}",
        ]
    )

    return "\n".join(lines)


def format_aggregated_stage_hints(skill: AggregatedSkill) -> Dict[str, str]:
    """Generate per-stage hint strings from an AggregatedSkill.

    When markdown_narrative is present, extract stage-appropriate excerpts:
    - stage:1 empty (planning free of priors)
    - stage:2-3 gets action guidance excerpts
    - stage:4 gets selection guidance + answer tendency excerpts

    Falls back to legacy per-stage routing when no markdown_narrative.
    """
    # --- New design: stage-separated extraction ---
    if skill.markdown_narrative:
        md = skill.markdown_narrative

        question_patterns = re.findall(
            r"\*\*Question pattern:\*\*\s*(.+?)(?:\n|$)",
            md,
        )
        action_space_tendencies = re.findall(
            r"\*\*Action-space tendency:\*\*\s*(.+?)(?:\n|$)",
            md,
        )
        temporal_matches = re.findall(
            r"\*\*Current/latest tendency:\*\*\s*(.+?)(?:\n|$)",
            md,
        )
        misreading_blocks = re.findall(
            r"\*\*Common misreadings:\*\*\s*\n((?:- .+\n?)+)",
            md,
        )

        # Extract action guidance sections
        action_matches = re.findall(
            r"### Action guidance\s*\n(.*?)(?=\n### |\n## Direction |\Z)",
            md, re.DOTALL,
        )
        case_evidence_matches = re.findall(
            r"### Case-grounded relation evidence\s*\n(.*?)(?=\n### |\n## Direction |\Z)",
            md, re.DOTALL,
        )
        action_hint = ""
        if action_matches or case_evidence_matches:
            parts: List[str] = []
            if question_patterns:
                parts.append("Question pattern: " + "; ".join(question_patterns))
            if action_space_tendencies:
                parts.append("Action-space tendency: " + "; ".join(action_space_tendencies))
            parts.extend(m.strip() for m in action_matches if m.strip())
            evidence_parts = [m.strip() for m in case_evidence_matches if m.strip()]
            if evidence_parts:
                parts.extend([f"Case-grounded relation evidence\n{part}" for part in evidence_parts])
            if parts:
                action_hint = (
                    "[SKILL-DERIVED ACTION GUIDANCE]\n"
                    "The following action guidance was synthesized from similar training cases.\n"
                    "---\n"
                    + "\n\n".join(parts)
                    + "\n---\n"
                    "Use these as soft preferences when comparing candidate actions."
                )

        # Extract selection guidance + answer tendency + misreadings
        selection_hint = ""
        sel_matches = re.findall(
            r"### Selection guidance\s*\n(.*?)(?=\n### |\n## Direction |\Z)",
            md, re.DOTALL,
        )
        final_reasoning_matches = re.findall(
            r"### Final reasoning guidance\s*\n(.*?)(?=\n### |\n## Direction |\Z)",
            md, re.DOTALL,
        )
        tendency_matches = re.findall(
            r"\*\*Action-space tendency:\*\*\s*(.+?)(?:\n|$)",
            md,
        )

        sel_parts = [m.strip() for m in sel_matches if m.strip()]
        final_reasoning_parts = [m.strip() for m in final_reasoning_matches if m.strip()]
        if sel_parts or final_reasoning_parts or tendency_matches or misreading_blocks:
            combined_lines = []
            if question_patterns:
                combined_lines.append("Question pattern: " + "; ".join(question_patterns))
            if tendency_matches:
                combined_lines.append("Action-space tendency: " + "; ".join(tendency_matches))
            if temporal_matches:
                combined_lines.append("Temporal tendency: " + "; ".join(temporal_matches))
            combined_lines.extend(sel_parts)
            combined_lines.extend(final_reasoning_parts)
            for mr in misreading_blocks:
                combined_lines.append("Common misreadings:\n" + mr.strip())
            selection_hint = (
                "[SKILL-DERIVED SELECTION GUIDANCE]\n"
                "The following selection guidance was synthesized from similar training cases.\n"
                "---\n"
                + "\n\n".join(combined_lines)
                + "\n---\n"
                "Use these as soft guidance when choosing among final candidates."
            )

        return {
            "stage:1": "",
            "stage:2": action_hint,
            "stage:3": action_hint,
            "stage:4": selection_hint,
        }

    # --- Legacy fallback: per-stage routing ---
    hints: Dict[str, str] = {}

    # Plan guidance for stages 1-2
    plan_parts: List[str] = []
    if skill.plan_guidance:
        plan_parts.append(skill.plan_guidance)
    plan_pitfalls = [p for p in skill.pitfalls if any(kw in p.lower() for kw in ("relation", "plan", "path", "traverse", "hop"))]
    if plan_pitfalls:
        plan_parts.append("Plan pitfalls: " + "; ".join(plan_pitfalls))
    if skill.negative_experiences:
        neg_plan = [e for e in skill.negative_experiences if any(kw in e.lower() for kw in ("plan", "relation", "path"))]
        if neg_plan:
            plan_parts.append("Negative plan experience: " + "; ".join(neg_plan[:3]))
    plan_hint = "\n".join(plan_parts)
    hints["stage:1"] = plan_hint
    hints["stage:2"] = plan_hint

    # Action guidance for stage 3
    action_parts: List[str] = []
    if skill.action_guidance:
        action_parts.append(skill.action_guidance)
    action_pitfalls = [p for p in skill.pitfalls if any(kw in p.lower() for kw in ("action", "tool", "call", "query", "execute"))]
    if action_pitfalls:
        action_parts.append("Action pitfalls: " + "; ".join(action_pitfalls))
    if skill.negative_experiences:
        neg_action = [e for e in skill.negative_experiences if any(kw in e.lower() for kw in ("action", "tool", "call", "query"))]
        if neg_action:
            action_parts.append("Negative action experience: " + "; ".join(neg_action[:3]))
    if skill.error_type_distribution.get("action_error", 0) >= 2:
        action_parts.insert(0, "[ATTENTION: High action error rate in similar cases — exercise extra caution in tool calls]")
    hints["stage:3"] = "\n".join(action_parts)

    # Reasoning guidance for stage 4
    reasoning_parts: List[str] = []
    if skill.reasoning_guidance or skill.combined_reasoning_hint:
        reasoning_parts.append(skill.reasoning_guidance or skill.combined_reasoning_hint)
    if skill.scope_guidance:
        reasoning_parts.append(f"Scope: {skill.scope_guidance}")
    if skill.answer_type_guidance:
        reasoning_parts.append(f"Answer type: {skill.answer_type_guidance}")
    if skill.temporal_guidance:
        reasoning_parts.append(f"Temporal: {skill.temporal_guidance}")
    reasoning_pitfalls = [p for p in skill.pitfalls if p not in set(plan_pitfalls) | set(action_pitfalls)]
    if reasoning_pitfalls:
        reasoning_parts.append("Reasoning pitfalls: " + "; ".join(reasoning_pitfalls))
    hints["stage:4"] = "\n".join(reasoning_parts)

    return hints
