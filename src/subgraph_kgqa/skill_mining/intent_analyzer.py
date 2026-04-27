"""Intent analysis mini-agent for question understanding.

Analyzes natural language questions to extract structured intent signals:
- Expected answer type (entity category + cardinality)
- Temporal scope (year-specific, current, all-time, etc.)
- Verb/scope inclusions (what entity types to include)
- Ambiguity flags

Feature-gated by KGQA_ENABLE_INTENT_ANALYSIS env var (default: off).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional


def is_intent_analysis_enabled() -> bool:
    return os.getenv("KGQA_ENABLE_INTENT_ANALYSIS", "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class IntentSignature:
    """Structured intent extracted from a question."""
    answer_type: str = ""               # "person", "organization", "location", "team", "language", etc.
    cardinality: str = ""               # "single", "multiple", "unknown"
    temporal_type: str = ""             # "year_specific", "as_of", "latest", "all_time", "none"
    temporal_reference: str = ""        # "2010", "current", etc.
    temporal_operator: str = ""         # "during", "before", "after", "as_of"
    scope_entity_types: List[str] = field(default_factory=list)
    scope_explanation: str = ""
    key_constraints: List[str] = field(default_factory=list)
    ambiguity_flags: List[str] = field(default_factory=list)
    intent_signature_str: str = ""      # Compact: "temporal:year_specific|answer:person_list"

    def to_signature_str(self) -> str:
        """Build compact intent signature string for retrieval matching."""
        parts = []
        if self.temporal_type and self.temporal_type != "none":
            parts.append(f"temporal:{self.temporal_type}")
        if self.answer_type:
            card_suffix = "_list" if self.cardinality == "multiple" else ""
            parts.append(f"answer:{self.answer_type}{card_suffix}")
        if self.scope_entity_types:
            parts.append(f"scope:{'+'.join(sorted(self.scope_entity_types))}")
        return "|".join(parts) if parts else "generic"


INTENT_ANALYSIS_PROMPT = """\
Analyze the following question and extract structured intent signals.
Return ONLY valid JSON (no markdown, no explanation).

Question: {question}

Extract:
1. "answer_type": What TYPE of entity is the answer? One of: person, organization, location, team, language, country, date, role, title, event, artwork, numeric, string, other
2. "cardinality": "single" or "multiple" or "unknown"
3. "temporal_type": Does the question have a time constraint? One of: "year_specific" (specific year like 2010), "as_of" (current/latest), "latest" (most recent), "all_time" (no time constraint), "none"
4. "temporal_reference": The specific time reference (e.g. "2010", "2011", "current") or empty string
5. "temporal_operator": "during", "before", "after", "as_of", or empty string
6. "scope_entity_types": List of entity types that should be INCLUDED in the answer scope (e.g. ["club", "national_team"] for "play for" questions, or empty list)
7. "scope_explanation": Brief explanation of scope decisions
8. "key_constraints": List of key constraints in the question (e.g. ["specific year 2010", "national team included"])
9. "ambiguity_flags": List of ambiguity types detected. Possible values: "temporal_ambiguous", "entity_type_ambiguous", "verb_scope_ambiguous", "role_ambiguous", "cardinality_ambiguous"

Examples:
- "who did cristiano ronaldo play for in 2010" -> {{"answer_type": "team", "cardinality": "multiple", "temporal_type": "year_specific", "temporal_reference": "2010", "temporal_operator": "during", "scope_entity_types": ["club", "national_team"], "scope_explanation": "play for includes both club and national team", "key_constraints": ["year 2010"], "ambiguity_flags": ["verb_scope_ambiguous"]}}
- "what is the capital of france" -> {{"answer_type": "location", "cardinality": "single", "temporal_type": "none", "temporal_reference": "", "temporal_operator": "", "scope_entity_types": [], "scope_explanation": "", "key_constraints": [], "ambiguity_flags": []}}
- "who is louisiana state senator" -> {{"answer_type": "person", "cardinality": "multiple", "temporal_type": "latest", "temporal_reference": "current", "temporal_operator": "as_of", "scope_entity_types": [], "scope_explanation": "senator refers to person role, not the legislative body", "key_constraints": ["Louisiana", "senator role"], "ambiguity_flags": ["entity_type_ambiguous"]}}

JSON:"""


def _parse_intent_response(raw: str) -> IntentSignature:
    """Parse LLM response into IntentSignature, with robust fallback."""
    # Extract JSON from response (may have markdown fences)
    json_match = re.search(r'\{[^{}]+\}', raw, re.S)
    if not json_match:
        return IntentSignature()

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return IntentSignature()

    sig = IntentSignature(
        answer_type=data.get("answer_type", ""),
        cardinality=data.get("cardinality", ""),
        temporal_type=data.get("temporal_type", "none"),
        temporal_reference=data.get("temporal_reference", ""),
        temporal_operator=data.get("temporal_operator", ""),
        scope_entity_types=data.get("scope_entity_types", []),
        scope_explanation=data.get("scope_explanation", ""),
        key_constraints=data.get("key_constraints", []),
        ambiguity_flags=data.get("ambiguity_flags", []),
    )
    sig.intent_signature_str = sig.to_signature_str()
    return sig


async def analyze_question_intent(
    question_text: str,
    call_llm: Optional[Callable[[List[Dict[str, str]]], Awaitable[str]]] = None,
) -> IntentSignature:
    """Analyze question intent using LLM.

    Parameters
    ----------
    question_text : str
        The natural language question to analyze.
    call_llm : async callable, optional
        LLM invocation function. If None, uses internal _call_llm helper.

    Returns
    -------
    IntentSignature
        Structured intent with answer type, temporal scope, etc.
    """
    if call_llm is None:
        call_llm = _default_call_llm

    prompt = INTENT_ANALYSIS_PROMPT.format(question=question_text)
    messages = [{"role": "user", "content": prompt}]

    try:
        raw = await call_llm(messages)
        return _parse_intent_response(raw)
    except Exception:
        return IntentSignature()


def analyze_question_intent_sync(question_text: str) -> IntentSignature:
    """Synchronous wrapper for intent analysis (for offline skill building)."""

    async def _run():
        return await analyze_question_intent(question_text)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _run())
                return future.result(timeout=30)
        return loop.run_until_complete(_run())
    except RuntimeError:
        return asyncio.run(_run())


async def _default_call_llm(messages: List[Dict[str, str]]) -> str:
    """Default LLM call using the shared client."""
    from ..llm_client import call_llm
    return await call_llm(messages, max_tokens=512, temperature=0.1, timeout_seconds=30.0, retries=2)


def resolve_skill_conflicts(
    skills: List[Any],
    question_intent: IntentSignature,
    max_skills: int = 3,
) -> tuple[List[Any], List[Dict[str, str]]]:
    """Filter/resolve skill conflicts against question intent.

    Parameters
    ----------
    skills : list of CaseSkillCard
        Retrieved skill cards.
    question_intent : IntentSignature
        Analyzed intent for the current question.
    max_skills : int
        Maximum number of skills to return.

    Returns
    -------
    tuple of (filtered_skills, conflict_log)
    """
    if not skills or not question_intent.intent_signature_str:
        return skills[:max_skills], []

    conflict_log: List[Dict[str, str]] = []

    # Score each skill by intent alignment
    scored = []
    for skill in skills:
        score = 1.0
        sig = getattr(skill, "intent_signature", "")

        # Temporal alignment
        if question_intent.temporal_type and question_intent.temporal_type != "none":
            if sig and f"temporal:{question_intent.temporal_type}" in sig:
                score += 1.0
            elif sig and "temporal:" in sig:
                # Different temporal type — penalize but don't discard
                score -= 0.5
                conflict_log.append({
                    "conflict_type": "temporal_mismatch",
                    "skill_id": getattr(skill, "case_id", "?"),
                    "detail": f"Question temporal={question_intent.temporal_type}, skill has different",
                })

        # Answer type alignment
        if question_intent.answer_type:
            expected = f"answer:{question_intent.answer_type}"
            if sig and expected in sig:
                score += 1.0
            elif sig and "answer:" in sig:
                score -= 0.5
                conflict_log.append({
                    "conflict_type": "answer_type_mismatch",
                    "skill_id": getattr(skill, "case_id", "?"),
                    "detail": f"Question expects {question_intent.answer_type}, skill differs",
                })

        scored.append((score, skill))

    # Sort by score descending, take top max_skills
    scored.sort(key=lambda x: x[0], reverse=True)
    filtered = [skill for _, skill in scored[:max_skills]]

    return filtered, conflict_log
