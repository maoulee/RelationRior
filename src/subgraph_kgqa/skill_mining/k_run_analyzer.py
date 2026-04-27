"""Analyze k repeated runs of the same case to identify instability patterns.

Given a case_id and a list of RawAttemptRecord objects from multiple runs of the
same variant, this module compares runs to detect agreement/disagreement on the
final answer, identifies what changed between runs, extracts common misreadings
and instability triggers, and returns an enriched CaseSkillCard.
"""

from __future__ import annotations

import copy
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .schemas import CaseSkillCard, RawAttemptRecord


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _normalize_answer_set(answers: List[str]) -> str:
    """Return a canonical string signature for an answer set."""
    return "|".join(sorted(set(str(a).strip().lower() for a in answers if str(a).strip())))


def _answer_signature(record: RawAttemptRecord) -> str:
    """Return a canonical signature for the predicted answers of a run."""
    return _normalize_answer_set(record.predicted_answers)


def _relation_set(record: RawAttemptRecord) -> Set[str]:
    return set(record.planned_relations) | set(record.candidate_constraint_relations)


def _domain_set(record: RawAttemptRecord) -> Set[str]:
    return set(record.explored_domains)


def _set_diff(sets_a: Sequence[Set[str]], sets_b: Sequence[Set[str]]) -> Set[str]:
    """Return elements present in any of sets_a but absent from all of sets_b."""
    union_a: Set[str] = set()
    for s in sets_a:
        union_a |= s
    intersection_b: Set[str] = set()
    initialized = False
    for s in sets_b:
        if not initialized:
            intersection_b = set(s)
            initialized = True
        else:
            intersection_b &= s
    if not initialized:
        return union_a
    return union_a - intersection_b


# ---------------------------------------------------------------------------
# Misreading / trigger extraction helpers
# ---------------------------------------------------------------------------

_MISREADING_PATTERNS: List[Tuple[str, str]] = [
    (r"\binventory\b.*\bcanonical\b|\bcanonical\b.*\binventory\b",
     "inventory vs canonical interpretation"),
    (r"\btemporal\b.*\bhistorical\b|\bhistorical\b.*\btemporal\b",
     "temporal vs historical scope"),
    (r"\bcurrent\b.*\bhistory\b|\bhistory\b.*\bcurrent\b",
     "current vs full history scope"),
    (r"\bsingle\b.*\bmultiple\b|\bmultiple\b.*\bsingle\b",
     "single vs multiple answer confusion"),
    (r"\bprimary\b.*\bsecondary\b|\bsecondary\b.*\bprimary\b",
     "primary vs secondary entity confusion"),
    (r"\bfirst\b.*\blast\b|\blast\b.*\bfirst\b",
     "first vs last temporal ordering"),
]


def _detect_misreading_from_text(text: str) -> List[str]:
    """Scan error text or notes for known misreading patterns."""
    lowered = text.lower()
    misreadings: List[str] = []
    for pattern, label in _MISREADING_PATTERNS:
        if re.search(pattern, lowered):
            misreadings.append(label)
    return misreadings


def _detect_misreading_from_answer_divergence(
    *,
    majority_answers: List[str],
    minority_answers: List[str],
    ground_truth: List[str],
) -> List[str]:
    """Infer misreadings from how minority answers diverge from the majority / GT."""
    misreadings: List[str] = []
    gt_set = {a.strip().lower() for a in ground_truth if a.strip()}
    minority_set = {a.strip().lower() for a in minority_answers if a.strip()}
    majority_set = {a.strip().lower() for a in majority_answers if a.strip()}

    # Minority produced too many answers
    if len(minority_set) > len(majority_set) + 1:
        misreadings.append("over-retention of candidates (kept too many answers)")
    # Minority produced too few answers
    if len(minority_set) < len(majority_set) and len(majority_set) > 1:
        misreadings.append("premature collapse to single answer when multiple were valid")
    # Minority answers are disjoint from GT
    if minority_set and gt_set and not (minority_set & gt_set):
        misreadings.append("answer family completely disjoint from ground truth")
    # Minority partially overlaps GT but includes wrong answers
    if minority_set and gt_set and (minority_set & gt_set) and (minority_set - gt_set):
        misreadings.append("mixed correct and incorrect answers in final selection")

    return misreadings


def _extract_instability_triggers(
    *,
    majority_runs: List[RawAttemptRecord],
    minority_runs: List[RawAttemptRecord],
) -> List[str]:
    """Identify conditions present in minority (unstable) runs but absent in majority (stable) ones."""
    triggers: List[str] = []

    majority_relation_sets = [_relation_set(r) for r in majority_runs]
    minority_relation_sets = [_relation_set(r) for r in minority_runs]
    majority_domain_sets = [_domain_set(r) for r in majority_runs]
    minority_domain_sets = [_domain_set(r) for r in minority_runs]

    # Relations explored only in minority runs
    minority_only_relations = _set_diff(minority_relation_sets, majority_relation_sets)
    if minority_only_relations:
        triggers.append(
            "different relations explored: " + ", ".join(sorted(minority_only_relations)[:5])
        )

    # Domains explored only in minority runs
    minority_only_domains = _set_diff(minority_domain_sets, majority_domain_sets)
    if minority_only_domains:
        triggers.append(
            "different domains explored: " + ", ".join(sorted(minority_only_domains)[:5])
        )

    # Turn count divergence
    majority_turns = [r.turns for r in majority_runs]
    minority_turns = [r.turns for r in minority_runs]
    if majority_turns and minority_turns:
        avg_majority = sum(majority_turns) / len(majority_turns)
        avg_minority = sum(minority_turns) / len(minority_turns)
        if avg_minority > avg_majority + 1.5:
            triggers.append("longer inference trajectory on unstable runs")
        elif avg_minority < avg_majority - 1.5:
            triggers.append("shorter inference trajectory on unstable runs (premature answer)")

    # Frontend errors
    minority_error_runs = [r for r in minority_runs if r.frontend_errors > 0]
    majority_error_runs = [r for r in majority_runs if r.frontend_errors > 0]
    if minority_error_runs and not majority_error_runs:
        triggers.append("frontend validation errors in unstable runs only")

    # Constraint relation differences
    majority_constraint_rels: Set[str] = set()
    for r in majority_runs:
        majority_constraint_rels |= set(r.candidate_constraint_relations)
    minority_constraint_rels: Set[str] = set()
    for r in minority_runs:
        minority_constraint_rels |= set(r.candidate_constraint_relations)
    extra_constraints = minority_constraint_rels - majority_constraint_rels
    if extra_constraints:
        triggers.append(
            "additional constraint relations in unstable runs: "
            + ", ".join(sorted(extra_constraints)[:5])
        )
    missing_constraints = majority_constraint_rels - minority_constraint_rels
    if missing_constraints and majority_constraint_rels:
        triggers.append(
            "missing constraint relations in unstable runs: "
            + ", ".join(sorted(missing_constraints)[:5])
        )

    return triggers


def _extract_wrong_answer_families(
    *,
    minority_runs: List[RawAttemptRecord],
    majority_answers: List[str],
    ground_truth: List[str],
) -> List[str]:
    """Identify semantically distinct answer families produced by minority runs."""
    gt_set = {a.strip().lower() for a in ground_truth if a.strip()}
    majority_set = {a.strip().lower() for a in majority_answers if a.strip()}
    wrong_families: List[str] = []
    seen_wrong: Set[str] = set()

    for run in minority_runs:
        for answer in run.predicted_answers:
            answer_str = str(answer)
            answer_norm = answer_str.strip().lower()
            if not answer_norm:
                continue
            if answer_norm in gt_set:
                continue
            if answer_norm in seen_wrong:
                continue
            seen_wrong.add(answer_norm)
            if answer_norm not in majority_set:
                wrong_families.append(answer_str.strip())

    return wrong_families[:5]


def _infer_intent_clarification(
    *,
    question: str,
    misreadings: List[str],
    instability_triggers: List[str],
) -> str:
    """Generate a brief intent clarification hint based on observed misreadings."""
    lowered = question.lower()
    hints: List[str] = []

    temporal_cues = ["current", "latest", "first", "last", "most recent"]
    if any(cue in lowered for cue in temporal_cues):
        if any("temporal" in m.lower() or "history" in m.lower() or "current" in m.lower() for m in misreadings):
            hints.append("The question asks for a specific temporal scope; distinguish between current/latest and full history.")

    if any("single" in m.lower() or "multiple" in m.lower() or "collapse" in m.lower() or "over-retention" in m.lower() for m in misreadings):
        hints.append("Determine answer cardinality carefully: keep all valid answers when the question is open-ended, collapse only when a unique answer is clearly intended.")

    if any("relation" in t.lower() or "domain" in t.lower() for t in instability_triggers):
        hints.append("Choose the relation whose whole-path semantics align with the answer type; avoid graph-near but semantically off-target relations.")

    if any("constraint" in t.lower() for t in instability_triggers):
        hints.append("Apply constraint filtering carefully before final selection.")

    if not hints:
        if misreadings or instability_triggers:
            hints.append("The question intent is easily misread; verify the answer scope and the primary action space carefully.")
        else:
            hints.append("")

    return " ".join(hints).strip()


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_k_runs(
    *,
    case_id: str,
    runs: List[RawAttemptRecord],
    base_card: Optional[CaseSkillCard] = None,
) -> CaseSkillCard:
    """Analyze k repeated runs for a single case to extract instability patterns.

    Parameters
    ----------
    case_id : str
        The case identifier.
    runs : list[RawAttemptRecord]
        Raw attempt records from k repeated runs of the same case variant.
    base_card : CaseSkillCard or None
        An existing card to enrich. If None, a minimal card is created.

    Returns
    -------
    CaseSkillCard
        A CaseSkillCard with instability-aware fields populated.
    """
    if not runs:
        if base_card:
            card = copy.deepcopy(base_card)
            card.run_count = 0
            card.instability_score = 0.0
            return card
        return CaseSkillCard(
            case_id=case_id,
            question="",
            question_type="unknown",
            retrieval_fields={},
            core_relation_domains=[],
            core_relations=[],
            constraint_guidance=[],
            answer_strategy={},
            run_count=0,
            instability_score=0.0,
        )

    # --- Group runs by predicted answer signature ---
    signature_groups: Dict[str, List[RawAttemptRecord]] = {}
    for run in runs:
        sig = _answer_signature(run)
        signature_groups.setdefault(sig, []).append(run)

    # Identify the majority answer (most common signature)
    majority_sig = max(signature_groups.keys(), key=lambda s: len(signature_groups[s]))
    majority_runs = signature_groups[majority_sig]
    minority_runs: List[RawAttemptRecord] = []
    for sig, group in signature_groups.items():
        if sig != majority_sig:
            minority_runs.extend(group)

    # --- Compute instability score ---
    disagreement_count = len(runs) - len(majority_runs)
    instability_score = disagreement_count / len(runs) if runs else 0.0

    # --- Gather majority answers and ground truth ---
    majority_answers: List[str] = []
    for r in majority_runs:
        majority_answers.extend(r.predicted_answers)
    majority_answers = list(dict.fromkeys(majority_answers))  # dedupe preserving order

    ground_truth: List[str] = []
    if runs:
        ground_truth = list(runs[0].ground_truth_answers)

    # --- Extract minority answers ---
    minority_answers: List[str] = []
    for r in minority_runs:
        minority_answers.extend(r.predicted_answers)

    # --- Detect misreadings ---
    misreadings: List[str] = []

    # From error text
    for r in minority_runs:
        if r.error_text:
            misreadings.extend(_detect_misreading_from_text(r.error_text))
    # From answer divergence
    if minority_runs:
        misreadings.extend(
            _detect_misreading_from_answer_divergence(
                majority_answers=majority_answers,
                minority_answers=minority_answers,
                ground_truth=ground_truth,
            )
        )
    # Dedupe misreadings
    seen = set()
    unique_misreadings: List[str] = []
    for m in misreadings:
        if m not in seen:
            seen.add(m)
            unique_misreadings.append(m)

    # --- Extract instability triggers ---
    instability_triggers: List[str] = []
    if minority_runs:
        instability_triggers = _extract_instability_triggers(
            majority_runs=majority_runs,
            minority_runs=minority_runs,
        )

    # --- Extract wrong answer families ---
    wrong_families: List[str] = []
    if minority_runs:
        wrong_families = _extract_wrong_answer_families(
            minority_runs=minority_runs,
            majority_answers=majority_answers,
            ground_truth=ground_truth,
        )

    # --- Infer intent clarification ---
    question = runs[0].question_text if runs else ""
    intent_clarification = _infer_intent_clarification(
        question=question,
        misreadings=unique_misreadings,
        instability_triggers=instability_triggers,
    )

    # --- Build or update the CaseSkillCard ---
    if base_card is not None:
        card = copy.deepcopy(base_card)
        card.intent_clarification = intent_clarification
        card.common_misreadings = unique_misreadings
        card.instability_triggers = instability_triggers
        card.wrong_but_related_answer_families = wrong_families
        card.run_count = len(runs)
        card.instability_score = round(instability_score, 4)
        return card

    # Build a minimal card from the majority run information
    from .case_skill import (
        _derive_domains_from_relations,
        _limit_answer_bearing_relations,
    )

    representative = majority_runs[0]
    primary_relations = _limit_answer_bearing_relations(
        representative.planned_relations, max_relations=2
    )
    core_domains = _derive_domains_from_relations(primary_relations)

    return CaseSkillCard(
        case_id=case_id,
        question=question,
        question_type="unknown",
        retrieval_fields={},
        core_relation_domains=core_domains,
        core_relations=primary_relations,
        constraint_guidance=[],
        answer_strategy={},
        intent_clarification=intent_clarification,
        common_misreadings=unique_misreadings,
        instability_triggers=instability_triggers,
        wrong_but_related_answer_families=wrong_families,
        run_count=len(runs),
        instability_score=round(instability_score, 4),
    )


def analyze_k_runs_from_dicts(
    *,
    case_id: str,
    run_dicts: List[Dict[str, Any]],
    base_card: Optional[CaseSkillCard] = None,
) -> CaseSkillCard:
    """Convenience wrapper that accepts run data as dicts and converts to RawAttemptRecord."""
    runs: List[RawAttemptRecord] = []
    skipped: List[Dict[str, str]] = []
    for rd in run_dicts:
        try:
            runs.append(RawAttemptRecord(**rd))
        except Exception as exc:
            skipped.append({"record": rd.get("record_id", "unknown"), "error": str(exc)})
    if skipped:
        print(f"[k_run_analyzer] Skipped {len(skipped)} records: {skipped}", file=sys.stderr)
    return analyze_k_runs(case_id=case_id, runs=runs, base_card=base_card)
