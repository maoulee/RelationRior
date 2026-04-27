"""Deterministic skill-card aggregation into a single markdown artifact.

Groups filtered CaseSkillCard objects by intent, synthesizes shared patterns,
and emits a multi-direction markdown document for injection into prompts.

Feature-gated by ``KGQA_SKILL_AGGREGATION_MODE`` env var:
  - ``per_skill``  : legacy, each card injected separately (current behavior)
  - ``aggregated`` : new default, cards aggregated into one markdown artifact
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Dict, List, Sequence

from .schemas import CaseSkillCard

# ---------------------------------------------------------------------------
# Env var helper
# ---------------------------------------------------------------------------

_AGGREGATION_MODE_ENV = "KGQA_SKILL_AGGREGATION_MODE"


def get_aggregation_mode() -> str:
    """Return current aggregation mode from env.

    Returns ``"aggregated"`` by default, or ``"per_skill"`` for legacy mode.
    """
    val = os.getenv(_AGGREGATION_MODE_ENV, "aggregated").strip().lower()
    return val if val in ("per_skill", "aggregated") else "aggregated"


# ---------------------------------------------------------------------------
# Clustering helpers
# ---------------------------------------------------------------------------

def _cluster_key(card: CaseSkillCard) -> str:
    """Determine cluster key for a card.

    Prefers ``intent_signature`` when non-empty, falls back to
    ``question_type``.
    """
    sig = getattr(card, "intent_signature", "")
    if sig:
        return sig
    return card.question_type or "unknown"


def _answer_cardinality(card: CaseSkillCard) -> str:
    """Return 'single', 'multiple', or 'unknown' from card metadata."""
    eat = card.expected_answer_type
    if eat is not None:
        card_str = getattr(eat, "cardinality", "") or getattr(eat, "count_hint", "")
        if card_str:
            return card_str.lower()
    strategy = card.answer_strategy or {}
    ec = str(strategy.get("expected_count", "")).lower()
    if "single" in ec:
        return "single"
    if "multi" in ec or "list" in ec:
        return "multiple"
    return "unknown"


def _temporal_label(card: CaseSkillCard) -> str:
    """Return a short temporal label for disagreement detection."""
    ts = card.temporal_scope
    if ts is None:
        return "none"
    return (ts.type or "none").lower()


# ---------------------------------------------------------------------------
# Cluster synthesis
# ---------------------------------------------------------------------------

def _action_space_mode_label(card: CaseSkillCard) -> str:
    """Map action_space_mode to a human-readable label.

    Returns the raw strategy value to preserve skill card fidelity,
    falling back to 'keep_whole_action_space' when unset.
    """
    strategy = card.answer_strategy or {}
    mode = strategy.get("action_space_mode", "")
    if mode:
        return mode
    return "keep_whole_action_space"


def _synthesize_direction(
    cluster: List[CaseSkillCard],
    cluster_label: str,
) -> str:
    """Build one Direction section from a cluster of skill cards.

    Uses the unified per-case template:
      Question + Key relations + Answer mode + Selection rule.
    """
    if cluster_label == "merged":
        direction_name = "When similar questions share the same answer direction"
    else:
        direction_name = cluster_label.replace("|", " / ").replace("_", " ").title()

    lines: List[str] = []
    lines.append(f"## Direction: {direction_name}")

    lines.append("")

    for c in cluster[:5]:
        q_text = c.question[:120] if c.question else "Unknown"
        core_rels = c.core_relations or []

        # Get answer strategy fields
        strategy = c.answer_strategy or {}
        mode_label = _action_space_mode_label(c)
        filter_attrs = strategy.get("filter_likely_attributes", [])
        selection_rule = strategy.get("selection_rule", "")

        lines.append(f'#### Case: "{q_text}"')
        rels_str = ", ".join(f"`{r}`" for r in core_rels[:4]) if core_rels else "varies"
        lines.append(f"Key relations: {rels_str}")
        lines.append(f"Answer mode: {mode_label}")
        if selection_rule:
            lines.append(f"Selection rule: {selection_rule}")
        elif filter_attrs:
            lines.append(f"Selection rule: filter by {', '.join(filter_attrs)}")
        if filter_attrs and mode_label != "keep_whole_action_space":
            lines.append(f"Filter attributes: {', '.join(filter_attrs)}")

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def aggregate_skills_to_markdown(
    cards: Sequence[CaseSkillCard],
    target_question: str = "",
) -> str:
    """Aggregate filtered skill cards into a single markdown artifact.

    Output contains multiple semantic direction sections when skills disagree.
    Language is advisory, not mandatory. No hard rules.

    Parameters
    ----------
    cards : Sequence[CaseSkillCard]
        Filtered skill cards to aggregate.
    target_question : str
        The question currently being answered (used in the header).

    Returns
    -------
    str
        Multi-direction markdown document.
    """
    if not cards:
        return ""

    # Step 1: Cluster by intent_signature / question_type
    clusters: Dict[str, List[CaseSkillCard]] = defaultdict(list)
    for card in cards:
        key = _cluster_key(card)
        clusters[key].append(card)

    # Step 2: Detect disagreement on answer cardinality / temporal scope
    # across clusters. If they agree, merge into one direction.
    cluster_keys = list(clusters.keys())

    def _cluster_cardinality(key: str) -> str:
        """Dominant cardinality within a cluster."""
        counts: Dict[str, int] = {}
        for c in clusters[key]:
            cv = _answer_cardinality(c)
            counts[cv] = counts.get(cv, 0) + 1
        return max(counts, key=lambda k: counts[k]) if counts else "unknown"

    def _cluster_temporal(key: str) -> str:
        """Dominant temporal label within a cluster."""
        counts: Dict[str, int] = {}
        for c in clusters[key]:
            tl = _temporal_label(c)
            counts[tl] = counts.get(tl, 0) + 1
        return max(counts, key=lambda k: counts[k]) if counts else "none"

    # Check whether clusters disagree on cardinality or temporal scope
    cardinalities_set = {_cluster_cardinality(k) for k in cluster_keys}
    temporals_set = {_cluster_temporal(k) for k in cluster_keys}

    # If all clusters agree on both axes, merge into one direction
    merged_directions: List[tuple] = []  # (label, cards)
    if len(cluster_keys) <= 1 or (len(cardinalities_set) <= 1 and len(temporals_set) <= 1):
        # Merge all into one direction
        all_cards: List[CaseSkillCard] = []
        for key in cluster_keys:
            all_cards.extend(clusters[key])
        merged_directions.append(("merged", all_cards))
    else:
        # Keep separate directions per cluster
        for key in cluster_keys:
            merged_directions.append((key, clusters[key]))

    # Step 3: Build output
    header = target_question.strip() if target_question else "(unspecified question)"
    parts: List[str] = [f"# Skill-Derived Directions for: {header}", ""]

    for idx, (label, cluster_cards) in enumerate(merged_directions):
        direction_letter = chr(ord("A") + idx)
        section = _synthesize_direction(cluster_cards, label)
        # Replace "## Direction:" with "## Direction X:" using the letter
        section = section.replace("## Direction:", f"## Direction {direction_letter}:", 1)
        parts.append(section)
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Stage extraction helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Direction / Case structured parser
# ---------------------------------------------------------------------------

def _parse_directions(
    markdown: str,
) -> List[Tuple[str, List[Tuple[str, str]]]]:
    """Parse aggregated markdown into structured directions.

    Returns
    -------
    List of ``(direction_header, [(case_title, case_body), ...])`` tuples.
    """
    if not markdown:
        return []

    directions: List[Tuple[str, List[Tuple[str, str]]]] = []
    current_dir = ""
    current_cases: List[Tuple[str, str]] = []
    current_case_title = ""
    current_case_lines: List[str] = []

    for line in markdown.splitlines():
        if line.startswith("## Direction"):
            # Flush previous case
            if current_case_title:
                current_cases.append(
                    (current_case_title, "\n".join(current_case_lines))
                )
            # Flush previous direction
            if current_dir and current_cases:
                directions.append((current_dir, current_cases))
            current_dir = line.strip()
            current_cases = []
            current_case_title = ""
            current_case_lines = []
        elif line.startswith("#### Case:"):
            # Flush previous case
            if current_case_title:
                current_cases.append(
                    (current_case_title, "\n".join(current_case_lines))
                )
            m = re.match(r'#### Case:\s*"([^"]+)"', line)
            current_case_title = m.group(1) if m else line.strip()
            current_case_lines = []
        elif current_case_title:
            current_case_lines.append(line)

    # Flush last case and direction
    if current_case_title:
        current_cases.append((current_case_title, "\n".join(current_case_lines)))
    if current_dir and current_cases:
        directions.append((current_dir, current_cases))

    return directions


def extract_action_stage_guidance(
    markdown_artifact: str,
    active_relations: set[str] | None = None,
) -> str:
    """Extract concise action-stage guidance from the aggregated markdown.

    Returns ``## Direction:`` headers, ``#### Case:`` titles with question
    text, and the ``Key relations:`` line for each case.  Optionally filters
    cases whose Key relations overlap with *active_relations*.

    This is the signal needed for stages 2-3 (action planning / path
    selection).
    """
    if not markdown_artifact:
        return ""

    # Build fast lookup for active relations when provided
    active_short: set[str] | None = None
    if active_relations:
        active_short = {r.split(".")[-1] for r in active_relations if r}

    directions = _parse_directions(markdown_artifact)
    parts: List[str] = []

    for dir_header, cases in directions:
        case_parts: List[str] = []
        for title, body in cases:
            # Find Key relations line
            rels_line = ""
            core_rels: set[str] = set()
            for line in body.strip().splitlines():
                stripped = line.strip()
                if stripped.startswith("Key relations:"):
                    rels_line = stripped
                    core_rels = set(re.findall(r"`([^`]+)`", stripped))
                    break

            # Filter by active_relations when provided
            if active_relations is not None:
                overlap = core_rels & active_relations
                if not overlap and active_short is not None:
                    short_in_case = {r.split(".")[-1] for r in core_rels}
                    overlap = short_in_case & active_short
                if not overlap:
                    continue

            if rels_line:
                case_parts.append(f'#### Case: "{title}"\n{rels_line}')

        if case_parts:
            parts.append(dir_header + "\n" + "\n\n".join(case_parts))

    return "\n\n".join(parts)


def extract_final_stage_guidance(
    markdown_artifact: str,
    active_relations: set[str] | None = None,
) -> str:
    """Extract final-stage guidance from the aggregated markdown.

    Returns ``## Direction:`` headers, ``#### Case:`` titles with question
    text, and the full case content (Key relations, Answer mode, Selection
    rule, Filter attributes) for cases whose Key relations overlap with
    *active_relations* (if supplied).

    Used at stages 4-5 only.

    Parameters
    ----------
    markdown_artifact : str
        The aggregated skill markdown.
    active_relations : set[str] | None
        If provided, only include cases where at least one relation in
        Key relations overlaps with *active_relations* (matched by full name
        or short name = last segment after ``.``).  When *None*, all
        case blocks are returned.
    """
    if not markdown_artifact:
        return ""

    # Build fast lookup for active relations when provided
    active_short: set[str] | None = None
    if active_relations:
        active_short = {r.split(".")[-1] for r in active_relations if r}

    directions = _parse_directions(markdown_artifact)
    parts: List[str] = []

    for dir_header, cases in directions:
        case_parts: List[str] = []
        for title, body in cases:
            # Parse backtick-wrapped relations from Key relations line
            core_rels: set[str] = set()
            for line in body.strip().splitlines():
                stripped = line.strip()
                if stripped.startswith("Key relations:"):
                    core_rels = set(re.findall(r"`([^`]+)`", stripped))
                    break

            # Filter by active_relations when provided
            if active_relations is not None:
                overlap = core_rels & active_relations
                if not overlap and active_short is not None:
                    short_in_case = {r.split(".")[-1] for r in core_rels}
                    overlap = short_in_case & active_short
                if not overlap:
                    continue

            full_block = body.strip()
            if full_block:
                case_parts.append(f'#### Case: "{title}"\n{full_block}')

        if case_parts:
            parts.append(dir_header + "\n" + "\n\n".join(case_parts))

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from .schemas import ExpectedAnswerType, TemporalScope

    card_a = CaseSkillCard(
        case_id="c1",
        question="Who was the president of the US in 2010?",
        question_type="temporal_single_entity",
        retrieval_fields={},
        core_relation_domains=["government"],
        core_relations=["office_holder", "president_of"],
        constraint_guidance=["year = 2010"],
        answer_strategy={"expected_count": "single"},
        action_space_experience="Traverse from country entity through office_holder with year filter.",
        final_selection_experience=["Filter candidates by start_date <= 2010 <= end_date."],
        common_pitfalls=["Confusing president with vice president."],
        common_misreadings=["Question asks for 'the' president (singular), not all office holders."],
        expected_answer_type=ExpectedAnswerType(entity_category="person", cardinality="single"),
        temporal_scope=TemporalScope(type="year_specific", reference="2010"),
        intent_signature="temporal:year_specific|answer:person_single",
        ambiguity_flags=[],
        wrong_but_related_answer_families=["vice presidents of the US"],
    )

    card_b = CaseSkillCard(
        case_id="c2",
        question="Who are the prime ministers of the UK?",
        question_type="list_entities",
        retrieval_fields={},
        core_relation_domains=["government"],
        core_relations=["prime_minister"],
        constraint_guidance=[],
        answer_strategy={"expected_count": "multiple"},
        action_space_experience="List all holders of the prime_minister relation.",
        final_selection_experience=["Return all entities matching the relation."],
        common_pitfalls=["Missing historical entries when only current is returned."],
        common_misreadings=["Question uses plural, expects a list."],
        expected_answer_type=ExpectedAnswerType(entity_category="person", cardinality="multiple"),
        temporal_scope=TemporalScope(type="all_time"),
        intent_signature="temporal:all_time|answer:person_list",
        ambiguity_flags=[],
        wrong_but_related_answer_families=["deputy prime ministers"],
    )

    card_c = CaseSkillCard(
        case_id="c3",
        question="Who was the president of the US in 2016?",
        question_type="temporal_single_entity",
        retrieval_fields={},
        core_relation_domains=["government"],
        core_relations=["office_holder", "president_of"],
        constraint_guidance=["year = 2016"],
        answer_strategy={"expected_count": "single"},
        action_space_experience="Same pattern as 2010 question -- traverse office_holder.",
        final_selection_experience=["Filter by year, return the single match."],
        common_pitfalls=["Off-by-one on year boundaries."],
        common_misreadings=["Do not return the vice president."],
        expected_answer_type=ExpectedAnswerType(entity_category="person", cardinality="single"),
        temporal_scope=TemporalScope(type="year_specific", reference="2016"),
        intent_signature="temporal:year_specific|answer:person_single",
        ambiguity_flags=[],
        wrong_but_related_answer_families=[],
    )

    md = aggregate_skills_to_markdown(
        [card_a, card_b, card_c],
        target_question="Who was the US president in 2016?",
    )

    print("=" * 60)
    print("FULL MARKDOWN ARTIFACT")
    print("=" * 60)
    print(md)
    print()

    print("=" * 60)
    print("ACTION STAGE GUIDANCE (stages 2-3)")
    print("=" * 60)
    action_g = extract_action_stage_guidance(md)
    print(action_g)
    print()

    print("=" * 60)
    print("FINAL STAGE GUIDANCE (stage 5)")
    print("=" * 60)
    final_g = extract_final_stage_guidance(md)
    print(final_g)
    print()

    # Verify they extract different subsections
    assert action_g != final_g, "Action and final guidance should differ."
    assert "Key relations:" in action_g, (
        "Action guidance should mention Key relations."
    )
    assert "Answer mode:" in final_g, (
        "Final guidance should contain Answer mode content."
    )
    # Action guidance should NOT contain Answer mode (it only has Key relations)
    assert "Answer mode:" not in action_g, (
        "Action guidance should not contain Answer mode."
    )
    print("Verification passed: extractors return different content.")
