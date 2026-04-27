#!/usr/bin/env python3
"""
Stage-Level Attribution Scorer

Analyzes KGQA pipeline results at each stage to identify where regressions occur.

Stages analyzed:
  1. Entity Linking  – check_entities results vs ground truth core_entities
  2. Schema Discovery – explore_schema results vs ground truth core_relations
  3. Plan Quality     – plan tool's selected relations vs core_relations
  4. Action Quality   – action retrieved candidates vs truth answers
  5. Filter Quality   – filter results vs truth answers (when filter is used)
  6. Final Answer     – final_answer F1 vs truth answers

Usage:
    PYTHONPATH=/zhaoshu/subgraph/src python3 scripts/prompt_tuning/tools/stage_attribution.py \
        --results reports/skill_enhanced_test/v2_cleanup_only_v5/results.json \
        --compare reports/skill_enhanced_test/v2_no_agg_with_audit_v5/results.json \
        --test-data data/webqsp/webqsp_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Normalization (same as calculate_f1)
# ---------------------------------------------------------------------------

def normalize(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch.isalnum() or ch in " .-_")
    return " ".join(s.split())


def entity_overlap(pred: List[str], gt: List[str]) -> Tuple[float, float, float]:
    """Return (precision, recall, f1) for normalized entity overlap."""
    if not pred or not gt:
        return (0.0, 0.0, 0.0)
    p_set = {normalize(x) for x in pred}
    g_set = {normalize(x) for x in gt}
    p_set.discard("")
    g_set.discard("")
    if not p_set or not g_set:
        return (0.0, 0.0, 0.0)
    common = p_set & g_set
    precision = len(common) / len(p_set) if p_set else 0.0
    recall = len(common) / len(g_set) if g_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return (precision, recall, f1)


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(test_data_path: Path) -> Dict[str, Dict]:
    """Load test data and index by case_id."""
    gt_map: Dict[str, Dict] = {}
    with test_data_path.open() as f:
        for line in f:
            case = json.loads(line)
            cid = case.get("id", "")
            gt = case.get("ground_truth", {})
            gt_map[cid] = {
                "core_entities": gt.get("core_entities", []),
                "core_relations": gt.get("core_relations", []),
                "oracle_relations": gt.get("oracle_relations", []),
                "truth_answers": gt.get("global_truth_answers", []),
            }
    return gt_map


# ---------------------------------------------------------------------------
# Stage extraction from trajectory
# ---------------------------------------------------------------------------

def extract_stages(trajectory: List[Dict]) -> Dict[str, Any]:
    """Extract per-stage information from a case trajectory."""
    stages = {
        "entity_linking": {"entities_found": [], "source_turn": None},
        "schema_discovery": {"relations_found": [], "source_turn": None},
        "plan": {"planned_relations": [], "planned_anchors": [], "source_turn": None},
        "action": {"candidates_retrieved": [], "action_relations_used": [], "source_turn": None},
        "filter": {"candidates_after_filter": [], "filter_relations": [], "source_turn": None},
        "final_answer": {"answer": [], "source_turn": None},
    }

    for step in trajectory:
        parsed = step.get("parsed_output", {})
        queries = parsed.get("queries", [])
        candidates = parsed.get("candidates", [])
        final_answer = parsed.get("final_answer", [])
        state = step.get("state_snapshot", {})
        turn_num = step.get("turn", 0)

        # Track candidates and final answer from each turn
        if candidates:
            stages["action"]["candidates_retrieved"] = candidates
            stages["action"]["source_turn"] = turn_num

        if final_answer:
            stages["final_answer"]["answer"] = final_answer
            stages["final_answer"]["source_turn"] = turn_num

        # Parse tool calls
        for q in queries:
            tool = q.get("tool_name", "")
            args = q.get("arguments", {})

            if tool == "check_entities":
                # Entity linking
                entities = args.get("entity_substring", "")
                if entities:
                    stages["entity_linking"]["entities_found"].append(entities)
                    stages["entity_linking"]["source_turn"] = turn_num

            elif tool == "explore_schema":
                # Schema discovery - just note it happened
                stages["schema_discovery"]["source_turn"] = stages["schema_discovery"].get("source_turn") or turn_num

            elif tool == "plan":
                # Plan quality
                related = args.get("related", [])
                maybe_related = args.get("maybe_related", [])
                constraint_rels = args.get("constraint_relations", [])
                anchors = args.get("anchor", [])
                stages["plan"]["planned_relations"] = related + maybe_related + constraint_rels
                stages["plan"]["planned_anchors"] = anchors
                stages["plan"]["source_turn"] = turn_num

            elif tool == "select_action":
                # Action execution
                stages["action"]["source_turn"] = stages["action"].get("source_turn") or turn_num

            elif tool == "filter":
                # Filter
                constraint_rels = args.get("constraint_relations", [])
                stages["filter"]["filter_relations"] = constraint_rels
                stages["filter"]["source_turn"] = turn_num

        # Extract entities and relations from backend results
        for br in step.get("backend_results", []):
            tool_name = getattr(br, "tool_name", "") if hasattr(br, "tool_name") else br.get("tool_name", "")
            resp = getattr(br, "response_text", "") if hasattr(br, "response_text") else br.get("response_text", "")

            if tool_name == "check_entities" and resp:
                # Extract entity names from response
                found = _extract_entities_from_check(resp)
                if found:
                    stages["entity_linking"]["entities_found"] = found
                    stages["entity_linking"]["source_turn"] = turn_num

            elif tool_name == "explore_schema" and resp:
                # Extract relation names from response
                found = _extract_relations_from_schema(resp)
                if found:
                    stages["schema_discovery"]["relations_found"] = found
                    stages["schema_discovery"]["source_turn"] = turn_num

            elif tool_name in ("action", "match_pattern") and resp:
                # Extract action results (leaf entities)
                found = _extract_entities_from_action(resp)
                if found:
                    stages["action"]["candidates_retrieved"] = found
                    stages["action"]["source_turn"] = turn_num

            elif tool_name == "filter" and resp:
                # Extract filtered candidates
                found = _extract_entities_from_filter(resp)
                if found:
                    stages["filter"]["candidates_after_filter"] = found
                    stages["filter"]["source_turn"] = turn_num

    # If filter stage has no results, copy action candidates as filter output
    if not stages["filter"]["candidates_after_filter"] and stages["action"]["candidates_retrieved"]:
        stages["filter"]["candidates_after_filter"] = stages["action"]["candidates_retrieved"]

    return stages


def _extract_entities_from_check(text: str) -> List[str]:
    """Extract entity names from check_entities backend response."""
    entities = []
    # Pattern: "- Entity Name [Context: ..."
    for m in re.finditer(r"-\s+([^\[\n]+?)(?:\s*\[Context:|\s*$)", text):
        name = m.group(1).strip()
        if name and len(name) > 1:
            entities.append(name)
    return entities[:20]


def _extract_relations_from_schema(text: str) -> List[str]:
    """Extract relation names from explore_schema backend response."""
    relations = []
    # Pattern: "- domain.relation_name" or "- full.relation.path"
    for m in re.finditer(r"-\s+([a-z_]+\.[a-z_]+\.[a-z_]+(?:\.[a-z_]+)*)", text):
        rel = m.group(1).strip()
        if rel:
            relations.append(rel)
    return relations[:50]


def _extract_entities_from_action(text: str) -> List[str]:
    """Extract entity names from action backend response."""
    entities = []
    # Look for "Leaf Entities" or "Target Entities" or candidate lists
    for pattern in [
        r'(?:Leaf|Target|CVT-Expanded)\s+(?:Entities?)\s*\([^)]*\):\s*\n\s*\[([^\]]+)\]',
        r'Candidates?:\s*\[([^\]]+)\]',
    ]:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            list_str = m.group(1)
            for item in re.findall(r"'([^']*)'|\"([^\"]*)\"|,\s*([^,\[\]]+)", list_str):
                for val in item:
                    if val and val.strip():
                        entities.append(val.strip())
    if not entities:
        # Try simpler pattern: lines starting with "  - entityname"
        for m in re.finditer(r"^\s+-\s+([A-Z][^\n]{2,80})", text, re.MULTILINE):
            entities.append(m.group(1).strip())
    return entities[:50]


def _extract_entities_from_filter(text: str) -> List[str]:
    """Extract filtered entity names from filter backend response."""
    entities = []
    # Look for "Valid:" or "Kept:" sections
    for pattern in [
        r'(?:Valid|Kept|Matched|Passed)\s*:\s*(\d+)\s*\n\s*\[([^\]]+)\]',
        r'(?:Valid|Kept|Matched|Passed)\s*:\s*\n\s*\[([^\]]+)\]',
    ]:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            list_str = m.group(1) if '[' in (m.group(1) or '') else m.group(0)
            # Extract from the brackets
            bracket_match = re.search(r'\[([^\]]+)\]', list_str)
            if bracket_match:
                list_str = bracket_match.group(1)
            for item in re.findall(r"'([^']*)'|\"([^\"]*)\"", list_str):
                for val in item:
                    if val and val.strip():
                        entities.append(val.strip())
    return entities[:50]


# ---------------------------------------------------------------------------
# Per-case scoring
# ---------------------------------------------------------------------------

def score_case_stages(
    case: Dict[str, Any],
    gt: Dict[str, Any],
) -> Dict[str, Any]:
    """Score a single case at each pipeline stage."""
    trajectory = case.get("trajectory", [])
    stages = extract_stages(trajectory)

    truth_answers = gt.get("truth_answers", [])
    core_entities = gt.get("core_entities", [])
    core_relations = gt.get("core_relations", [])

    # Stage 1: Entity Linking - did check_entities find entities matching core_entities?
    el_pred = stages["entity_linking"]["entities_found"]
    el_p, el_r, el_f1 = entity_overlap(el_pred, core_entities)

    # Stage 2: Schema Discovery - did explore_schema find relations matching core_relations?
    sd_pred = stages["schema_discovery"]["relations_found"]
    # Partial match: any core_relation substring appears in found relations
    sd_recall = 0.0
    if core_relations and sd_pred:
        matched = sum(1 for cr in core_relations
                      if any(cr in sr or sr in cr for sr in sd_pred))
        sd_recall = matched / len(core_relations)

    # Stage 3: Plan Quality - did plan select relations matching core_relations?
    plan_pred = stages["plan"]["planned_relations"]
    plan_recall = 0.0
    plan_precision = 0.0
    if core_relations and plan_pred:
        matched_gt = sum(1 for cr in core_relations
                         if any(cr in pr or pr in cr for pr in plan_pred))
        plan_recall = matched_gt / len(core_relations)
        matched_pred = sum(1 for pr in plan_pred
                          if any(cr in pr or pr in cr for cr in core_relations))
        plan_precision = matched_pred / len(plan_pred) if plan_pred else 0.0
    plan_f1 = (2 * plan_precision * plan_recall / (plan_precision + plan_recall)
               if (plan_precision + plan_recall) > 0 else 0.0)

    # Stage 4: Action Quality - did action retrieve truth answers as candidates?
    action_candidates = stages["action"]["candidates_retrieved"]
    ac_p, ac_r, ac_f1 = entity_overlap(action_candidates, truth_answers)

    # Stage 5: Filter Quality - did filter keep truth answers?
    filter_candidates = stages["filter"]["candidates_after_filter"]
    fc_p, fc_r, fc_f1 = entity_overlap(filter_candidates, truth_answers)

    # Stage 6: Final Answer F1
    fa_pred = stages["final_answer"]["answer"]
    fa_p, fa_r, fa_f1 = entity_overlap(fa_pred, truth_answers)

    return {
        "case_id": case.get("case_id", "?"),
        "question": case.get("question", ""),
        "overall_f1": case.get("f1", 0.0),
        "stages": {
            "entity_linking": {"precision": el_p, "recall": el_r, "f1": el_f1,
                              "found": len(el_pred), "gt_count": len(core_entities)},
            "schema_discovery": {"recall": sd_recall,
                                "found": len(sd_pred), "gt_count": len(core_relations)},
            "plan": {"precision": plan_precision, "recall": plan_recall, "f1": plan_f1,
                    "planned": len(plan_pred), "gt_count": len(core_relations)},
            "action": {"precision": ac_p, "recall": ac_r, "f1": ac_f1,
                      "candidates": len(action_candidates), "gt_count": len(truth_answers)},
            "filter": {"precision": fc_p, "recall": fc_r, "f1": fc_f1,
                      "candidates": len(filter_candidates), "gt_count": len(truth_answers)},
            "final_answer": {"precision": fa_p, "recall": fa_r, "f1": fa_f1,
                           "answer_count": len(fa_pred), "gt_count": len(truth_answers)},
        },
        "turns": case.get("turns", 0),
        "frontend_errors": case.get("frontend_errors", 0),
    }


# ---------------------------------------------------------------------------
# Aggregate reporting
# ---------------------------------------------------------------------------

def aggregate_stages(scored_cases: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Aggregate per-stage metrics across all cases."""
    stage_names = ["entity_linking", "schema_discovery", "plan", "action", "filter", "final_answer"]
    agg = {}
    for sn in stage_names:
        metrics = defaultdict(list)
        for sc in scored_cases:
            stage = sc["stages"].get(sn, {})
            for k, v in stage.items():
                if isinstance(v, (int, float)):
                    metrics[k].append(v)
        agg[sn] = {k: sum(v) / len(v) for k, v in metrics.items() if v}
        agg[sn]["count"] = len(scored_cases)
    return agg


def render_stage_table(agg: Dict, label: str) -> str:
    """Render aggregate stage metrics as markdown table."""
    lines = [
        f"## Stage Metrics: {label}\n",
        "| Stage | Avg Recall | Avg Precision | Avg F1 | Notes |",
        "|-------|-----------|---------------|--------|-------|",
    ]
    for sn in ["entity_linking", "schema_discovery", "plan", "action", "filter", "final_answer"]:
        m = agg.get(sn, {})
        recall = m.get("recall", 0.0)
        precision = m.get("precision", 0.0)
        f1 = m.get("f1", 0.0)
        notes_parts = []
        if "found" in m:
            notes_parts.append(f"found={m['found']:.1f}")
        if "candidates" in m:
            notes_parts.append(f"cands={m['candidates']:.1f}")
        if "planned" in m:
            notes_parts.append(f"planned={m['planned']:.1f}")
        if "gt_count" in m:
            notes_parts.append(f"gt={m['gt_count']:.1f}")
        notes = ", ".join(notes_parts)
        lines.append(f"| {sn} | {recall:.3f} | {precision:.3f} | {f1:.3f} | {notes} |")

    return "\n".join(lines) + "\n"


def render_comparison_table(
    agg_a: Dict, agg_b: Dict,
    label_a: str, label_b: str,
) -> str:
    """Render side-by-side stage comparison."""
    lines = [
        "## Stage Comparison\n",
        f"| Stage | Recall({label_a}) | Recall({label_b}) | Δ Recall | "
        f"F1({label_a}) | F1({label_b}) | Δ F1 |",
        "|-------|-----------------|-----------------|----------|"
        "------------|------------|-------|",
    ]
    for sn in ["entity_linking", "schema_discovery", "plan", "action", "filter", "final_answer"]:
        ma = agg_a.get(sn, {})
        mb = agg_b.get(sn, {})
        ra = ma.get("recall", 0.0)
        rb = mb.get("recall", 0.0)
        fa = ma.get("f1", 0.0)
        fb = mb.get("f1", 0.0)
        dr = rb - ra
        df = fb - fa
        lines.append(
            f"| {sn} | {ra:.3f} | {rb:.3f} | {dr:+.3f} | "
            f"{fa:.3f} | {fb:.3f} | {df:+.3f} |"
        )
    return "\n".join(lines) + "\n"


def render_per_case_diff(
    scored_a: List[Dict], scored_b: List[Dict],
    label_a: str, label_b: str,
    top_n: int = 30,
    focus_stage: Optional[str] = None,
) -> str:
    """Render per-case stage comparison for cases with biggest F1 delta."""
    map_a = {s["case_id"]: s for s in scored_a}
    map_b = {s["case_id"]: s for s in scored_b}
    common = sorted(set(map_a.keys()) & set(map_b.keys()))

    rows = []
    for cid in common:
        a = map_a[cid]
        b = map_b[cid]
        delta_f1 = b["overall_f1"] - a["overall_f1"]

        # Find stage with biggest recall drop
        stage_deltas = {}
        for sn in ["entity_linking", "schema_discovery", "plan", "action", "filter", "final_answer"]:
            sa = a["stages"].get(sn, {})
            sb = b["stages"].get(sn, {})
            dr = sb.get("recall", 0) - sa.get("recall", 0)
            df = sb.get("f1", 0) - sa.get("f1", 0)
            stage_deltas[sn] = {"recall_delta": dr, "f1_delta": df}

        rows.append({
            "cid": cid,
            "question": a["question"][:50],
            "f1_a": a["overall_f1"],
            "f1_b": b["overall_f1"],
            "delta": delta_f1,
            "stage_deltas": stage_deltas,
        })

    rows.sort(key=lambda r: r["delta"])

    # Find worst bottleneck stage
    stage_drop_totals = defaultdict(float)
    for r in rows:
        for sn, sd in r["stage_deltas"].items():
            stage_drop_totals[sn] += sd["recall_delta"]

    lines = [
        f"## Per-Case Stage Attribution (top {top_n} regressed)\n",
        f"### Stage Recall Drop Summary (across all {len(rows)} cases)\n",
        "| Stage | Total Recall Δ | Avg Recall Δ |",
        "|-------|---------------|-------------|",
    ]
    for sn in ["entity_linking", "schema_discovery", "plan", "action", "filter", "final_answer"]:
        total = stage_drop_totals[sn]
        avg = total / len(rows) if rows else 0
        lines.append(f"| {sn} | {total:+.2f} | {avg:+.4f} |")

    lines.extend([
        "",
        f"### Top {top_n} Regressed Cases\n",
        "| Case | F1(A) | F1(B) | Δ | Biggest Stage Drop |",
        "|------|-------|-------|---|-------------------|",
    ])
    for r in rows[:top_n]:
        # Find worst stage
        worst_stage = min(r["stage_deltas"].items(), key=lambda x: x[1]["recall_delta"])
        lines.append(
            f"| {r['cid']} | {r['f1_a']:.2f} | {r['f1_b']:.2f} | {r['delta']:+.2f} "
            f"| {worst_stage[0]} ({worst_stage[1]['recall_delta']:+.2f}) |"
        )

    return "\n".join(lines) + "\n"


def render_error_classification(scored: List[Dict], gt_map: Dict) -> str:
    """Classify errors by pipeline stage for a single result set."""
    lines = [
        "## Error Classification by Stage\n",
    ]

    stage_error_counts = Counter()
    total = len(scored)
    correct = sum(1 for s in scored if s["overall_f1"] >= 0.5)
    wrong = total - correct

    for sc in scored:
        if sc["overall_f1"] >= 0.5:
            continue
        stages = sc["stages"]
        # Classify: which stage first fails?
        if stages["entity_linking"]["recall"] < 0.01:
            stage_error_counts["entity_linking_fail"] += 1
        elif stages["plan"]["recall"] < 0.01:
            stage_error_counts["plan_fail"] += 1
        elif stages["action"]["recall"] < 0.01:
            stage_error_counts["action_fail"] += 1
        elif stages["filter"]["recall"] < stages["action"]["recall"] - 0.1:
            stage_error_counts["filter_loses_candidates"] += 1
        elif stages["final_answer"]["f1"] < stages["filter"]["recall"]:
            stage_error_counts["final_answer_selection_error"] += 1
        else:
            stage_error_counts["unknown"] += 1

    lines.append(f"**Total**: {total}, **Correct (F1≥0.5)**: {correct}, **Wrong**: {wrong}\n")
    lines.append("| Error Type | Count | % of Wrong |")
    lines.append("|-----------|-------|------------|")
    for etype, count in stage_error_counts.most_common():
        pct = count / wrong * 100 if wrong else 0
        lines.append(f"| {etype} | {count} | {pct:.1f}% |")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage-level attribution scoring")
    parser.add_argument("--results", type=Path, required=True, help="Primary results.json")
    parser.add_argument("--compare", type=Path, default=None, help="Comparison results.json")
    parser.add_argument("--test-data", type=Path, default=Path("data/webqsp/webqsp_test.jsonl"),
                       help="Test data for ground truth")
    parser.add_argument("--top-n", type=int, default=30, help="Top N cases to show in comparison")
    parser.add_argument("--label", default=None, help="Label for primary results")
    parser.add_argument("--compare-label", default=None, help="Label for comparison results")
    args = parser.parse_args()

    # Load ground truth
    print(f"Loading ground truth from {args.test_data}...", file=sys.stderr)
    gt_map = load_ground_truth(args.test_data)
    print(f"  Ground truth for {len(gt_map)} cases", file=sys.stderr)

    # Load primary results
    with open(args.results) as f:
        primary_data = json.load(f)
    label_a = args.label or args.results.parent.name

    # Score primary
    scored_a = []
    for case in primary_data:
        cid = case.get("case_id", "")
        gt = gt_map.get(cid, {})
        if not gt.get("truth_answers"):
            continue
        scored_a.append(score_case_stages(case, gt))
    print(f"  Scored {len(scored_a)}/{len(primary_data)} cases for {label_a}", file=sys.stderr)

    agg_a = aggregate_stages(scored_a)

    report_parts = [
        "# Stage Attribution Report\n",
        f"**Primary**: `{label_a}` ({len(scored_a)} cases)\n",
        render_stage_table(agg_a, label_a),
        render_error_classification(scored_a, gt_map),
    ]

    # Load comparison if provided
    if args.compare:
        with open(args.compare) as f:
            compare_data = json.load(f)
        label_b = args.compare_label or args.compare.parent.name

        scored_b = []
        for case in compare_data:
            cid = case.get("case_id", "")
            gt = gt_map.get(cid, {})
            if not gt.get("truth_answers"):
                continue
            scored_b.append(score_case_stages(case, gt))
        print(f"  Scored {len(scored_b)}/{len(compare_data)} cases for {label_b}", file=sys.stderr)

        agg_b = aggregate_stages(scored_b)

        report_parts.extend([
            f"\n**Comparison**: `{label_b}` ({len(scored_b)} cases)\n",
            render_stage_table(agg_b, label_b),
            render_comparison_table(agg_a, agg_b, label_a, label_b),
            render_per_case_diff(scored_a, scored_b, label_a, label_b, top_n=args.top_n),
        ])

    report = "\n".join(report_parts)
    print(report)

    # Save
    out_path = args.results.parent / "stage_attribution_report.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\n> Saved to: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
