#!/usr/bin/env python3
"""Analyze test results and compute per-stage (per-turn) F1 scores.

Usage:
    PYTHONPATH=/zhaoshu/subgraph/src python3 scripts/prompt_tuning/tools/per_stage_scoring.py \
        --results /path/to/results.json \
        [--compare /path/to/other_results.json] \
        [--top-n 20] \
        [--focus fe|regression|improvement] \
        [--dir /path/to/dir_with_results_json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from subgraph_kgqa.rl.plugin import calculate_f1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: Path) -> List[Dict[str, Any]]:
    """Load a results.json file and return list of case dicts."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_results_from_dir(directory: Path) -> List[Dict[str, Any]]:
    """Load all results.json files found directly under *directory*."""
    combined: List[Dict[str, Any]] = []
    for p in sorted(directory.glob("**/results.json")):
        combined.extend(load_results(p))
    return combined


# ---------------------------------------------------------------------------
# Per-turn analysis
# ---------------------------------------------------------------------------

def _safe_final_answer(parsed_output: Optional[Dict]) -> Optional[List[str]]:
    """Return final_answer list if it is truthy, else None."""
    if not parsed_output:
        return None
    fa = parsed_output.get("final_answer")
    if fa and isinstance(fa, list) and len(fa) > 0:
        return fa
    return None


def analyze_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """Compute per-turn metrics for a single case.

    Returns a dict with:
        case_id, question, ground_truth, overall_f1,
        total_turns, turns: list of per-turn dicts
    """
    cid = case.get("case_id", "?")
    question = case.get("question", "")
    gt: List[str] = case.get("ground_truth", [])
    overall_f1 = case.get("f1", 0.0)
    trajectory = case.get("trajectory") or []

    turn_records: List[Dict[str, Any]] = []
    cumulative_fe = 0

    for step in trajectory:
        turn_num = step.get("turn", len(turn_records) + 1)
        parsed = step.get("parsed_output")
        fe_list = step.get("frontend_errors") or []
        cumulative_fe += len(fe_list)

        fa = _safe_final_answer(parsed)
        turn_f1: Optional[float] = None
        if fa is not None:
            turn_f1 = calculate_f1(fa, gt)

        consistency_info = step.get("consistency") or {}
        consistency_used = consistency_info.get("consistency_used", False)

        turn_records.append({
            "turn": turn_num,
            "final_answer": fa,
            "f1": turn_f1,
            "cumulative_fe": cumulative_fe,
            "fe_this_turn": len(fe_list),
            "consistency_used": consistency_used,
            "consistency_info": consistency_info,
        })

    return {
        "case_id": cid,
        "question": question,
        "ground_truth": gt,
        "overall_f1": overall_f1,
        "total_turns": len(trajectory),
        "turns": turn_records,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_f1(val: Optional[float]) -> str:
    if val is None:
        return "  -  "
    return f"{val:.2f}"


def _fmt_turn_f1s(turns: List[Dict]) -> str:
    parts = []
    for t in turns:
        label = f"T{t['turn']}"
        val = _fmt_f1(t["f1"])
        parts.append(f"{label}={val}")
    return " ".join(parts)


def _fmt_cumulative_fes(turns: List[Dict]) -> str:
    parts = []
    for t in turns:
        parts.append(f"T{t['turn']}={t['cumulative_fe']}")
    return " ".join(parts)


def _fmt_consistency(turns: List[Dict]) -> str:
    parts = []
    for t in turns:
        marker = "Y" if t["consistency_used"] else "."
        parts.append(f"T{t['turn']}={marker}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Markdown table output
# ---------------------------------------------------------------------------

def render_case_table(analyses: List[Dict], title: str = "Per-Case Per-Turn F1") -> str:
    """Render a markdown table of per-case results."""
    lines: List[str] = []
    lines.append(f"## {title}\n")

    # Header
    lines.append("| Case ID | Question (truncated) | Turn-by-turn F1 | Cum FE | Consistency | Final F1 | Turns |")
    lines.append("|---------|---------------------|-----------------|--------|-------------|----------|-------|")

    for a in analyses:
        q_short = a["question"][:40] + ("..." if len(a["question"]) > 40 else "")
        f1_str = _fmt_turn_f1s(a["turns"])
        fe_str = _fmt_cumulative_fes(a["turns"])
        con_str = _fmt_consistency(a["turns"])
        lines.append(
            f"| {a['case_id']} | {q_short} | {f1_str} | {fe_str} | {con_str} "
            f"| {a['overall_f1']:.2f} | {a['total_turns']} |"
        )

    return "\n".join(lines) + "\n"


def render_aggregate_stats(analyses: List[Dict], label: str = "Aggregate") -> str:
    """Render aggregate per-turn statistics as a markdown table."""
    lines: List[str] = []
    lines.append(f"## {label} Aggregate Per-Turn Statistics\n")

    max_turn = max((a["total_turns"] for a in analyses), default=0)

    # Collect per-turn data
    fa_counts: Dict[int, int] = Counter()       # how many cases had final_answer at turn N
    f1_sums: Dict[int, float] = defaultdict(float)
    f1_counts: Dict[int, int] = Counter()
    fe_sums: Dict[int, int] = defaultdict(int)   # cumulative FE at turn N
    fe_counts: Dict[int, int] = Counter()
    con_counts: Dict[int, int] = Counter()       # consistency_used at turn N
    total_cases = len(analyses)

    for a in analyses:
        for t in a["turns"]:
            tn = t["turn"]
            if t["final_answer"] is not None:
                fa_counts[tn] += 1
                f1_sums[tn] += t["f1"] if t["f1"] is not None else 0.0
                f1_counts[tn] += 1
            fe_sums[tn] += t["cumulative_fe"]
            fe_counts[tn] += 1
            if t["consistency_used"]:
                con_counts[tn] += 1

    lines.append("| Turn | Cases w/ FA | FA Rate | Avg F1 (when FA) | Avg Cum FE | Consistency Rate |")
    lines.append("|------|------------|---------|-------------------|------------|------------------|")

    for tn in range(1, max_turn + 1):
        n_at_turn = sum(1 for a in analyses if any(t["turn"] == tn for t in a["turns"]))
        fa_rate = fa_counts[tn] / n_at_turn if n_at_turn else 0.0
        avg_f1 = f1_sums[tn] / f1_counts[tn] if f1_counts[tn] else 0.0
        avg_fe = fe_sums[tn] / fe_counts[tn] if fe_counts[tn] else 0.0
        con_rate = con_counts[tn] / n_at_turn if n_at_turn else 0.0
        lines.append(
            f"| T{tn} | {fa_counts[tn]}/{n_at_turn} | {fa_rate:.2f} | {avg_f1:.3f} "
            f"| {avg_fe:.2f} | {con_rate:.2f} |"
        )

    # Overall summary
    lines.append(f"\n**Total cases**: {total_cases}")
    cases_with_fa = sum(1 for a in analyses if any(t["final_answer"] is not None for t in a["turns"]))
    lines.append(f"**Cases with any final_answer**: {cases_with_fa}/{total_cases}")
    avg_final = sum(a["overall_f1"] for a in analyses) / total_cases if total_cases else 0.0
    lines.append(f"**Average final F1**: {avg_final:.4f}")
    total_fe = sum(t["cumulative_fe"] for a in analyses for t in a["turns"] if t["turn"] == a["total_turns"])
    lines.append(f"**Total frontend errors (at last turn)**: {total_fe}")

    return "\n".join(lines) + "\n"


def render_comparison(
    analyses_a: List[Dict],
    analyses_b: List[Dict],
    label_a: str = "A",
    label_b: str = "B",
    top_n: int = 20,
) -> str:
    """Render a case-by-case comparison between two result sets."""
    lines: List[str] = []
    lines.append("## Case-by-Case Comparison\n")

    map_a = {a["case_id"]: a for a in analyses_a}
    map_b = {b["case_id"]: b for b in analyses_b}

    common_ids = sorted(set(map_a.keys()) & set(map_b.keys()))
    only_a = sorted(set(map_a.keys()) - set(map_b.keys()))
    only_b = sorted(set(map_b.keys()) - set(map_a.keys()))

    lines.append(f"**Common cases**: {len(common_ids)}, **Only in {label_a}**: {len(only_a)}, **Only in {label_b}**: {len(only_b)}\n")

    # Build comparison rows
    rows: List[Dict[str, Any]] = []
    for cid in common_ids:
        a = map_a[cid]
        b = map_b[cid]
        delta = b["overall_f1"] - a["overall_f1"]

        # Check trajectory divergence: same number of turns?
        same_turns = a["total_turns"] == b["total_turns"]

        # Per-turn F1 comparison
        turn_deltas: Dict[int, float] = {}
        max_t = max(a["total_turns"], b["total_turns"])
        a_turn_map = {t["turn"]: t for t in a["turns"]}
        b_turn_map = {t["turn"]: t for t in b["turns"]}
        for tn in range(1, max_t + 1):
            at = a_turn_map.get(tn)
            bt = b_turn_map.get(tn)
            af = at["f1"] if at else None
            bf = bt["f1"] if bt else None
            if af is not None and bf is not None:
                turn_deltas[tn] = bf - af

        # Check if FE patterns differ
        a_fe_last = a["turns"][-1]["cumulative_fe"] if a["turns"] else 0
        b_fe_last = b["turns"][-1]["cumulative_fe"] if b["turns"] else 0
        fe_diverged = a_fe_last != b_fe_last

        rows.append({
            "case_id": cid,
            "question": a["question"],
            "a_f1": a["overall_f1"],
            "b_f1": b["overall_f1"],
            "delta": delta,
            "same_turns": same_turns,
            "turn_deltas": turn_deltas,
            "a_total_turns": a["total_turns"],
            "b_total_turns": b["total_turns"],
            "a_fe_last": a_fe_last,
            "b_fe_last": b_fe_last,
            "fe_diverged": fe_diverged,
            "a_analysis": a,
            "b_analysis": b,
        })

    # Sort by absolute delta
    rows.sort(key=lambda r: -abs(r["delta"]))

    # Improvements / regressions summary
    improved = [r for r in rows if r["delta"] > 0]
    regressed = [r for r in rows if r["delta"] < 0]
    unchanged = [r for r in rows if r["delta"] == 0]
    diverged = [r for r in rows if not r["same_turns"] or r["fe_diverged"]]

    lines.append("### Summary\n")
    lines.append(f"- Improved ({label_b} > {label_a}): **{len(improved)}**")
    lines.append(f"- Regressed ({label_b} < {label_a}): **{len(regressed)}**")
    lines.append(f"- Unchanged: **{len(unchanged)}**")
    lines.append(f"- Trajectory diverged: **{len(diverged)}**\n")

    if improved:
        mean_imp = sum(r["delta"] for r in improved) / len(improved)
        lines.append(f"- Mean improvement: **+{mean_imp:.4f}**")
    if regressed:
        mean_reg = sum(r["delta"] for r in regressed) / len(regressed)
        lines.append(f"- Mean regression: **{mean_reg:.4f}**")

    # Top-N table
    lines.append(f"\n### Top {top_n} by |delta|\n")
    lines.append("| Case ID | Q (trunc) | F1(A) | F1(B) | Delta | Turns A | Turns B | FE A | FE B | Diverted? |")
    lines.append("|---------|-----------|-------|-------|-------|---------|---------|------|------|-----------|")

    for r in rows[:top_n]:
        q_short = r["question"][:30] + ("..." if len(r["question"]) > 30 else "")
        div_marker = "**Y**" if (not r["same_turns"] or r["fe_diverged"]) else ""
        lines.append(
            f"| {r['case_id']} | {q_short} | {r['a_f1']:.2f} | {r['b_f1']:.2f} "
            f"| {r['delta']:+.2f} | {r['a_total_turns']} | {r['b_total_turns']} "
            f"| {r['a_fe_last']} | {r['b_fe_last']} | {div_marker} |"
        )

    # Per-turn comparison for top-N
    lines.append(f"\n### Per-Turn F1 Comparison (top {min(top_n, 10)} cases)\n")
    for r in rows[:min(top_n, 10)]:
        lines.append(f"#### {r['case_id']} (delta={r['delta']:+.2f})\n")
        a = r["a_analysis"]
        b = r["b_analysis"]
        lines.append(f"| Turn | F1({label_a}) | F1({label_b}) | Delta |")
        lines.append("|------|-----------|-----------|-------|")
        max_t = max(a["total_turns"], b["total_turns"])
        a_map = {t["turn"]: t for t in a["turns"]}
        b_map = {t["turn"]: t for t in b["turns"]}
        for tn in range(1, max_t + 1):
            af = _fmt_f1(a_map[tn]["f1"]) if tn in a_map else "n/a"
            bf = _fmt_f1(b_map[tn]["f1"]) if tn in b_map else "n/a"
            if tn in r["turn_deltas"]:
                dd = f"{r['turn_deltas'][tn]:+.2f}"
            else:
                dd = "-"
            lines.append(f"| T{tn} | {af} | {bf} | {dd} |")
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def apply_focus(
    analyses: List[Dict],
    focus: Optional[str],
) -> List[Dict]:
    """Filter analyses based on --focus flag."""
    if focus is None:
        return analyses
    focus = focus.lower().strip()
    if focus == "fe":
        return [a for a in analyses if any(t["cumulative_fe"] > 0 for t in a["turns"])]
    # unknown focus => no filter
    return analyses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_report(
    results_paths: List[Path],
    compare_path: Optional[Path] = None,
    top_n: int = 20,
    focus: Optional[str] = None,
) -> str:
    """Build the full markdown report string."""
    sections: List[str] = []

    # Load primary results
    primary: List[Dict] = []
    for p in results_paths:
        primary.extend(load_results(p))

    if not primary:
        return "No results found.\n"

    analyses = [analyze_case(c) for c in primary]

    # Apply focus filter
    filtered = apply_focus(analyses, focus)
    if focus and len(filtered) < len(analyses):
        sections.append(f"> Focus filter `{focus}`: showing {len(filtered)}/{len(analyses)} cases\n")

    # Header
    src_label = ", ".join(str(p) for p in results_paths)
    sections.append("# Per-Stage Scoring Report\n")
    sections.append(f"**Source**: `{src_label}`\n")
    sections.append(f"**Cases**: {len(primary)}\n")

    # Per-case table
    sections.append(render_case_table(filtered, title="Per-Case Per-Turn F1"))

    # Aggregate stats
    sections.append(render_aggregate_stats(analyses, label="Primary"))

    # Comparison
    if compare_path is not None:
        compare_data = load_results(compare_path)
        comp_analyses = [analyze_case(c) for c in compare_data]
        comp_filtered = apply_focus(comp_analyses, focus)
        sections.append(render_comparison(
            analyses, comp_analyses,
            label_a=results_paths[0].stem,
            label_b=compare_path.stem,
            top_n=top_n,
        ))
        # Also print aggregate for comparison set
        sections.append(render_aggregate_stats(comp_analyses, label="Comparison"))

    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description="Per-stage (per-turn) F1 scoring for KGQA results")
    parser.add_argument(
        "--results", nargs="+", type=Path,
        help="Path(s) to results.json file(s)",
    )
    parser.add_argument(
        "--dir", type=Path, default=None,
        help="Directory to scan for results.json files",
    )
    parser.add_argument(
        "--compare", type=Path, default=None,
        help="Second results.json to compare against",
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="Number of top-delta cases to show in comparison (default: 20)",
    )
    parser.add_argument(
        "--focus", choices=["fe"], default=None,
        help="Filter: 'fe' = only cases with frontend errors",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not write .md report file (only stdout)",
    )
    args = parser.parse_args()

    # Collect result paths
    result_paths: List[Path] = []
    if args.results:
        result_paths.extend(args.results)
    if args.dir:
        for p in sorted(args.dir.glob("**/results.json")):
            result_paths.append(p)
    if not result_paths:
        parser.error("Provide --results and/or --dir")

    report = build_report(
        result_paths,
        compare_path=args.compare,
        top_n=args.top_n,
        focus=args.focus,
    )

    # Print to stdout
    print(report)

    # Save .md file next to the first results file
    if not args.no_save and result_paths:
        out_path = result_paths[0].parent / "per_stage_scoring_report.md"
        out_path.write_text(report, encoding="utf-8")
        print(f"\n> Report saved to: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
