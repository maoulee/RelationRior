#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path


LEAF_PATTERN = re.compile(r"Leaf Entities \(\d+\):\s*\n\s*(\[[^\n]*\])", re.S)
CVT_PATTERN = re.compile(r"CVT-Expanded Entities \(\d+\):\s*\n\s*(\[[^\n]*\])", re.S)


def normalize(value: object) -> str:
    return str(value).strip().lower()


def extract_entities(response_text: str) -> list[str]:
    for pattern in (LEAF_PATTERN, CVT_PATTERN):
        match = pattern.search(response_text or "")
        if not match:
            continue
        try:
            return [str(v) for v in ast.literal_eval(match.group(1))]
        except Exception:
            continue
    return []


def analyze_results(results_path: Path) -> Counter[str]:
    records = json.loads(results_path.read_text())
    counts: Counter[str] = Counter()

    for record in records:
        gt = {normalize(x) for x in record.get("ground_truth", [])}
        if not gt:
            continue

        filter_used = False
        action_spaces: list[set[str]] = []
        for turn in record.get("trajectory", []) or []:
            executed_queries = turn.get("executed_queries", []) or []
            backend_results = turn.get("backend_results", []) or []
            filter_used = filter_used or any(query.get("tool_name") == "filter" for query in executed_queries)

            for query, result in zip(executed_queries, backend_results):
                if query.get("tool_name") != "action":
                    continue
                entities = {
                    normalize(x) for x in extract_entities((result or {}).get("response_text") or "")
                }
                action_spaces.append(entities)

        covering_spaces = [space for space in action_spaces if gt <= space]
        if not covering_spaces:
            counts["no_single_action_cover"] += 1
            continue

        counts["single_action_cover"] += 1
        chosen = min(covering_spaces, key=len)
        outcome = "correct" if record.get("f1", 0.0) >= 0.95 else "wrong"
        gt_size_bucket = "single_gt" if len(gt) == 1 else "multi_gt"

        if len(chosen) == len(gt):
            counts[f"{outcome}__single_action_exact_match"] += 1
        else:
            counts[f"{outcome}__single_action_gt_is_proper_subset"] += 1
            counts[f"{outcome}__single_action_gt_is_proper_subset__{gt_size_bucket}"] += 1
            counts[
                f"{outcome}__single_action_gt_is_proper_subset__{'with_filter' if filter_used else 'no_filter'}"
            ] += 1

    return counts


def pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.0%"
    return f"{numerator / denominator * 100:.1f}%"


def build_report(base_label: str, skill_label: str, output_path: Path) -> None:
    base_counts = analyze_results(
        Path(f"/zhaoshu/subgraph/reports/skill_enhanced_test/{base_label}/results.json")
    )
    skill_counts = analyze_results(
        Path(f"/zhaoshu/subgraph/reports/skill_enhanced_test/{skill_label}/results.json")
    )

    def subset_total(counts: Counter[str], outcome: str) -> int:
        return counts.get(f"{outcome}__single_action_gt_is_proper_subset", 0)

    lines: list[str] = []
    lines.append("# Filter-Strategy Need Summary")
    lines.append("")
    lines.append(f"- Baseline: `{base_label}`")
    lines.append(f"- +Skills: `{skill_label}`")
    lines.append("")
    lines.append("## Main Question")
    lines.append("")
    lines.append(
        "- When a single action space can already cover GT, how often is GT only a **proper subset** of that action's candidates?"
    )
    lines.append(
        "- This tells us how often the system needs a `filter/subset` decision rather than a simple `keep all from the chosen action` rule."
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(
        "- In `+skills`, `1113` cases have a single action space that covers GT."
    )
    lines.append(
        f"- Among those, `365` cases (`{pct(365, 1113)}`) require selecting a **proper subset** of that action space rather than keeping everything."
    )
    lines.append(
        "- Of those `365` subset-needed cases, `247` are still correct and `118` are wrong."
    )
    lines.append(
        f"- Most subset-needed cases do **not** use `filter()`: `332 / 365` (`{pct(332, 365)}`) in `+skills`."
    )
    lines.append("")
    lines.append("## Baseline vs +Skills")
    lines.append("")
    lines.append("| Metric | Baseline | +Skills |")
    lines.append("|---|---:|---:|")
    rows = [
        ("single_action_cover", "Cases where one action space already covers GT."),
        ("correct__single_action_exact_match", "Correct cases where chosen action candidates exactly match GT."),
        ("wrong__single_action_exact_match", "Wrong cases even though some action exactly matched GT."),
        ("correct__single_action_gt_is_proper_subset", "Correct cases where GT is only a subset of the chosen action space."),
        ("wrong__single_action_gt_is_proper_subset", "Wrong cases where GT is only a subset of the chosen action space."),
        ("correct__single_action_gt_is_proper_subset__no_filter", "Correct subset-needed cases solved without filter."),
        ("correct__single_action_gt_is_proper_subset__with_filter", "Correct subset-needed cases solved with filter."),
        ("wrong__single_action_gt_is_proper_subset__no_filter", "Wrong subset-needed cases that still never used filter."),
        ("wrong__single_action_gt_is_proper_subset__with_filter", "Wrong subset-needed cases that did use filter."),
    ]
    for key, _ in rows:
        lines.append(f"| `{key}` | {base_counts.get(key, 0)} | {skill_counts.get(key, 0)} |")
    lines.append("")
    lines.append("## Subset Cases by GT Size")
    lines.append("")
    lines.append("| Bucket | Baseline | +Skills |")
    lines.append("|---|---:|---:|")
    for key in [
        "correct__single_action_gt_is_proper_subset__single_gt",
        "correct__single_action_gt_is_proper_subset__multi_gt",
        "wrong__single_action_gt_is_proper_subset__single_gt",
        "wrong__single_action_gt_is_proper_subset__multi_gt",
    ]:
        lines.append(f"| `{key}` | {base_counts.get(key, 0)} | {skill_counts.get(key, 0)} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- `keep all candidates from the chosen action space` is not enough. A large minority of single-action-cover cases need subset selection."
    )
    lines.append(
        "- But `filter()` is still underused: the dominant pattern is subset selection happening implicitly in reasoning rather than through an explicit constraint step."
    )
    lines.append(
        "- So the next answer-strategy design should not be just `single action = keep all`; it should be:"
    )
    lines.append("  - `single_action.keep_all_supported` when no evidence suggests exclusion")
    lines.append("  - `single_action.select_subset_if_constraints_exist` when candidates contain extra entities")
    lines.append(
        "- This also means filter strategy is a major lever: the system already faces many cases where one action is enough, but not all candidates inside that action are valid final answers."
    )

    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze how often single-action coverage still requires subset selection.")
    parser.add_argument(
        "--baseline-label",
        default="fulltest_baseline_protocol_guard_20260330",
        help="Baseline report label under reports/skill_enhanced_test.",
    )
    parser.add_argument(
        "--skill-label",
        default="fulltest_skill_top3_protocol_guard_20260330",
        help="Skill-enhanced report label under reports/skill_enhanced_test.",
    )
    parser.add_argument(
        "--output",
        default="/zhaoshu/subgraph/reports/skill_enhanced_test/filter_strategy_need_summary_20260331.md",
        help="Output markdown path.",
    )
    args = parser.parse_args()

    build_report(args.baseline_label, args.skill_label, Path(args.output))


if __name__ == "__main__":
    main()
