#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


LEAF_PATTERN = re.compile(r"Leaf Entities \(\d+\):\s*\n\s*(\[[^\n]*\])", re.S)
CVT_PATTERN = re.compile(r"CVT-Expanded Entities \(\d+\):\s*\n\s*(\[[^\n]*\])", re.S)


def normalize(value: object) -> str:
    return str(value).strip().lower()


def extract_entities(response_text: str) -> tuple[list[str], str]:
    leaf_match = LEAF_PATTERN.search(response_text or "")
    if leaf_match:
        try:
            return [str(v) for v in ast.literal_eval(leaf_match.group(1))], "leaf"
        except Exception:
            pass

    cvt_match = CVT_PATTERN.search(response_text or "")
    if cvt_match:
        try:
            return [str(v) for v in ast.literal_eval(cvt_match.group(1))], "cvt"
        except Exception:
            pass

    return [], "none"


def load_skill(skill_id: str, cache: dict[str, dict]) -> dict:
    if skill_id not in cache:
        path = Path(f"/zhaoshu/subgraph/skills/webqsp_train_case_skills_en/{skill_id}.json")
        cache[skill_id] = json.loads(path.read_text())
    return cache[skill_id]


def answer_count_bucket(value: str | None) -> str:
    return "single" if value == "single" else "multi"


def analyze_case(record: dict, skill_cache: dict[str, dict]) -> dict:
    gt = {normalize(x) for x in record.get("ground_truth", [])}
    pred = {normalize(x) for x in record.get("predicted", [])}
    filter_used = False
    action_spaces: list[dict] = []

    for turn in record.get("trajectory", []) or []:
        executed_queries = turn.get("executed_queries", []) or []
        backend_results = turn.get("backend_results", []) or []
        filter_used = filter_used or any(query.get("tool_name") == "filter" for query in executed_queries)

        for query, result in zip(executed_queries, backend_results):
            if query.get("tool_name") != "action":
                continue

            response_text = (result or {}).get("response_text") or ""
            entities, entity_source = extract_entities(response_text)
            action_spaces.append(
                {
                    "path": tuple(
                        step.get("relation", "")
                        for step in query.get("arguments", {}).get("path", []) or []
                    ),
                    "entities": {normalize(x) for x in entities},
                    "entity_count": len(entities),
                    "entity_source": entity_source,
                    "suggested_filter": "[Suggested Filter Relations]" in response_text,
                }
            )

    gt_action_indices = [idx for idx, space in enumerate(action_spaces) if gt & space["entities"]]
    pred_action_indices = [idx for idx, space in enumerate(action_spaces) if pred & space["entities"]]
    gt_single_cover = any(gt <= space["entities"] for space in action_spaces) if gt else False
    pred_single_cover = any(pred <= space["entities"] for space in action_spaces) if pred else False

    skills = [
        load_skill(skill_id, skill_cache)
        for skill_id in record.get("skill_bundle", {}).get("selected_case_ids", [])
    ]
    answer_count_votes = Counter(
        answer_count_bucket(skill.get("answer_strategy", {}).get("answer_count")) for skill in skills
    )
    majority_count = answer_count_votes.most_common(1)[0][0] if answer_count_votes else "single"
    expected_count = "single" if len(record.get("ground_truth", [])) <= 1 else "multi"

    return {
        "case_id": record["case_id"],
        "question": record.get("question", ""),
        "ground_truth": record.get("ground_truth", []),
        "predicted": record.get("predicted", []),
        "action_count": len(action_spaces),
        "gt_action_indices": gt_action_indices,
        "pred_action_indices": pred_action_indices,
        "gt_single_cover": gt_single_cover,
        "pred_single_cover": pred_single_cover,
        "filter_used": filter_used,
        "answer_count_votes": answer_count_votes,
        "majority_count": majority_count,
        "expected_count": expected_count,
        "count_mismatch": majority_count != expected_count,
        "single_gt_multicandidate_no_filter": (
            len(gt) == 1
            and len(gt_action_indices) == 1
            and action_spaces[gt_action_indices[0]]["entity_count"] > 1
            and not filter_used
        ),
        "single_gt_multicandidate_with_filter": (
            len(gt) == 1
            and len(gt_action_indices) == 1
            and action_spaces[gt_action_indices[0]]["entity_count"] > 1
            and filter_used
        ),
        "single_gt_singlecandidate": (
            len(gt) == 1
            and len(gt_action_indices) == 1
            and action_spaces[gt_action_indices[0]]["entity_count"] <= 1
        ),
        "multi_gt_all_in_one_action": len(gt) > 1 and gt_single_cover,
        "multi_gt_not_all_in_one_action": len(gt) > 1 and not gt_single_cover,
        "single_gt_in_multiple_actions": len(gt) == 1 and len(gt_action_indices) > 1,
        "single_gt_missing_from_actions": len(gt) == 1 and len(gt_action_indices) == 0,
    }


def pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.0%"
    return f"{numerator / denominator * 100:.1f}%"


def build_report(skill_label: str, output_path: Path) -> None:
    results_path = Path(f"/zhaoshu/subgraph/reports/skill_enhanced_test/{skill_label}/results.json")
    records = json.loads(results_path.read_text())
    correct_records = [record for record in records if record.get("f1", 0.0) >= 0.95]

    skill_cache: dict[str, dict] = {}
    summaries = [analyze_case(record, skill_cache) for record in correct_records]

    counts = Counter()
    examples: defaultdict[str, list[str]] = defaultdict(list)
    for summary in summaries:
        for key in [
            "multi_gt_all_in_one_action",
            "multi_gt_not_all_in_one_action",
            "single_gt_multicandidate_no_filter",
            "single_gt_multicandidate_with_filter",
            "single_gt_singlecandidate",
            "single_gt_in_multiple_actions",
            "single_gt_missing_from_actions",
            "pred_single_cover",
            "count_mismatch",
        ]:
            if summary.get(key):
                counts[key] += 1
                if len(examples[key]) < 5:
                    examples[key].append(summary["case_id"])

    lines: list[str] = []
    lines.append("# Correct-Case Action-Space Strategy")
    lines.append("")
    lines.append(f"- Source: `{skill_label}`")
    lines.append("")
    lines.append("## Main Findings")
    lines.append("")
    lines.append(
        f"- Correct cases total: `{len(summaries)}`."
    )
    lines.append(
        f"- In correct cases, final answers are usually coverable by a single action space: `{counts['pred_single_cover']}` / `{len(summaries)}` (`{pct(counts['pred_single_cover'], len(summaries))}`)."
    )
    lines.append(
        f"- For correct multi-answer cases, the dominant pattern is still `all GT already in one action`: `{counts['multi_gt_all_in_one_action']}` vs `{counts['multi_gt_not_all_in_one_action']}` not fully covered by one action."
    )
    lines.append(
        f"- For correct single-answer cases, `one action + multiple candidates + no filter` is common: `{counts['single_gt_multicandidate_no_filter']}` cases."
    )
    lines.append(
        f"- By contrast, `one action + multiple candidates + with filter` is much smaller: `{counts['single_gt_multicandidate_with_filter']}` cases."
    )
    lines.append(
        f"- Even among correct cases, loaded-skill `answer_count` still mismatches GT cardinality in `{counts['count_mismatch']}` / `{len(summaries)}` (`{pct(counts['count_mismatch'], len(summaries))}`)."
    )
    lines.append("")
    lines.append("## What This Suggests")
    lines.append("")
    lines.append(
        "- The more stable abstraction is not `single vs multiple answers` by itself, but `how many action spaces should contribute to the final answer`."
    )
    lines.append(
        "- In correct runs, the dominant healthy pattern is: **final answer comes from one action space**; then within that action space, keep all graph-supported answers unless there is clear evidence to filter them."
    )
    lines.append(
        "- Cross-action merging should remain a minority behavior, because it is uncommon even among successful trajectories."
    )
    lines.append("")
    lines.append("## Correct-Case Breakdown")
    lines.append("")
    lines.append("| Pattern | Count | Share | Interpretation |")
    lines.append("|---|---:|---:|---|")
    rows = [
        (
            "pred_single_cover",
            "Final correct answer set is fully coverable by one action space.",
        ),
        (
            "multi_gt_all_in_one_action",
            "Multi-answer GT already sits inside one action space.",
        ),
        (
            "multi_gt_not_all_in_one_action",
            "Multi-answer GT is not fully coverable by one action space.",
        ),
        (
            "single_gt_multicandidate_no_filter",
            "Single-answer GT sits in one multi-candidate action space, and the model still solved it without filter.",
        ),
        (
            "single_gt_multicandidate_with_filter",
            "Single-answer GT sits in one multi-candidate action space, and filter was used.",
        ),
        (
            "single_gt_singlecandidate",
            "Single-answer GT sits in one action space with one extracted candidate.",
        ),
        (
            "single_gt_in_multiple_actions",
            "Single-answer GT appears in multiple action spaces.",
        ),
        (
            "single_gt_missing_from_actions",
            "Correct answer is not visible in extracted action-space entities under this parser.",
        ),
        (
            "count_mismatch",
            "Loaded skill bundle majority `answer_count` disagrees with GT cardinality, yet the case is still correct.",
        ),
    ]
    for key, interpretation in rows:
        lines.append(
            f"| `{key}` | {counts.get(key, 0)} | {pct(counts.get(key, 0), len(summaries))} | {interpretation} |"
        )
    lines.append("")
    lines.append("## Representative Examples")
    lines.append("")
    for key in [
        "multi_gt_all_in_one_action",
        "single_gt_multicandidate_no_filter",
        "single_gt_multicandidate_with_filter",
        "count_mismatch",
    ]:
        labels = ", ".join(f"`{case_id}`" for case_id in examples.get(key, [])) or "-"
        lines.append(f"- `{key}`: {labels}")
    lines.append("")
    lines.append("## Interpretation for Answer Strategy")
    lines.append("")
    lines.append(
        "- A better answer-strategy unit is: `single action space: keep all supported answers` vs `single action space: choose a filtered subset`."
    )
    lines.append(
        "- This is more faithful to successful trajectories than using `single/multiple` as a global answer-count prior."
    )
    lines.append(
        "- In particular, `answer_count` is still mismatched in many correct runs, which means cardinality alone is too coarse to be the main strategy variable."
    )

    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze correct cases at action-space granularity.")
    parser.add_argument(
        "--skill-label",
        default="fulltest_skill_top3_protocol_guard_20260330",
        help="Skill-enhanced report label under reports/skill_enhanced_test.",
    )
    parser.add_argument(
        "--output",
        default="/zhaoshu/subgraph/reports/skill_enhanced_test/correct_case_action_space_strategy_20260331.md",
        help="Output markdown path.",
    )
    args = parser.parse_args()

    build_report(args.skill_label, Path(args.output))


if __name__ == "__main__":
    main()
