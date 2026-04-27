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


def extract_action_entities(response_text: str) -> tuple[list[str], str]:
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


def collect_action_spaces(record: dict) -> tuple[list[dict], bool]:
    action_spaces: list[dict] = []
    filter_used = False

    for turn in record.get("trajectory", []) or []:
        executed_queries = turn.get("executed_queries", []) or []
        backend_results = turn.get("backend_results", []) or []

        filter_used = filter_used or any(
            query.get("tool_name") == "filter" for query in executed_queries
        )

        for query, result in zip(executed_queries, backend_results):
            if query.get("tool_name") != "action":
                continue

            response_text = (result or {}).get("response_text") or ""
            entities, entity_source = extract_action_entities(response_text)
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

    return action_spaces, filter_used


def classify_record(record: dict) -> dict:
    gt = {normalize(x) for x in record.get("ground_truth", [])}
    pred = {normalize(x) for x in record.get("predicted", [])}
    action_spaces, filter_used = collect_action_spaces(record)

    action_union = set().union(*(space["entities"] for space in action_spaces)) if action_spaces else set()
    gt_action_indices = [idx for idx, space in enumerate(action_spaces) if gt & space["entities"]]
    pred_action_indices = [idx for idx, space in enumerate(action_spaces) if pred & space["entities"]]
    single_action_covers_all_gt = any(gt <= space["entities"] for space in action_spaces) if gt else False
    all_gt_covered_by_union = gt <= action_union if gt else False

    bucket = None

    if len(gt) > 1 and len(pred) <= 1:
        if single_action_covers_all_gt:
            bucket = "multi_gt_collapsed__all_gt_in_one_action"
        elif all_gt_covered_by_union:
            bucket = "multi_gt_collapsed__gt_only_coverable_across_actions"
        else:
            bucket = "multi_gt_collapsed__gt_not_fully_reached"
    elif len(gt) == 1:
        if len(gt_action_indices) == 1:
            gt_space = action_spaces[gt_action_indices[0]]
            if gt_space["entity_count"] > 1:
                bucket = (
                    "single_gt__one_action_multicandidate__with_filter"
                    if filter_used
                    else "single_gt__one_action_multicandidate__no_filter"
                )
            else:
                bucket = "single_gt__one_action_singlecandidate"
        elif len(gt_action_indices) > 1:
            bucket = "single_gt__multiple_actions_contain_gt"
        else:
            bucket = "single_gt__gt_missing_from_actions"

    return {
        "bucket": bucket,
        "filter_used": filter_used,
        "action_count": len(action_spaces),
        "gt_action_indices": gt_action_indices,
        "pred_action_indices": pred_action_indices,
        "single_action_covers_all_gt": single_action_covers_all_gt,
        "all_gt_covered_by_union": all_gt_covered_by_union,
        "gt_in_many_actions_only_one_has_gt": (
            len(gt) == 1 and len(action_spaces) > 1 and len(gt_action_indices) == 1
        ),
        "gt_single_action_suggested_filter": (
            len(gt) == 1
            and
            len(gt_action_indices) == 1
            and action_spaces[gt_action_indices[0]]["entity_count"] > 1
            and not filter_used
            and action_spaces[gt_action_indices[0]]["suggested_filter"]
        ),
    }


def analyze_results(results_path: Path) -> dict:
    records = json.loads(results_path.read_text())
    wrong_records = [record for record in records if record.get("f1", 0.0) < 0.95]

    bucket_counts: Counter[str] = Counter()
    extras: Counter[str] = Counter()
    examples: defaultdict[str, list[str]] = defaultdict(list)

    for record in wrong_records:
        summary = classify_record(record)
        bucket = summary["bucket"]
        if bucket:
            bucket_counts[bucket] += 1
            if len(examples[bucket]) < 5:
                examples[bucket].append(record["case_id"])

        if summary["gt_in_many_actions_only_one_has_gt"]:
            extras["single_gt__many_actions_only_one_has_gt"] += 1
            if len(examples["single_gt__many_actions_only_one_has_gt"]) < 5:
                examples["single_gt__many_actions_only_one_has_gt"].append(record["case_id"])

        if summary["gt_single_action_suggested_filter"]:
            extras["single_gt__one_action_multicandidate__suggested_filter_present"] += 1
            if len(examples["single_gt__one_action_multicandidate__suggested_filter_present"]) < 5:
                examples["single_gt__one_action_multicandidate__suggested_filter_present"].append(
                    record["case_id"]
                )

    return {
        "wrong_total": len(wrong_records),
        "bucket_counts": bucket_counts,
        "extras": extras,
        "examples": examples,
    }


def pct(count: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{count / total * 100:.1f}%"


def build_report(base_label: str, skill_label: str, output_path: Path) -> None:
    base = analyze_results(
        Path(f"/zhaoshu/subgraph/reports/skill_enhanced_test/{base_label}/results.json")
    )
    skill = analyze_results(
        Path(f"/zhaoshu/subgraph/reports/skill_enhanced_test/{skill_label}/results.json")
    )

    lines: list[str] = []
    lines.append("# Action-Space Failure Analysis")
    lines.append("")
    lines.append(f"- Baseline: `{base_label}`")
    lines.append(f"- +Skills: `{skill_label}`")
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append(
        "- When a case is wrong, is the GT already concentrated in one action space, or is it scattered across multiple action spaces?"
    )
    lines.append(
        "- For `single GT + one action + multiple candidates`, did the agent actually invoke `filter()`?"
    )
    lines.append("")
    lines.append("## Core Takeaways")
    lines.append("")
    lines.append(
        f"- The largest wrong-case bucket in `+skills` is still `multiple GT collapsed to one answer while all GT already live in a single action space`: `{skill['bucket_counts'].get('multi_gt_collapsed__all_gt_in_one_action', 0)}` / `{skill['wrong_total']}` (`{pct(skill['bucket_counts'].get('multi_gt_collapsed__all_gt_in_one_action', 0), skill['wrong_total'])}`)."
    )
    lines.append(
        "- True `GT only coverable across multiple actions` cases are almost nonexistent in this run. This supports a conservative rule: joining answers across different action spaces should be rare and heavily gated."
    )
    lines.append(
        f"- `single GT + one action + multiple candidates + no filter` remains a visible bucket in `+skills`: `{skill['bucket_counts'].get('single_gt__one_action_multicandidate__no_filter', 0)}` cases."
    )
    lines.append(
        f"- Among those `+skills` cases, `{skill['extras'].get('single_gt__one_action_multicandidate__suggested_filter_present', 0)}` already had backend-suggested filter relations, but the model still did not call `filter()`."
    )
    lines.append(
        f"- Pure action-space miss is still real, but smaller: `single GT never reached any action` is `{skill['bucket_counts'].get('single_gt__gt_missing_from_actions', 0)}` cases."
    )
    lines.append("")
    lines.append("## Multiple-GT Cases That Were Collapsed to One Prediction")
    lines.append("")
    lines.append("| Bucket | Baseline | +Skills | Interpretation |")
    lines.append("|---|---:|---:|---|")
    multi_rows = [
        (
            "multi_gt_collapsed__all_gt_in_one_action",
            "All GT answers were already available inside one action space. This is a reasoning/selection problem, not an action-merging problem.",
        ),
        (
            "multi_gt_collapsed__gt_only_coverable_across_actions",
            "GT is only fully recoverable by combining multiple action spaces.",
        ),
        (
            "multi_gt_collapsed__gt_not_fully_reached",
            "The explored action spaces never fully covered GT, so this is still upstream/action-related.",
        ),
    ]
    for key, interp in multi_rows:
        lines.append(
            f"| `{key}` | {base['bucket_counts'].get(key, 0)} | {skill['bucket_counts'].get(key, 0)} | {interp} |"
        )
    lines.append("")
    lines.append("## Single-GT Cases")
    lines.append("")
    lines.append("| Bucket | Baseline | +Skills | Interpretation |")
    lines.append("|---|---:|---:|---|")
    single_rows = [
        (
            "single_gt__one_action_multicandidate__no_filter",
            "GT exists in exactly one action space, but that space has multiple candidates and no filter was used.",
        ),
        (
            "single_gt__one_action_multicandidate__with_filter",
            "Filter was used, but the final answer was still wrong.",
        ),
        (
            "single_gt__one_action_singlecandidate",
            "GT exists in one action space with only one extracted candidate; downstream readout is likely the issue.",
        ),
        (
            "single_gt__multiple_actions_contain_gt",
            "GT appears in multiple action spaces, so action-space ambiguity is real.",
        ),
        (
            "single_gt__gt_missing_from_actions",
            "GT never entered any explored action space.",
        ),
    ]
    for key, interp in single_rows:
        lines.append(
            f"| `{key}` | {base['bucket_counts'].get(key, 0)} | {skill['bucket_counts'].get(key, 0)} | {interp} |"
        )
    lines.append("")
    lines.append("## Extra Signals")
    lines.append("")
    lines.append("| Signal | Baseline | +Skills |")
    lines.append("|---|---:|---:|")
    for key in [
        "single_gt__many_actions_only_one_has_gt",
        "single_gt__one_action_multicandidate__suggested_filter_present",
    ]:
        lines.append(
            f"| `{key}` | {base['extras'].get(key, 0)} | {skill['extras'].get(key, 0)} |"
        )
    lines.append("")
    lines.append("## Representative Examples")
    lines.append("")
    rep_keys = [
        "multi_gt_collapsed__all_gt_in_one_action",
        "single_gt__one_action_multicandidate__no_filter",
        "single_gt__gt_missing_from_actions",
        "single_gt__many_actions_only_one_has_gt",
    ]
    for key in rep_keys:
        base_examples = ", ".join(f"`{x}`" for x in base["examples"].get(key, [])) or "-"
        skill_examples = ", ".join(f"`{x}`" for x in skill["examples"].get(key, [])) or "-"
        lines.append(f"- `{key}`")
        lines.append(f"  Baseline: {base_examples}")
        lines.append(f"  +Skills: {skill_examples}")
    lines.append("")
    lines.append("## Interpretation for Rule Design")
    lines.append("")
    lines.append(
        "- If multiple GT answers are all supported by the same action space, the default should be **keep them together unless graph evidence explicitly filters them**."
    )
    lines.append(
        "- If answers would need to be combined across different action spaces, do **not** merge by default. Treat this as a much rarer, higher-risk case."
    )
    lines.append(
        "- For `single GT + one action + multiple candidates`, add a stricter Stage-4/5 rule: if backend already exposed a suggested filter relation, the model should try `filter()` before collapsing by intuition."
    )
    lines.append(
        "- Cases where GT never entered any action space remain true relation/action failures; those should be handled by better relation priors or action selection, not by final-stage reasoning rules."
    )

    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze answer/action-space coupling in wrong cases.")
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
        default="/zhaoshu/subgraph/reports/skill_enhanced_test/action_space_patterns_20260331.md",
        help="Output markdown path.",
    )
    args = parser.parse_args()

    build_report(args.baseline_label, args.skill_label, Path(args.output))


if __name__ == "__main__":
    main()
