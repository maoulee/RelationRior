#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests


LEAF_PATTERN = re.compile(r"Leaf Entities \(\d+\):\s*\n\s*(\[[^\n]*\])", re.S)
CVT_PATTERN = re.compile(r"CVT-Expanded Entities \(\d+\):\s*\n\s*(\[[^\n]*\])", re.S)
SUGGESTED_RELATION_PATTERN = re.compile(
    r"\[Suggested Filter Relations\]:\s*\n(?:The following relations.*\n)?((?:\s*-\s*[^\n]+\n)+)",
    re.S,
)


def normalize(value: object) -> str:
    return str(value).strip().lower()


def extract_entities(response_text: str) -> list[str]:
    for pattern in (LEAF_PATTERN, CVT_PATTERN):
        match = pattern.search(response_text or "")
        if not match:
            continue
        try:
            values = ast.literal_eval(match.group(1))
            return [str(v) for v in values]
        except Exception:
            continue
    return []


def extract_suggested_relations(response_text: str) -> list[str]:
    match = SUGGESTED_RELATION_PATTERN.search(response_text or "")
    if not match:
        return []
    relations: list[str] = []
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        relation = line[2:].strip()
        if relation:
            relations.append(relation)
    return relations


def classify_filter_outcome(
    input_candidates: list[str],
    filtered_candidates: list[str],
    ground_truth: set[str],
) -> str:
    input_norm = {normalize(x) for x in input_candidates}
    filtered_norm = {normalize(x) for x in filtered_candidates}

    if not filtered_norm:
        return "empty_result"
    if filtered_norm == input_norm:
        return "all_pass_no_prune"
    if ground_truth and ground_truth <= input_norm:
        if ground_truth <= filtered_norm:
            if filtered_norm == ground_truth:
                return "pruned_exactly_to_gt"
            return "pruned_but_gt_kept_with_distractors"
        return "pruned_and_dropped_gt"
    if ground_truth & input_norm:
        if ground_truth & filtered_norm:
            return "partial_gt_kept"
        return "partial_gt_dropped"
    return "strict_subset_no_gt_overlap"


def analyze_results(results_path: Path, kg_api_url: str) -> tuple[Counter[str], dict[str, list[dict[str, Any]]]]:
    records = json.loads(results_path.read_text())
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    endpoint = f"{kg_api_url.rstrip('/')}/v2/filter"

    for record in records:
        case_id = record["case_id"]
        ground_truth = {normalize(x) for x in record.get("ground_truth", [])}
        for turn in record.get("trajectory", []) or []:
            for query, backend_result in zip(
                turn.get("executed_queries", []) or [],
                turn.get("backend_results", []) or [],
            ):
                if query.get("tool_name") != "action":
                    continue

                response_text = (backend_result or {}).get("response_text") or ""
                if "[Suggested Filter Relations]" not in response_text:
                    continue

                counts["action_blocks_with_suggestions"] += 1
                candidates = extract_entities(response_text)
                relations = extract_suggested_relations(response_text)

                if len(candidates) < 2:
                    counts["skipped_not_enough_candidates"] += 1
                    continue
                if not relations:
                    counts["skipped_no_relations_parsed"] += 1
                    continue

                counts["parseable_blocks"] += 1
                input_norm = {normalize(x) for x in candidates}
                if ground_truth <= input_norm and ground_truth:
                    counts["gt_fully_in_action_candidates"] += 1
                elif ground_truth & input_norm:
                    counts["gt_partially_in_action_candidates"] += 1
                else:
                    counts["gt_not_in_action_candidates"] += 1

                payload = {
                    "sample_id": case_id,
                    "candidates": candidates,
                    "constraint_relations": relations,
                    "constraint_entities": [],
                }
                try:
                    response = requests.post(endpoint, json=payload, timeout=60)
                    response.raise_for_status()
                    data = response.json()
                except Exception as exc:
                    counts["request_error"] += 1
                    if len(examples["request_error"]) < 5:
                        examples["request_error"].append(
                            {
                                "case_id": case_id,
                                "candidates": candidates,
                                "relations": relations,
                                "error": str(exc),
                            }
                        )
                    continue

                filtered = [str(x) for x in data.get("found_end_entities", []) or []]
                bucket = classify_filter_outcome(candidates, filtered, ground_truth)
                counts[bucket] += 1

                if len(examples[bucket]) < 5:
                    examples[bucket].append(
                        {
                            "case_id": case_id,
                            "question": record.get("question", ""),
                            "candidates": candidates,
                            "ground_truth": record.get("ground_truth", []),
                            "relations": relations,
                            "filtered": filtered,
                            "response_excerpt": (data.get("response_text") or "")[:1200],
                        }
                    )

    return counts, examples


def pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.0%"
    return f"{numerator / denominator * 100:.1f}%"


def build_report(results_label: str, kg_api_url: str, output_path: Path) -> None:
    results_path = Path(f"/zhaoshu/subgraph/reports/skill_enhanced_test/{results_label}/results.json")
    counts, examples = analyze_results(results_path, kg_api_url)

    parseable = counts.get("parseable_blocks", 0)
    gt_ready = counts.get("gt_fully_in_action_candidates", 0)

    lines: list[str] = []
    lines.append("# Filter Effectiveness Check")
    lines.append("")
    lines.append(f"- Results label: `{results_label}`")
    lines.append(f"- Backend endpoint: `{kg_api_url.rstrip('/')}/v2/filter`")
    lines.append("")
    lines.append("## Main Findings")
    lines.append("")
    lines.append(
        f"- Action blocks with `[Suggested Filter Relations]`: `{counts.get('action_blocks_with_suggestions', 0)}`"
    )
    lines.append(f"- Parseable action blocks (relations + >=2 candidates): `{parseable}`")
    lines.append(
        f"- Among parseable blocks, GT is fully inside that action candidate set in `{gt_ready}` cases (`{pct(gt_ready, parseable)}`)."
    )
    lines.append("")
    lines.append(
        "- Directly calling `filter()` with **only** the backend-suggested relations usually does **not** finish the disambiguation by itself."
    )
    lines.append(
        "- Reason: the current filter API checks whether a relation exists and shows its values, but it does not know **which value** the question wants. So suggested relations are often better as an evidence-inspection tool than as a hard selector."
    )
    lines.append("")
    lines.append("## Outcome of Direct `filter(constraint_relations=suggested_relations)` Calls")
    lines.append("")
    lines.append("| Bucket | Count | Share of Parseable |")
    lines.append("|---|---:|---:|")
    for key in [
        "all_pass_no_prune",
        "pruned_exactly_to_gt",
        "pruned_but_gt_kept_with_distractors",
        "pruned_and_dropped_gt",
        "partial_gt_kept",
        "partial_gt_dropped",
        "strict_subset_no_gt_overlap",
        "empty_result",
    ]:
        value = counts.get(key, 0)
        lines.append(f"| `{key}` | {value} | {pct(value, parseable)} |")
    lines.append("")
    lines.append("## GT Coverage Before Filter")
    lines.append("")
    lines.append("| Bucket | Count | Share of Parseable |")
    lines.append("|---|---:|---:|")
    for key in [
        "gt_fully_in_action_candidates",
        "gt_partially_in_action_candidates",
        "gt_not_in_action_candidates",
    ]:
        value = counts.get(key, 0)
        lines.append(f"| `{key}` | {value} | {pct(value, parseable)} |")
    lines.append("")
    lines.append("## Representative Examples")
    lines.append("")
    for bucket in [
        "all_pass_no_prune",
        "pruned_exactly_to_gt",
        "pruned_but_gt_kept_with_distractors",
        "pruned_and_dropped_gt",
        "empty_result",
    ]:
        items = examples.get(bucket, [])
        if not items:
            continue
        lines.append(f"### `{bucket}`")
        lines.append("")
        for item in items:
            lines.append(
                f"- `{item['case_id']}` question: `{item['question']}` | candidates: `{item['candidates']}` | GT: `{item['ground_truth']}` | suggested relations: `{item['relations']}` | filtered: `{item['filtered']}`"
            )
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- If `all_pass_no_prune` dominates, then the backend suggestion is still useful, but mainly because it exposes discriminative **values** for the model to inspect, not because it automatically selects the right candidate."
    )
    lines.append(
        "- So a good Stage 4/5 rule is: backend-suggested relations should trigger a `filter()` evidence check, but the model still needs to compare the returned values against the question before collapsing candidates."
    )

    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether backend-suggested filter relations directly distinguish candidates.")
    parser.add_argument(
        "--results-label",
        default="fulltest_skill_top3_protocol_guard_20260330",
        help="Report label under reports/skill_enhanced_test.",
    )
    parser.add_argument(
        "--kg-api-url",
        default="http://127.0.0.1:8001",
        help="KG backend base URL.",
    )
    parser.add_argument(
        "--output",
        default="/zhaoshu/subgraph/reports/skill_enhanced_test/filter_effectiveness_20260331.md",
        help="Markdown output path.",
    )
    args = parser.parse_args()

    build_report(args.results_label, args.kg_api_url, Path(args.output))


if __name__ == "__main__":
    main()
