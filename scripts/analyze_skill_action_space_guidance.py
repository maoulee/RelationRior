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

SUBSET_KEYWORDS = (
    "specific",
    "explicitly",
    "specified",
    "matching",
    "narrow",
    "filter",
    "constraint",
    "current",
    "latest",
    "most recent",
    "named actor",
    "named person",
)
KEEP_ALL_KEYWORDS = (
    "all valid",
    "list all",
    "all matching",
    "all applicable",
    "all supported",
    "all graph-supported",
    "all members",
    "all languages",
)


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


def load_skill(skill_id: str, cache: dict[str, dict]) -> dict:
    if skill_id not in cache:
        path = Path(f"/zhaoshu/subgraph/skills/webqsp_train_case_skills_en/{skill_id}.json")
        cache[skill_id] = json.loads(path.read_text())
    return cache[skill_id]


def answer_count_bucket(value: str | None) -> str:
    return "single" if value == "single" else "multi"


def skill_rule_text(skill: dict) -> str:
    strategy = skill.get("answer_strategy", {}) or {}
    return " ".join(str(v) for v in strategy.values()).lower()


def has_subset_cue(skill: dict) -> bool:
    text = skill_rule_text(skill)
    return any(keyword in text for keyword in SUBSET_KEYWORDS)


def has_keep_all_cue(skill: dict) -> bool:
    text = skill_rule_text(skill)
    return any(keyword in text for keyword in KEEP_ALL_KEYWORDS)


def analyze_case(record: dict, skill_cache: dict[str, dict]) -> dict | None:
    gt = {normalize(x) for x in record.get("ground_truth", [])}
    if not gt:
        return None

    action_spaces: list[set[str]] = []
    for turn in record.get("trajectory", []) or []:
        for query, result in zip(turn.get("executed_queries", []) or [], turn.get("backend_results", []) or []):
            if query.get("tool_name") != "action":
                continue
            entities = {normalize(x) for x in extract_entities((result or {}).get("response_text") or "")}
            if entities:
                action_spaces.append(entities)

    covering_spaces = [space for space in action_spaces if gt <= space]
    if not covering_spaces:
        return None

    chosen = min(covering_spaces, key=len)
    subset_needed = len(chosen) > len(gt)
    gt_size = "single_gt" if len(gt) == 1 else "multi_gt"
    exact = not subset_needed

    selected_skill_ids = (record.get("skill_bundle") or {}).get("selected_case_ids", [])
    skills = [load_skill(skill_id, skill_cache) for skill_id in selected_skill_ids]
    if not skills:
        return None

    count_votes = Counter(answer_count_bucket((skill.get("answer_strategy") or {}).get("answer_count")) for skill in skills)
    majority_count = count_votes.most_common(1)[0][0] if count_votes else "single"
    expected_count = "single" if len(gt) <= 1 else "multi"
    subset_cue_present = any(has_subset_cue(skill) for skill in skills)
    keep_all_cue_present = any(has_keep_all_cue(skill) for skill in skills)

    return {
        "case_id": record["case_id"],
        "question": record.get("question", ""),
        "f1": record.get("f1", 0.0),
        "ground_truth": record.get("ground_truth", []),
        "predicted": record.get("predicted", []),
        "subset_needed": subset_needed,
        "exact": exact,
        "gt_size": gt_size,
        "majority_count": majority_count,
        "expected_count": expected_count,
        "count_aligned": majority_count == expected_count,
        "subset_cue_present": subset_cue_present,
        "keep_all_cue_present": keep_all_cue_present,
        "selected_case_ids": selected_skill_ids,
        "count_votes": dict(count_votes),
        "selected_rule_texts": [
            (skill.get("case_id") or skill.get("question_id") or "", (skill.get("answer_strategy") or {}).get("selection_rule", ""))
            for skill in skills
        ],
    }


def pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.0%"
    return f"{numerator / denominator * 100:.1f}%"


def build_report(results_label: str, output_path: Path) -> None:
    records = json.loads(Path(f"/zhaoshu/subgraph/reports/skill_enhanced_test/{results_label}/results.json").read_text())
    skill_cache: dict[str, dict] = {}
    summaries = [summary for summary in (analyze_case(record, skill_cache) for record in records) if summary]

    def bucket(predicate):
        return [item for item in summaries if predicate(item)]

    subset_cases = bucket(lambda item: item["subset_needed"])
    exact_cases = bucket(lambda item: item["exact"])

    subset_correct = [item for item in subset_cases if item["f1"] >= 0.95]
    subset_wrong = [item for item in subset_cases if item["f1"] < 0.95]
    exact_correct = [item for item in exact_cases if item["f1"] >= 0.95]
    exact_wrong = [item for item in exact_cases if item["f1"] < 0.95]

    def count(items: list[dict], key: str) -> int:
        return sum(1 for item in items if item[key])

    lines: list[str] = []
    lines.append("# Skill Guidance At Action-Space Granularity")
    lines.append("")
    lines.append(f"- Results label: `{results_label}`")
    lines.append("")
    lines.append("## Main Findings")
    lines.append("")
    lines.append(
        f"- Single-action-cover cases with **exact match** between one action space and GT: `{len(exact_cases)}`"
    )
    lines.append(
        f"- Single-action-cover cases where GT is only a **proper subset** of one action space: `{len(subset_cases)}`"
    )
    lines.append(
        "- So the key skill question is not just `single vs multiple`, but whether the loaded skill hints a `keep all` or `select subset` behavior within one chosen action space."
    )
    lines.append("")
    lines.append("## Exact-Match Cases (One Action Already Matches GT Exactly)")
    lines.append("")
    lines.append("| Bucket | Cases | Count Prior Aligned | Keep-All Cue Present | Subset Cue Present |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, items in [("correct", exact_correct), ("wrong", exact_wrong)]:
        lines.append(
            f"| `{name}` | {len(items)} | {count(items, 'count_aligned')} ({pct(count(items, 'count_aligned'), len(items))}) | {count(items, 'keep_all_cue_present')} ({pct(count(items, 'keep_all_cue_present'), len(items))}) | {count(items, 'subset_cue_present')} ({pct(count(items, 'subset_cue_present'), len(items))}) |"
        )
    lines.append("")
    lines.append("## Subset-Needed Cases (GT Is Only Part of One Action)")
    lines.append("")
    lines.append("| Bucket | Cases | Count Prior Aligned | Subset Cue Present | Keep-All Cue Present |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, items in [("correct", subset_correct), ("wrong", subset_wrong)]:
        lines.append(
            f"| `{name}` | {len(items)} | {count(items, 'count_aligned')} ({pct(count(items, 'count_aligned'), len(items))}) | {count(items, 'subset_cue_present')} ({pct(count(items, 'subset_cue_present'), len(items))}) | {count(items, 'keep_all_cue_present')} ({pct(count(items, 'keep_all_cue_present'), len(items))}) |"
        )
    lines.append("")
    lines.append("## Subset-Needed Cases by GT Cardinality")
    lines.append("")
    lines.append("| Bucket | Cases | Count Prior Aligned | Subset Cue Present |")
    lines.append("|---|---:|---:|---:|")
    for bucket_name in ["single_gt", "multi_gt"]:
        items = [item for item in subset_cases if item["gt_size"] == bucket_name]
        lines.append(
            f"| `{bucket_name}` | {len(items)} | {count(items, 'count_aligned')} ({pct(count(items, 'count_aligned'), len(items))}) | {count(items, 'subset_cue_present')} ({pct(count(items, 'subset_cue_present'), len(items))}) |"
        )
    lines.append("")
    lines.append("## Representative Subset-Needed Wrong Cases")
    lines.append("")
    shown = 0
    for item in subset_wrong:
        lines.append(
            f"- `{item['case_id']}`: GT `{item['ground_truth']}` vs Pred `{item['predicted']}` | count votes `{item['count_votes']}` | subset cue `{item['subset_cue_present']}` | keep-all cue `{item['keep_all_cue_present']}` | question: `{item['question']}`"
        )
        shown += 1
        if shown >= 5:
            break
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- The selected skill bundle already carries a visible `single/multi` prior via `answer_strategy.answer_count`; this part is explicit."
    )
    lines.append(
        "- But `subset` guidance is much fuzzier. It mainly appears indirectly in selection-rule wording such as `specific`, `explicitly`, or `specified`, rather than as a first-class structured field."
    )
    lines.append(
        "- In fact, those subset-like wording cues are close to saturated in the selected skill bundles, so they are **not very discriminative**: the current corpus overuses `specific-selection` language even for many cases where keeping all answers from one action would be correct."
    )
    lines.append(
        "- So for single-action subset cases, the current skill system helps somewhat, but not in a cleanly typed way. The model still has to infer whether to keep all candidates or keep only a proper subset."
    )
    lines.append(
        "- This supports the next refinement: keep `answer_count`, but add an action-space-level strategy notion like `keep_all_from_one_action` vs `select_subset_within_one_action`."
    )

    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze whether loaded skills provide useful answer strategy at action-space granularity.")
    parser.add_argument(
        "--results-label",
        default="fulltest_skill_top3_protocol_guard_20260330",
        help="Report label under reports/skill_enhanced_test.",
    )
    parser.add_argument(
        "--output",
        default="/zhaoshu/subgraph/reports/skill_enhanced_test/skill_action_space_guidance_20260331.md",
        help="Markdown output path.",
    )
    args = parser.parse_args()

    build_report(args.results_label, Path(args.output))


if __name__ == "__main__":
    main()
