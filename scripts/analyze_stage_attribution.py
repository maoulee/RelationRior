#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def normalize(value: object) -> str:
    return str(value).strip().lower()


def classify_case(record: dict) -> tuple[str, dict[str, int]]:
    gt = {normalize(x) for x in record.get("ground_truth", [])}
    pred = {normalize(x) for x in record.get("predicted", [])}
    trajectory = record.get("trajectory", []) or []

    has_plan = False
    has_action_hint = False
    any_action = False
    all_leaf_entities: set[str] = set()
    retrieved_candidates: set[str] = set()
    frontend_stage_counts: Counter[str] = Counter()
    frontend_type_counts: Counter[str] = Counter()

    for turn in trajectory:
        for query in turn.get("executed_queries", []) or []:
            tool = query.get("tool_name")
            if tool in ("plan", "plan_subquestion"):
                has_plan = True
            if tool in ("action", "match_pattern", "select_action"):
                any_action = True

        state_snapshot = turn.get("state_snapshot", {}) or {}
        has_action_hint = has_action_hint or bool(state_snapshot.get("has_action_hint"))
        all_leaf_entities.update(normalize(x) for x in (state_snapshot.get("all_leaf_entities") or []))
        retrieved_candidates.update(
            normalize(x) for x in (state_snapshot.get("retrieved_candidates") or [])
        )

        for frontend_error in turn.get("frontend_errors", []) or []:
            if isinstance(frontend_error, dict):
                error_type = frontend_error.get("error_type", "UNKNOWN")
                tool_name = frontend_error.get("tool_name", "")
            else:
                error_type = str(frontend_error)
                tool_name = ""
            frontend_type_counts[error_type] += 1

            if tool_name in ("plan", "plan_subquestion"):
                frontend_stage_counts["plan"] += 1
            elif tool_name in ("action", "match_pattern", "select_action"):
                frontend_stage_counts["action"] += 1
            elif tool_name == "filter":
                frontend_stage_counts["reasoning"] += 1
            else:
                frontend_stage_counts["other"] += 1

    candidate_pool = all_leaf_entities | retrieved_candidates
    gt_in_candidates = bool(gt & candidate_pool)
    final_correct = bool(gt & pred) and record.get("f1", 0.0) >= 0.95

    if final_correct:
        primary = "correct_with_compliance" if frontend_stage_counts else "correct"
    elif frontend_stage_counts:
        if frontend_stage_counts["plan"] >= max(
            frontend_stage_counts["action"],
            frontend_stage_counts["reasoning"],
            frontend_stage_counts["other"],
        ):
            primary = "plan_compliance"
        elif frontend_stage_counts["action"] >= max(
            frontend_stage_counts["plan"],
            frontend_stage_counts["reasoning"],
            frontend_stage_counts["other"],
        ):
            primary = "action_compliance"
        elif frontend_stage_counts["reasoning"] >= max(
            frontend_stage_counts["plan"],
            frontend_stage_counts["action"],
            frontend_stage_counts["other"],
        ):
            primary = "reasoning_compliance"
        else:
            primary = "other_compliance"
    elif gt_in_candidates:
        primary = "reasoning"
    elif has_action_hint or any_action:
        primary = "action"
    elif has_plan:
        primary = "plan"
    else:
        primary = "plan"

    return primary, dict(frontend_type_counts)


def summarize(records: dict[str, dict]) -> tuple[Counter[str], Counter[str], Counter[str], Counter[str]]:
    all_counts: Counter[str] = Counter()
    wrong_counts: Counter[str] = Counter()
    partial_counts: Counter[str] = Counter()
    frontend_error_types: Counter[str] = Counter()

    for record in records.values():
        primary, frontend_types = classify_case(record)
        all_counts[primary] += 1
        if record.get("f1", 0.0) < 0.95:
            wrong_counts[primary] += 1
        if 0.0 < record.get("f1", 0.0) < 0.95:
            partial_counts[primary] += 1
        frontend_error_types.update(frontend_types)

    return all_counts, wrong_counts, partial_counts, frontend_error_types


def build_report(base_label: str, skill_label: str, output_path: Path) -> None:
    base_results = {
        record["case_id"]: record
        for record in json.loads(
            Path(f"/zhaoshu/subgraph/reports/skill_enhanced_test/{base_label}/results.json").read_text()
        )
    }
    skill_results = {
        record["case_id"]: record
        for record in json.loads(
            Path(f"/zhaoshu/subgraph/reports/skill_enhanced_test/{skill_label}/results.json").read_text()
        )
    }

    base_all, base_wrong, _, base_fe = summarize(base_results)
    skill_all, skill_wrong, _, skill_fe = summarize(skill_results)

    transitions: Counter[tuple[str, str]] = Counter()
    transition_examples: defaultdict[tuple[str, str], list[tuple[str, float, float]]] = defaultdict(list)

    for case_id, base_record in base_results.items():
        skill_record = skill_results[case_id]
        base_primary, _ = classify_case(base_record)
        skill_primary, _ = classify_case(skill_record)
        if abs(base_record["f1"] - skill_record["f1"]) > 1e-9:
            transitions[(base_primary, skill_primary)] += 1
            if len(transition_examples[(base_primary, skill_primary)]) < 5:
                transition_examples[(base_primary, skill_primary)].append(
                    (case_id, base_record["f1"], skill_record["f1"])
                )

    lines: list[str] = []
    lines.append("# Stage Attribution Analysis")
    lines.append("")
    lines.append(f"- Baseline: `{base_label}`")
    lines.append(f"- Skills: `{skill_label}`")
    lines.append("")
    lines.append("## Attribution Rule")
    lines.append("")
    lines.append("- `plan`: no candidate path reaches GT after planning.")
    lines.append("- `action`: action/action-space exists, but GT never enters the candidate pool.")
    lines.append("- `reasoning`: GT enters the candidate pool, but final answer is still wrong/partial.")
    lines.append("- `*_compliance`: frontend validation errors are the dominant blocking factor.")
    lines.append("")
    lines.append("## Wrong-Case Breakdown")
    lines.append("")
    lines.append("| Bucket | Baseline Wrong | +Skills Wrong | Delta |")
    lines.append("|---|---:|---:|---:|")
    for key in [
        "plan",
        "action",
        "reasoning",
        "plan_compliance",
        "action_compliance",
        "reasoning_compliance",
        "other_compliance",
    ]:
        base_value = base_wrong.get(key, 0)
        skill_value = skill_wrong.get(key, 0)
        lines.append(f"| `{key}` | {base_value} | {skill_value} | {skill_value - base_value:+d} |")
    lines.append("")
    lines.append("## Key Takeaways")
    lines.append("")
    lines.append(
        f"- Pure `plan` failures are almost absent: baseline `{base_wrong.get('plan', 0)}`, +skills `{skill_wrong.get('plan', 0)}`."
    )
    lines.append(
        f"- `action` failures drop from `{base_wrong.get('action', 0)}` to `{skill_wrong.get('action', 0)}`: skills are helping relation/action-space selection."
    )
    lines.append(
        f"- `reasoning` failures rise from `{base_wrong.get('reasoning', 0)}` to `{skill_wrong.get('reasoning', 0)}`: more cases now reach the right candidate pool but are finalized incorrectly."
    )
    base_compliance = (
        base_wrong.get("plan_compliance", 0)
        + base_wrong.get("action_compliance", 0)
        + base_wrong.get("reasoning_compliance", 0)
        + base_wrong.get("other_compliance", 0)
    )
    skill_compliance = (
        skill_wrong.get("plan_compliance", 0)
        + skill_wrong.get("action_compliance", 0)
        + skill_wrong.get("reasoning_compliance", 0)
        + skill_wrong.get("other_compliance", 0)
    )
    lines.append(
        f"- Compliance-style failures change only slightly: baseline `{base_compliance}`, +skills `{skill_compliance}`."
    )
    lines.append("")
    lines.append("## Frontend Error Types")
    lines.append("")
    lines.append("| Error Type | Baseline Count | +Skills Count | Delta |")
    lines.append("|---|---:|---:|---:|")
    for error_type in sorted(set(base_fe) | set(skill_fe)):
        lines.append(
            f"| `{error_type}` | {base_fe.get(error_type, 0)} | {skill_fe.get(error_type, 0)} | {skill_fe.get(error_type, 0) - base_fe.get(error_type, 0):+d} |"
        )
    lines.append("")
    lines.append("## Changed-Case Transition Highlights")
    lines.append("")
    for (src, dst), count in transitions.most_common(12):
        examples = ", ".join(
            f"`{case_id}` ({base_f1:.2f}->{skill_f1:.2f})"
            for case_id, base_f1, skill_f1 in transition_examples[(src, dst)]
        )
        lines.append(f"- `{src} -> {dst}`: `{count}` case(s). Examples: {examples}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- The current skill design is already improving the upstream search space: fewer failures remain in the `action` bucket."
    )
    lines.append(
        "- The main regression source is now downstream: answer-slot drift, over-shrinking, or choosing the wrong item from an already-good candidate pool."
    )
    lines.append(
        "- In other words: the next correction target should be `reasoning` and answer-slot control, not broad planning logic."
    )

    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze baseline vs skill stage attribution.")
    parser.add_argument(
        "--baseline-label",
        default="fulltest_baseline_protocol_guard_20260330",
        help="Baseline report label under reports/skill_enhanced_test",
    )
    parser.add_argument(
        "--skills-label",
        default="fulltest_skill_top3_protocol_guard_20260330",
        help="Skills report label under reports/skill_enhanced_test",
    )
    parser.add_argument(
        "--output",
        default="/zhaoshu/subgraph/reports/skill_enhanced_test/stage_attribution_20260331.md",
        help="Output markdown path",
    )
    args = parser.parse_args()

    build_report(args.baseline_label, args.skills_label, Path(args.output))
    print(args.output)


if __name__ == "__main__":
    main()
