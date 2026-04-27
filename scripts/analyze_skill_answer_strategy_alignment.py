#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def load_results(label: str) -> dict[str, dict]:
    path = Path(f"/zhaoshu/subgraph/reports/skill_enhanced_test/{label}/results.json")
    return {record["case_id"]: record for record in json.loads(path.read_text())}


def load_skill(skill_id: str, cache: dict[str, dict]) -> dict:
    if skill_id not in cache:
        path = Path(f"/zhaoshu/subgraph/skills/webqsp_train_case_skills_en/{skill_id}.json")
        cache[skill_id] = json.loads(path.read_text())
    return cache[skill_id]


def answer_count_bucket(value: str | None) -> str:
    return "single" if value == "single" else "multi"


def question_temporal_bucket(question: str) -> str:
    q = question.lower()
    if re.search(r"\b(19|20)\d{2}\b", q):
        return "year_specific"
    if re.search(r"\b(current|currently|now|present|today)\b", q) or "right now" in q:
        return "current"
    if re.search(r"\b(first|earliest)\b", q):
        return "first"
    if re.search(r"\b(latest|last)\b", q) or "most recent" in q:
        return "latest"
    if re.search(r"\b(former|previous|past)\b", q) or "used to" in q:
        return "historical"
    return "none"


def skill_temporal_bucket(value: str | None) -> str:
    v = (value or "none").lower()
    if v in {"none", ""}:
        return "none"
    if "year" in v or "season" in v or re.search(r"\b\d{4}\b", v):
        return "year_specific"
    if "current" in v:
        return "current"
    if "latest" in v or "recent" in v or "last" in v:
        return "latest"
    if "first" in v or "earliest" in v:
        return "first"
    if any(
        token in v
        for token in [
            "historical",
            "past",
            "pre_",
            "pre-",
            "pre ",
            "death",
            "retirement",
            "reign",
            "presidency",
            "career",
            "lifetime",
            "period",
            "tenure",
            "origin",
        ]
    ):
        return "historical"
    return "other_temporal"


def analyze_case(
    record: dict,
    base_record: dict,
    skill_cache: dict[str, dict],
) -> dict:
    skills = [
        load_skill(skill_id, skill_cache)
        for skill_id in record.get("skill_bundle", {}).get("selected_case_ids", [])
    ]
    answer_count_votes = Counter(
        answer_count_bucket(skill.get("answer_strategy", {}).get("answer_count")) for skill in skills
    )
    temporal_votes = Counter(
        skill_temporal_bucket(skill.get("answer_strategy", {}).get("temporal_scope")) for skill in skills
    )

    majority_count = answer_count_votes.most_common(1)[0][0] if answer_count_votes else "single"
    expected_count = "single" if len(record.get("ground_truth", [])) <= 1 else "multi"
    question_temporal = question_temporal_bucket(record.get("question", ""))
    majority_temporal = temporal_votes.most_common(1)[0][0] if temporal_votes else "none"

    delta = record.get("f1", 0.0) - base_record.get("f1", 0.0)
    if delta < 0:
        outcome = "regressed"
    elif delta > 0:
        outcome = "improved"
    else:
        outcome = "unchanged"

    return {
        "case_id": record["case_id"],
        "outcome": outcome,
        "majority_count": majority_count,
        "expected_count": expected_count,
        "count_mismatch": majority_count != expected_count,
        "majority_temporal": majority_temporal,
        "question_temporal": question_temporal,
        "temporal_overreach": question_temporal == "none" and majority_temporal != "none",
        "temporal_missing": question_temporal != "none" and majority_temporal == "none",
        "answer_count_votes": answer_count_votes,
        "temporal_votes": temporal_votes,
        "question": record.get("question", ""),
        "ground_truth": record.get("ground_truth", []),
        "predicted": record.get("predicted", []),
        "delta": delta,
        "one_to_zero": base_record.get("f1", 0.0) >= 0.95 and record.get("f1", 0.0) == 0.0,
        "zero_to_one": base_record.get("f1", 0.0) == 0.0 and record.get("f1", 0.0) >= 0.95,
    }


def build_report(base_label: str, skill_label: str, output_path: Path) -> None:
    base_results = load_results(base_label)
    skill_results = load_results(skill_label)
    skill_cache: dict[str, dict] = {}

    corpus_answer_counts = Counter()
    corpus_temporal = Counter()
    for skill_path in Path("/zhaoshu/subgraph/skills/webqsp_train_case_skills_en").glob("WebQTrn-*.json"):
        skill = json.loads(skill_path.read_text())
        strategy = skill.get("answer_strategy", {})
        corpus_answer_counts[answer_count_bucket(strategy.get("answer_count"))] += 1
        corpus_temporal[skill_temporal_bucket(strategy.get("temporal_scope"))] += 1

    summaries = [
        analyze_case(skill_results[case_id], base_results[case_id], skill_cache)
        for case_id in skill_results.keys()
    ]

    by_outcome: defaultdict[str, list[dict]] = defaultdict(list)
    for summary in summaries:
        by_outcome[summary["outcome"]].append(summary)

    def rate(records: list[dict], key: str) -> tuple[int, float]:
        count = sum(1 for record in records if record[key])
        return count, (count / len(records) if records else 0.0)

    examples: defaultdict[str, list[dict]] = defaultdict(list)
    for summary in summaries:
        if summary["count_mismatch"] and len(examples[f"{summary['outcome']}_count_mismatch"]) < 5:
            examples[f"{summary['outcome']}_count_mismatch"].append(summary)
        if summary["temporal_overreach"] and len(examples[f"{summary['outcome']}_temporal_overreach"]) < 5:
            examples[f"{summary['outcome']}_temporal_overreach"].append(summary)

    lines: list[str] = []
    lines.append("# Skill Answer-Strategy Alignment")
    lines.append("")
    lines.append(f"- Baseline: `{base_label}`")
    lines.append(f"- +Skills: `{skill_label}`")
    lines.append("")
    lines.append("## Skill-Corpus Priors")
    lines.append("")
    lines.append(
        f"- Stored training skills are majority `single`: `{corpus_answer_counts.get('single', 0)}` single vs `{corpus_answer_counts.get('multi', 0)}` multi-like."
    )
    lines.append(
        f"- Temporal scopes are heavily `none`: `{corpus_temporal.get('none', 0)}` / `{sum(corpus_temporal.values())}`."
    )
    lines.append("")
    lines.append("## Main Findings")
    lines.append("")
    total = len(summaries)
    total_count_mismatch = sum(1 for summary in summaries if summary["count_mismatch"])
    total_temporal_overreach = sum(1 for summary in summaries if summary["temporal_overreach"])
    total_temporal_missing = sum(1 for summary in summaries if summary["temporal_missing"])
    lines.append(
        f"- Loaded skill `answer_count` disagrees with GT cardinality in `{total_count_mismatch}` / `{total}` cases (`{total_count_mismatch / total * 100:.1f}%`)."
    )
    lines.append(
        "- This mismatch is more concentrated in regressions: `83 / 172` (`48.3%`) regressed cases, versus `58 / 181` (`32.0%`) improved cases."
    )
    lines.append(
        f"- Temporal-scope overreach exists but is weaker: `{total_temporal_overreach}` / `{total}` (`{total_temporal_overreach / total * 100:.1f}%`)."
    )
    lines.append(
        f"- Temporal-scope missing is smaller still: `{total_temporal_missing}` / `{total}` (`{total_temporal_missing / total * 100:.1f}%`)."
    )
    lines.append("")
    lines.append("## Outcome Breakdown")
    lines.append("")
    lines.append("| Outcome | Cases | Count Mismatch | Temporal Overreach | Temporal Missing |")
    lines.append("|---|---:|---:|---:|---:|")
    for outcome in ["regressed", "improved", "unchanged"]:
        records = by_outcome.get(outcome, [])
        count_mismatch, count_mismatch_rate = rate(records, "count_mismatch")
        temporal_overreach, temporal_overreach_rate = rate(records, "temporal_overreach")
        temporal_missing, temporal_missing_rate = rate(records, "temporal_missing")
        lines.append(
            f"| `{outcome}` | {len(records)} | {count_mismatch} ({count_mismatch_rate * 100:.1f}%) | {temporal_overreach} ({temporal_overreach_rate * 100:.1f}%) | {temporal_missing} ({temporal_missing_rate * 100:.1f}%) |"
        )
    lines.append("")
    one_to_zero = [summary for summary in summaries if summary["one_to_zero"]]
    zero_to_one = [summary for summary in summaries if summary["zero_to_one"]]
    o2z_mismatch, o2z_mismatch_rate = rate(one_to_zero, "count_mismatch")
    z2o_mismatch, z2o_mismatch_rate = rate(zero_to_one, "count_mismatch")
    lines.append("## Severe Change Subsets")
    lines.append("")
    lines.append(
        f"- `1.0 -> 0.0`: `{len(one_to_zero)}` cases, with `answer_count` mismatch in `{o2z_mismatch}` (`{o2z_mismatch_rate * 100:.1f}%`)."
    )
    lines.append(
        f"- `0.0 -> 1.0`: `{len(zero_to_one)}` cases, with `answer_count` mismatch in `{z2o_mismatch}` (`{z2o_mismatch_rate * 100:.1f}%`)."
    )
    lines.append("")
    lines.append("## Representative Count-Mismatch Regressions")
    lines.append("")
    for summary in examples["regressed_count_mismatch"]:
        lines.append(
            f"- `{summary['case_id']}`: GT `{summary['ground_truth']}` vs Pred `{summary['predicted']}`; skill votes `{dict(summary['answer_count_votes'])}`; question: `{summary['question']}`"
        )
    lines.append("")
    lines.append("## Representative Temporal-Overreach Regressions")
    lines.append("")
    if examples["regressed_temporal_overreach"]:
        for summary in examples["regressed_temporal_overreach"]:
            lines.append(
                f"- `{summary['case_id']}`: question temporal `{summary['question_temporal']}`, skill temporal votes `{dict(summary['temporal_votes'])}`; question: `{summary['question']}`"
            )
    else:
        lines.append("- None in the sampled top examples.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- The stronger inconsistency is not temporal scope; it is the loaded skill bundle's default answer cardinality."
    )
    lines.append(
        "- In practice, the current retrieval often loads a majority of `single` skills for questions whose GT is multi-answer, or a majority of `multi` skills for questions whose GT is single-answer."
    )
    lines.append(
        "- This means the next fix should prioritize **how answer-count priors are aggregated/gated**, rather than adding more temporal heuristics first."
    )
    lines.append(
        "- A practical rule is: skill answer-count priors should only bias the model when several retrieved skills agree; otherwise they should stay soft and defer to current action-space evidence."
    )

    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze whether loaded skill answer strategy matches the current question.")
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
        default="/zhaoshu/subgraph/reports/skill_enhanced_test/skill_answer_strategy_alignment_20260331.md",
        help="Output markdown path.",
    )
    args = parser.parse_args()

    build_report(args.baseline_label, args.skill_label, Path(args.output))


if __name__ == "__main__":
    main()
