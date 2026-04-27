#!/usr/bin/env python3
"""
Detailed per-stage comparison between two test runs.

Produces a markdown report with:
  - Aggregate stage-level comparison (Plan, Action, Reason F1)
  - Per-case regression/improvement analysis with stage attribution
  - Hit@1 movement analysis
  - Skill hint impact analysis

Usage:
    python scripts/compare_test_runs.py \
        reports/skill_enhanced_test/baseline/results.json \
        reports/skill_enhanced_test/experiment/results.json \
        --output reports/comparison_report.md

    # Optional: include skill hint content from trajectories
    python scripts/compare_test_runs.py baseline.json experiment.json \
        --show-hints \
        --top-n 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_results(path: str) -> Dict[str, Dict[str, Any]]:
    """Load results.json and index by case_id."""
    with open(path) as f:
        data = json.load(f)
    return {r["case_id"]: r for r in data}


def extract_reasoning(raw: str) -> str:
    """Extract <reasoning>...</reasoning> blocks."""
    matches = re.findall(r"<reasoning>(.*?)</reasoning>", raw, re.DOTALL)
    return "\n".join(m.strip() for m in matches)


def _stage_f1(result: Dict, stage: str) -> float:
    """Extract F1 for a stage from nested stage_scores structure.

    Handles both nested ``{plan: {f1: 0.5}}`` and flat ``{plan_f1: 0.5}``.
    """
    ss = result.get("stage_scores", {})
    # Nested format
    nested = ss.get(stage, {})
    if isinstance(nested, dict) and "f1" in nested:
        return nested["f1"]
    # Flat format
    return ss.get(f"{stage}_f1", 0)


def classify_stage_attribution(b: Dict, n: Dict) -> str:
    """Determine which stage drives the F1 change."""
    plan_diff = _stage_f1(n, "plan") - _stage_f1(b, "plan")
    action_diff = _stage_f1(n, "action") - _stage_f1(b, "action")
    reason_diff = _stage_f1(n, "final_reasoning") - _stage_f1(b, "final_reasoning")

    # Use absolute magnitude to find dominant stage
    diffs = {"plan": abs(plan_diff), "action": abs(action_diff), "reason": abs(reason_diff)}
    if max(diffs.values()) < 0.01:
        return "none"

    stages = []
    for name, diff in [("plan", plan_diff), ("action", action_diff), ("reason", reason_diff)]:
        if abs(diff) >= 0.3:
            stages.append(f"{name}({diff:+.2f})")

    if not stages:
        dominant = max(diffs, key=diffs.get)
        stages.append(f"{dominant}")

    return ", ".join(stages)


def get_skill_hints(trajectory: List[Dict]) -> str:
    """Extract skill hint content from trajectory feedback."""
    hints = []
    for step in trajectory:
        feedback = step.get("feedback", "")
        if "Historical Case Reference" in str(feedback) or "Reference question:" in str(feedback):
            # Extract the relevant section
            for block in feedback.split("\n\n"):
                if "Reference question:" in block or "Historical Case Reference" in block:
                    hints.append(block[:300])
    return "\n---\n".join(hints) if hints else "(no skill hints found)"


def compare_runs(
    baseline_path: str,
    experiment_path: str,
    output_path: Optional[str] = None,
    show_hints: bool = False,
    top_n: int = 20,
) -> str:
    """Compare two test runs and produce a markdown report."""

    baseline = load_results(baseline_path)
    experiment = load_results(experiment_path)

    common = sorted(set(baseline.keys()) & set(experiment.keys()))

    if not common:
        return "ERROR: No common cases found between the two runs."

    # === Aggregate metrics ===
    agg = {"baseline": defaultdict(float), "experiment": defaultdict(float)}

    for cid in common:
        b, n = baseline[cid], experiment[cid]
        for key in ["f1", "hit_at_1"]:
            agg["baseline"][key] += b.get(key, 0)
            agg["experiment"][key] += n.get(key, 0)
        # Stage scores
        for stage_key, stage_name in [("plan_f1", "plan"), ("action_f1", "action"), ("reason_f1", "final_reasoning")]:
            agg["baseline"][stage_key] += _stage_f1(b, stage_name)
            agg["experiment"][stage_key] += _stage_f1(n, stage_name)

    n_cases = len(common)

    # === Case-level comparison ===
    cases = []
    for cid in common:
        b, n = baseline[cid], experiment[cid]
        f1_diff = n.get("f1", 0) - b.get("f1", 0)
        hit_diff = n.get("hit_at_1", 0) - b.get("hit_at_1", 0)
        plan_b_val = _stage_f1(b, "plan")
        plan_n_val = _stage_f1(n, "plan")
        action_b_val = _stage_f1(b, "action")
        action_n_val = _stage_f1(n, "action")
        reason_b_val = _stage_f1(b, "final_reasoning")
        reason_n_val = _stage_f1(n, "final_reasoning")

        cases.append({
            "id": cid,
            "f1_b": b.get("f1", 0), "f1_n": n.get("f1", 0), "f1_diff": f1_diff,
            "hit_b": b.get("hit_at_1", 0), "hit_n": n.get("hit_at_1", 0), "hit_diff": hit_diff,
            "plan_b": plan_b_val,
            "plan_n": plan_n_val, "plan_diff": plan_n_val - plan_b_val,
            "action_b": action_b_val,
            "action_n": action_n_val, "action_diff": action_n_val - action_b_val,
            "reason_b": reason_b_val,
            "reason_n": reason_n_val, "reason_diff": reason_n_val - reason_b_val,
            "pred_b": b.get("predicted", ["-"])[0] if b.get("predicted") else "-",
            "pred_n": n.get("predicted", ["-"])[0] if n.get("predicted") else "-",
            "gt": b.get("ground_truth", ["-"])[0] if b.get("ground_truth") else "-",
            "question": b.get("question", ""),
        })

    # Classify
    regressions = sorted([c for c in cases if c["f1_diff"] < -0.01], key=lambda x: x["f1_diff"])
    improvements = sorted([c for c in cases if c["f1_diff"] > 0.01], key=lambda x: -x["f1_diff"])
    unchanged = [c for c in cases if abs(c["f1_diff"]) <= 0.01]

    hit1_dropped = sorted([c for c in cases if c["hit_diff"] < 0], key=lambda x: x["f1_diff"])
    hit1_gained = sorted([c for c in cases if c["hit_diff"] > 0], key=lambda x: -x["f1_diff"])

    # === Stage attribution ===
    stage_attribution = {"plan": 0, "action": 0, "reason": 0, "none": 0}
    for c in regressions:
        diffs = {"plan": abs(c["plan_diff"]), "action": abs(c["action_diff"]), "reason": abs(c["reason_diff"])}
        if max(diffs.values()) < 0.01:
            stage_attribution["none"] += 1
        else:
            stage_attribution[max(diffs, key=diffs.get)] += 1

    # === Build report ===
    lines = []
    bl = Path(baseline_path).parent.name
    el = Path(experiment_path).parent.name

    lines.append(f"# Detailed Test Run Comparison")
    lines.append(f"")
    lines.append(f"- Baseline: `{bl}`")
    lines.append(f"- Experiment: `{el}`")
    lines.append(f"- Common cases: {n_cases}")
    lines.append(f"")

    # Aggregate table
    lines.append(f"## 1. Aggregate Stage Comparison")
    lines.append(f"")
    lines.append(f"| Metric | Baseline | Experiment | Delta | Delta % |")
    lines.append(f"|--------|----------|------------|-------|---------|")

    for key, label in [("plan_f1", "Plan F1"), ("action_f1", "Action F1"), ("reason_f1", "Reason F1"), ("f1", "Final F1"), ("hit_at_1", "Hit@1")]:
        b_val = agg["baseline"][key] / n_cases
        n_val = agg["experiment"][key] / n_cases
        delta = n_val - b_val
        pct = (delta / b_val * 100) if b_val > 0 else 0
        marker = " **" if abs(delta) > 0.01 else ""
        lines.append(f"| {label} | {b_val:.4f} | {n_val:.4f} | {delta:+.4f}{marker} | {pct:+.1f}% |")

    lines.append(f"")

    # Summary counts
    lines.append(f"## 2. Case Movement Summary")
    lines.append(f"")
    lines.append(f"| Category | Count |")
    lines.append(f"|----------|-------|")
    lines.append(f"| Regressions (F1 < -0.01) | {len(regressions)} |")
    lines.append(f"| Improvements (F1 > +0.01) | {len(improvements)} |")
    lines.append(f"| Unchanged | {len(unchanged)} |")
    lines.append(f"| Hit@1 dropped | {len(hit1_dropped)} |")
    lines.append(f"| Hit@1 gained | {len(hit1_gained)} |")
    lines.append(f"")

    # Stage attribution
    lines.append(f"## 3. Stage Attribution for Regressions")
    lines.append(f"")
    lines.append(f"Which stage drives the worst F1 drop in each regressed case:")
    lines.append(f"")
    lines.append(f"| Stage | Count |")
    lines.append(f"|-------|-------|")
    for stage, count in sorted(stage_attribution.items(), key=lambda x: -x[1]):
        lines.append(f"| {stage} | {count} |")
    lines.append(f"")

    # Avg stage diffs in regressions
    if regressions:
        avg_plan = sum(c["plan_diff"] for c in regressions) / len(regressions)
        avg_action = sum(c["action_diff"] for c in regressions) / len(regressions)
        avg_reason = sum(c["reason_diff"] for c in regressions) / len(regressions)
        lines.append(f"Avg stage delta in regressions (n={len(regressions)}):")
        lines.append(f"  Plan: {avg_plan:+.4f} | Action: {avg_action:+.4f} | Reason: {avg_reason:+.4f}")
        lines.append(f"")

    if improvements:
        avg_plan_i = sum(c["plan_diff"] for c in improvements) / len(improvements)
        avg_action_i = sum(c["action_diff"] for c in improvements) / len(improvements)
        avg_reason_i = sum(c["reason_diff"] for c in improvements) / len(improvements)
        lines.append(f"Avg stage delta in improvements (n={len(improvements)}):")
        lines.append(f"  Plan: {avg_plan_i:+.4f} | Action: {avg_action_i:+.4f} | Reason: {avg_reason_i:+.4f}")
        lines.append(f"")

    # Detailed regression table
    lines.append(f"## 4. Top Regressions (worst F1 drop)")
    lines.append(f"")
    lines.append(f"| Case | F1 Base | F1 Exp | Delta | Plan | Action | Reason | Baseline Pred → Experiment Pred |")
    lines.append(f"|------|---------|--------|-------|------|--------|--------|-------------------------------|")
    for c in regressions[:top_n]:
        bp = c["pred_b"][:30]
        np_ = c["pred_n"][:30]
        lines.append(f"| {c['id']} | {c['f1_b']:.2f} | {c['f1_n']:.2f} | {c['f1_diff']:+.2f} | {c['plan_diff']:+.2f} | {c['action_diff']:+.2f} | {c['reason_diff']:+.2f} | {bp} → {np_} |")
    lines.append(f"")

    # Detailed improvement table
    lines.append(f"## 5. Top Improvements (best F1 gain)")
    lines.append(f"")
    lines.append(f"| Case | F1 Base | F1 Exp | Delta | Plan | Action | Reason | Baseline Pred → Experiment Pred |")
    lines.append(f"|------|---------|--------|-------|------|--------|--------|-------------------------------|")
    for c in improvements[:top_n]:
        bp = c["pred_b"][:30]
        np_ = c["pred_n"][:30]
        lines.append(f"| {c['id']} | {c['f1_b']:.2f} | {c['f1_n']:.2f} | {c['f1_diff']:+.2f} | {c['plan_diff']:+.2f} | {c['action_diff']:+.2f} | {c['reason_diff']:+.2f} | {bp} → {np_} |")
    lines.append(f"")

    # Hit@1 analysis
    lines.append(f"## 6. Hit@1 Movement Analysis")
    lines.append(f"")

    if hit1_dropped:
        lines.append(f"### Hit@1 Dropped ({len(hit1_dropped)} cases)")
        lines.append(f"")
        lines.append(f"| Case | Question | Baseline Pred | Experiment Pred | GT | Plan | Action | Reason |")
        lines.append(f"|------|----------|---------------|-----------------|-----|------|--------|--------|")
        for c in hit1_dropped[:top_n]:
            q = c["question"][:40]
            lines.append(f"| {c['id']} | {q} | {c['pred_b'][:25]} | {c['pred_n'][:25]} | {c['gt'][:25]} | {c['plan_diff']:+.2f} | {c['action_diff']:+.2f} | {c['reason_diff']:+.2f} |")
        lines.append(f"")

    if hit1_gained:
        lines.append(f"### Hit@1 Gained ({len(hit1_gained)} cases)")
        lines.append(f"")
        lines.append(f"| Case | Question | Baseline Pred | Experiment Pred | GT | Plan | Action | Reason |")
        lines.append(f"|------|----------|---------------|-----------------|-----|------|--------|--------|")
        for c in hit1_gained[:top_n]:
            q = c["question"][:40]
            lines.append(f"| {c['id']} | {q} | {c['pred_b'][:25]} | {c['pred_n'][:25]} | {c['gt'][:25]} | {c['plan_diff']:+.2f} | {c['action_diff']:+.2f} | {c['reason_diff']:+.2f} |")
        lines.append(f"")

    # Skill hint analysis (if trajectories available)
    if show_hints and hit1_dropped:
        lines.append(f"## 7. Skill Hint Content for Hit@1 Dropped Cases")
        lines.append(f"")
        for c in hit1_dropped[:5]:
            cid = c["id"]
            n_data = experiment.get(cid, {})
            traj = n_data.get("trajectory", [])
            hints = get_skill_hints(traj)
            lines.append(f"### {cid}: {c['question']}")
            lines.append(f"")
            lines.append(f"GT: {c['gt']}")
            lines.append(f"Baseline: {c['pred_b']} | Experiment: {c['pred_n']}")
            lines.append(f"")
            lines.append(f"Injected skill hints:")
            lines.append(f"```")
            lines.append(hints[:800])
            lines.append(f"```")
            lines.append(f"")

    report = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report)
        print(f"Report saved to {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Detailed per-stage comparison between two test runs")
    parser.add_argument("baseline", help="Path to baseline results.json")
    parser.add_argument("experiment", help="Path to experiment results.json")
    parser.add_argument("--output", "-o", default=None, help="Output markdown file path")
    parser.add_argument("--show-hints", action="store_true", help="Include skill hint content from trajectories")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top cases to show per category")
    args = parser.parse_args()

    report = compare_runs(args.baseline, args.experiment, args.output, args.show_hints, args.top_n)

    if not args.output:
        print(report)


if __name__ == "__main__":
    main()
