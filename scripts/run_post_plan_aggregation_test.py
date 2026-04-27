#!/usr/bin/env python3
"""
Post-Plan Aggregation Test Runner

Runs the skill-enhanced test with post-plan aggregation mode
(disabled pre-plan injection, aggregated skill mode) and generates
a comparison report against the best baseline metrics.

Usage:
    # Full run (all 1494 cases)
    python scripts/run_post_plan_aggregation_test.py

    # Quick smoke test
    python scripts/run_post_plan_aggregation_test.py --limit-cases 20

    # Specific case
    python scripts/run_post_plan_aggregation_test.py --case-id WebQTest-6
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]

# Best baseline metrics (protocol_guard_action_id_experiment + v5_filter_then_answer)
BEST_BASELINE = {
    "variant": "protocol_guard_action_id_experiment",
    "stage5_prompt_variant": "v5_filter_then_answer",
    "skill_mode": "per_skill (legacy)",
    "pre_plan_injection": "enabled",
    "avg_f1": 0.7918,
    "hit_at_1": 0.8213,
    "hit_at_0_8": 0.7222,
    "exact_match": 0.6948,
}


def _run_test(args: argparse.Namespace) -> Path:
    """Run the skill-enhanced test via subprocess and return the output directory."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_skill_enhanced_test.py"),
        "--variant", "protocol_guard_action_id_experiment",
        "--stage5-prompt-variant", "v5_filter_then_answer",
        "--label", args.label,
        "--skill-top-k", str(args.skill_top_k),
        "--max-concurrency", str(args.max_concurrency),
        "--max-turns", str(args.max_turns),
    ]

    if args.limit_cases is not None:
        cmd.extend(["--limit-cases", str(args.limit_cases)])

    for cid in (args.case_id or []):
        cmd.extend(["--case-id", cid])

    env_override = {
        "KGQA_SKILL_PRE_PLAN_INJECTION": "0",
        "KGQA_SKILL_AGGREGATION_MODE": "aggregated",
    }

    print("=" * 60)
    print("Post-Plan Aggregation Test")
    print("=" * 60)
    print(f"Environment overrides:")
    for k, v in env_override.items():
        print(f"  {k}={v}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    import os
    env = {**os.environ, **env_override}

    result = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if result.returncode != 0:
        print(f"\nTest runner exited with code {result.returncode}")
        # Still attempt to generate report if results exist

    output_dir = ROOT / "reports" / "skill_enhanced_test" / args.label
    return output_dir


def _generate_comparison_report(output_dir: Path) -> str:
    """Read results JSON and generate comparison markdown."""
    results_path = output_dir / "results.json"
    if not results_path.exists():
        return f"# Post-Plan Aggregation vs Best Baseline\n\nERROR: results.json not found at {results_path}\n"

    with results_path.open() as f:
        results = json.load(f)

    total = len(results)
    failed = sum(1 for r in results if r.get("is_failed"))
    avg_f1 = mean(r["f1"] for r in results) if results else 0.0
    hit_at_1 = mean(r.get("hit_at_1", 0.0) for r in results) if results else 0.0
    hit_at_0_8 = mean(1.0 if r["f1"] >= 0.8 else 0.0 for r in results) if results else 0.0
    exact_match = mean(1.0 if r["f1"] >= 0.95 else 0.0 for r in results) if results else 0.0

    delta_f1 = avg_f1 - BEST_BASELINE["avg_f1"]
    delta_hit1 = hit_at_1 - BEST_BASELINE["hit_at_1"]
    delta_hit08 = hit_at_0_8 - BEST_BASELINE["hit_at_0_8"]
    delta_em = exact_match - BEST_BASELINE["exact_match"]

    def _fmt_delta(d: float) -> str:
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.4f}"

    hit1_regression_threshold = 0.01
    hit1_regression = delta_hit1 < -hit1_regression_threshold
    regression_status = "FAIL" if hit1_regression else "PASS"

    lines = [
        "# Post-Plan Aggregation vs Best Baseline",
        "",
        "## Configuration",
        "",
        "| Setting | Best Baseline | Post-Plan Aggregation |",
        "|---|---|---|",
        f"| variant | {BEST_BASELINE['variant']} | protocol_guard_action_id_experiment |",
        f"| stage5_prompt_variant | {BEST_BASELINE['stage5_prompt_variant']} | v5_filter_then_answer |",
        f"| skill_mode | {BEST_BASELINE['skill_mode']} | aggregated (new) |",
        f"| pre_plan_injection | {BEST_BASELINE['pre_plan_injection']} | disabled |",
        "",
        "## Metrics Comparison",
        "",
        "| Metric | Best Baseline | Post-Plan Aggregation | Delta |",
        "|---|---:|---:|---:|",
        f"| Avg F1 | {BEST_BASELINE['avg_f1']:.4f} | {avg_f1:.4f} | {_fmt_delta(delta_f1)} |",
        f"| Hit@1 | {BEST_BASELINE['hit_at_1']:.4f} | {hit_at_1:.4f} | {_fmt_delta(delta_hit1)} |",
        f"| Hit@0.8 | {BEST_BASELINE['hit_at_0_8']:.4f} | {hit_at_0_8:.4f} | {_fmt_delta(delta_hit08)} |",
        f"| EM | {BEST_BASELINE['exact_match']:.4f} | {exact_match:.4f} | {_fmt_delta(delta_em)} |",
        "",
        "## Summary Statistics",
        "",
        f"- Total cases: {total}",
        f"- Failed cases: {failed}",
        "",
        "## Regression Check",
        "",
        f"- Hit@1 regression: **{regression_status}** (delta={_fmt_delta(delta_hit1)} vs {hit1_regression_threshold:.0%} threshold)",
        "",
    ]

    if hit1_regression:
        lines.extend([
            "> **WARNING**: Hit@1 regressed by more than 1% compared to best baseline.",
            "> Investigate aggregated skill quality and prompt compatibility.",
            "",
        ])

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run post-plan aggregation test and compare against best baseline."
    )
    parser.add_argument(
        "--label", default="post_plan_aggregation",
        help="Label for this run (default: post_plan_aggregation)",
    )
    parser.add_argument(
        "--limit-cases", type=int, default=None,
        help="Limit number of cases (default: None = all 1494)",
    )
    parser.add_argument(
        "--case-id", action="append", default=None,
        help="Run only specific case id(s); may be repeated",
    )
    parser.add_argument(
        "--skill-top-k", type=int, default=3,
        help="Number of similar skills to retrieve (default: 3)",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=32,
        help="Maximum concurrent cases (default: 32)",
    )
    parser.add_argument(
        "--max-turns", type=int, default=8,
        help="Maximum turns per case (default: 8)",
    )
    args = parser.parse_args()

    # Step 1: Run the test
    output_dir = _run_test(args)

    # Step 2: Generate comparison report
    report_text = _generate_comparison_report(output_dir)
    report_dir = ROOT / "reports" / "skill_enhanced_test" / "post_plan_aggregation"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "comparison.md"
    report_path.write_text(report_text)
    print(f"\nComparison report saved to: {report_path}")
    print("\n" + report_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
