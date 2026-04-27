#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = {
    "old_good": ROOT / "tmp" / "prompt_tuning" / "old_good_50.jsonl",
    "old_bad": ROOT / "tmp" / "prompt_tuning" / "old_bad_100.jsonl",
    "unstable": ROOT / "tmp" / "prompt_tuning" / "unstable_all.jsonl",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run prompt-tuning screening over isolated evaluation splits.")
    parser.add_argument("--variants", nargs="+", required=True, help="Stage-5 prompt variants to evaluate.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(DEFAULT_SPLITS.keys()),
        default=sorted(DEFAULT_SPLITS.keys()),
        help="Evaluation splits to run (default: all).",
    )
    parser.add_argument("--repeats", type=int, default=2, help="Repeats per variant/split.")
    parser.add_argument("--concurrency", type=int, default=8, help="Case concurrency inside each run.")
    parser.add_argument("--variant", default="protocol_guard_action_id_experiment", help="Base system prompt variant.")
    parser.add_argument("--max-turns", type=int, default=8, help="Base max turns per case.")
    parser.add_argument("--skill-top-k", type=int, default=3, help="Retrieved skill top-k.")
    parser.add_argument(
        "--skills-root",
        default=str(ROOT / "skills" / "webqsp_train_case_skills_actionspace_v1"),
        help="Skill corpus root.",
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "reports" / "skill_enhanced_test" / "prompt_tuning_v1"),
        help="Root directory for prompt-tuning reports.",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env["KGQA_SKILL_REASONING_INJECTION_MODE"] = env.get("KGQA_SKILL_REASONING_INJECTION_MODE", "per_skill")
    env["KGQA_EXTEND_TURNS_ON_FRONTEND_ERROR"] = env.get("KGQA_EXTEND_TURNS_ON_FRONTEND_ERROR", "1")
    env["KGQA_MAX_TURNS_WITH_FRONTEND_REPAIR"] = env.get("KGQA_MAX_TURNS_WITH_FRONTEND_REPAIR", "16")
    env["PYTHONUNBUFFERED"] = "1"

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for variant_name in args.variants:
        for split_name in args.splits:
            split_path = DEFAULT_SPLITS[split_name]
            if not split_path.exists():
                raise FileNotFoundError(f"Missing split file: {split_path}. Run prepare_splits.py first.")
            for repeat_idx in range(1, args.repeats + 1):
                label = f"{variant_name}_{split_name}_r{repeat_idx}"
                cmd = [
                    sys.executable,
                    str(ROOT / "scripts" / "run_skill_enhanced_test.py"),
                    "--data-path",
                    str(split_path),
                    "--train-data-path",
                    str(ROOT / "data" / "webqsp" / "webqsp_train.jsonl"),
                    "--skills-root",
                    str(args.skills_root),
                    "--variant",
                    args.variant,
                    "--max-turns",
                    str(args.max_turns),
                    "--skill-top-k",
                    str(args.skill_top_k),
                    "--max-concurrency",
                    str(args.concurrency),
                    "--label",
                    label,
                    "--output-dir",
                    str(output_root),
                    "--stage5-prompt-variant",
                    variant_name,
                ]
                print(f"\n=== Running {label} ===")
                print(" ".join(cmd))
                subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
