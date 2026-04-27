#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]


def _load_wrong_case_ids(results_path: Path, threshold: float) -> List[str]:
    records = json.loads(results_path.read_text())
    wrong = [
        str(record.get("case_id", ""))
        for record in records
        if str(record.get("case_id", "")).strip() and float(record.get("f1", 0.0) or 0.0) < threshold
    ]
    if not wrong:
        raise ValueError(f"No wrong cases found in {results_path} with threshold={threshold}")
    return wrong


def _run(cmd: List[str], *, env: dict[str, str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate actionspace_v1 train skills and re-evaluate wrong WebQSP test cases repeatedly."
    )
    parser.add_argument(
        "--source-results",
        default="reports/skill_enhanced_test/fulltest_skill_top3_protocol_guard_20260330/results.json",
        help="Reference results.json used to define the wrong-case set",
    )
    parser.add_argument("--wrong-threshold", type=float, default=0.999999)
    parser.add_argument("--data-path", default="data/webqsp/webqsp_test.jsonl")
    parser.add_argument("--train-data-path", default="data/webqsp/webqsp_train.jsonl")
    parser.add_argument("--skills-root", default="skills")
    parser.add_argument("--output-subdir", default="webqsp_train_case_skills_actionspace_v1")
    parser.add_argument("--variant", default="protocol_guard_action_id_experiment")
    parser.add_argument("--skill-top-k", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--build-concurrency", type=int, default=32)
    parser.add_argument("--test-concurrency", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--label-prefix", default="actionspace_v1_wrong600_repeat")
    parser.add_argument("--reasoning-injection-mode", default="per_skill", choices=["aggregate", "per_skill"])
    parser.add_argument("--skip-regen", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    source_results = ROOT / args.source_results
    wrong_case_ids = _load_wrong_case_ids(source_results, args.wrong_threshold)
    print(
        json.dumps(
            {
                "source_results": str(source_results),
                "wrong_case_count": len(wrong_case_ids),
                "first_cases": wrong_case_ids[:20],
                "reasoning_injection_mode": args.reasoning_injection_mode,
                "build_concurrency": args.build_concurrency,
                "test_concurrency": args.test_concurrency,
                "repeats": args.repeats,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    env = os.environ.copy()
    env["KGQA_SKILL_REASONING_INJECTION_MODE"] = args.reasoning_injection_mode

    if not args.skip_regen:
        build_cmd = [
            sys.executable,
            "scripts/build_full_train_case_skills.py",
            "--data-path",
            args.train_data_path,
            "--skills-root",
            args.skills_root,
            "--output-subdir",
            args.output_subdir,
            "--concurrency",
            str(args.build_concurrency),
        ]
        if args.skip_llm:
            build_cmd.append("--skip-llm")
        _run(build_cmd, env=env)

    resolved_skills_root = str((ROOT / args.skills_root / args.output_subdir).resolve())
    for repeat_idx in range(1, max(1, args.repeats) + 1):
        label = f"{args.label_prefix}_r{repeat_idx}"
        test_cmd = [
            sys.executable,
            "scripts/run_skill_enhanced_test.py",
            "--data-path",
            args.data_path,
            "--train-data-path",
            args.train_data_path,
            "--skills-root",
            resolved_skills_root,
            "--variant",
            args.variant,
            "--skill-top-k",
            str(args.skill_top_k),
            "--max-turns",
            str(args.max_turns),
            "--max-concurrency",
            str(args.test_concurrency),
            "--label",
            label,
        ]
        for case_id in wrong_case_ids:
            test_cmd.extend(["--case-id", case_id])
        _run(test_cmd, env=env)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
