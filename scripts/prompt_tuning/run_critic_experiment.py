#!/usr/bin/env python3
"""Run critic experiments on consistency analysis reports.

Usage:
    python scripts/prompt_tuning/run_critic_experiment.py --critic <name> --report <path> [options]

Supported critics:
    strict-quantitative  Cross-checks every numeric claim against source CSV
    narrative-logic      Evaluates logical coherence of cause-effect chains
    toolcall-specialist  Analyzes tool-call divergence between consistency arms

Examples:
    python scripts/prompt_tuning/run_critic_experiment.py --critic strict-quantitative \
        --report reports/skill_enhanced_test/prompt_tuning_v1/full_decision_consistency_stability_judgment.md \
        --csv reports/skill_enhanced_test/prompt_tuning_v1/full_consistency_case_comparison.csv

    python scripts/prompt_tuning/run_critic_experiment.py --critic narrative-logic \
        --report reports/skill_enhanced_test/prompt_tuning_v1/full_decision_consistency_stability_judgment.md

    python scripts/prompt_tuning/run_critic_experiment.py --critic toolcall-specialist \
        --report reports/skill_enhanced_test/prompt_tuning_v1/full_decision_consistency_toolcall_analysis.md \
        --arm-a results/skill_enhanced_test/prompt_tuning_v1/arm_a_results.json \
        --arm-b results/skill_enhanced_test/prompt_tuning_v1/arm_b_results.json
"""

import argparse
import json
import sys
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "skills" / "consistency_critic_skills"


def load_critic_config(critic_name: str) -> dict:
    """Load the JSON config for a given critic."""
    # Map CLI names to filenames
    name_map = {
        "strict-quantitative": "strict_quantitative.json",
        "narrative-logic": "narrative_logic.json",
        "toolcall-specialist": "toolcall_specialist.json",
    }

    filename = name_map.get(critic_name)
    if not filename:
        print(f"Unknown critic: {critic_name}")
        print(f"Available: {', '.join(name_map.keys())}")
        sys.exit(1)

    config_path = SKILLS_DIR / filename
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        return json.load(f)


def build_prompt(config: dict, args) -> str:
    """Build the critic prompt from config template and CLI args."""
    template = config["invocation"]["prompt_template"]
    protocol = "\n".join(f"{i+1}. {step}" for i, step in enumerate(config["review_protocol"]))
    severity = "\n".join(f"- {level}: {desc}" for level, desc in config["severity_framework"].items())
    output = "\n".join(f"- {k}: {v}" for k, v in config["output_format"].items())

    replacements = {
        "{report_path}": args.report or "N/A",
        "{csv_path}": args.csv or "N/A",
        "{arm_a_results}": args.arm_a or "N/A",
        "{arm_b_results}": args.arm_b or "N/A",
        "{review_protocol}": protocol,
        "{severity_framework}": severity,
        "{output_format}": output,
    }

    result = template
    for key, value in replacements.items():
        result = result.replace(key, value)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run critic experiments on consistency reports")
    parser.add_argument("--critic", required=True, help="Critic name (strict-quantitative, narrative-logic, toolcall-specialist)")
    parser.add_argument("--report", required=True, help="Path to the report file to review")
    parser.add_argument("--csv", help="Path to source CSV (for strict-quantitative)")
    parser.add_argument("--arm-a", help="Path to Arm A results JSON (for toolcall-specialist)")
    parser.add_argument("--arm-b", help="Path to Arm B results JSON (for toolcall-specialist)")
    parser.add_argument("--output", help="Output file path (default: stdout)")
    parser.add_argument("--agent-type", default="oh-my-claudecode:critic",
                        help="Agent type for invocation")

    args = parser.parse_args()

    # Load critic config
    config = load_critic_config(args.critic)
    print(f"Critic: {config['name']}", file=sys.stderr)
    print(f"Description: {config['description']}", file=sys.stderr)

    # Build prompt
    prompt = build_prompt(config, args)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(prompt)
        print(f"Prompt written to {output_path}", file=sys.stderr)
        print("Run with an appropriate agent to get the review.", file=sys.stderr)
    else:
        print(prompt)


if __name__ == "__main__":
    main()
