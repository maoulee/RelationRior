#!/usr/bin/env python3
"""
Experiment Agent — unified entry point for KGQA experiments.

Subcommands:
  run          Batch experiment (wraps run_skill_enhanced_test.py)
  analyze      Deep single-case or multi-case analysis with RL reward breakdown
  compare      Side-by-side comparison of two experiment runs
  score        Retroactively score an existing results.json with RL rewards
  cases        List/filter cases from a results file by stage or score criteria

All scores use the framework's own reward functions (plug_v10 / rl_reward_scorer).

Examples:
  # Run 100 bad-case experiment
  python scripts/experiment_agent.py run \
    --data-path tmp/prompt_tuning/old_bad_100.jsonl \
    --label skill_v3_bad_100

  # Analyze specific case
  python scripts/experiment_agent.py analyze \
    --results reports/.../results.json \
    --case-id WebQTest-1234

  # Compare two runs
  python scripts/experiment_agent.py compare \
    reports/.../baseline/results.json \
    reports/.../experiment/results.json

  # Score existing results
  python scripts/experiment_agent.py score \
    --results reports/.../results.json

  # List cases where plan F1 dropped
  python scripts/experiment_agent.py cases \
    --results reports/.../results.json \
    --where "plan_f1<0.3" --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Lazy imports — only load heavy modules when the subcommand needs them
# ---------------------------------------------------------------------------

def _import_reward_functions():
    """Import reward computation from rl_reward_scorer."""
    from scripts.prompt_tuning.tools.rl_reward_scorer import (
        compute_all_rewards,
        compute_f1_reward,
        compute_plan_reward,
        compute_action_reward,
        compute_reasoning_reward,
        compute_behavior_reward,
        compute_compliance_reward,
        compute_lazy_reward,
        compute_env_reward,
        reconstruct_rollout_info,
        REWARD_NAMES,
    )
    return dict(
        compute_all_rewards=compute_all_rewards,
        compute_f1_reward=compute_f1_reward,
        compute_plan_reward=compute_plan_reward,
        compute_action_reward=compute_action_reward,
        compute_reasoning_reward=compute_reasoning_reward,
        compute_behavior_reward=compute_behavior_reward,
        compute_compliance_reward=compute_compliance_reward,
        compute_lazy_reward=compute_lazy_reward,
        compute_env_reward=compute_env_reward,
        reconstruct_rollout_info=reconstruct_rollout_info,
        REWARD_NAMES=REWARD_NAMES,
    )


def _import_f1():
    """Import calculate_f1 / normalize."""
    try:
        from plug_v10 import calculate_f1, normalize, parse_boxed_answers
        return calculate_f1, normalize, parse_boxed_answers
    except ImportError:
        pass
    try:
        from subgraph_kgqa.rl.plugin import calculate_f1
        def normalize(s: str) -> str:
            if not isinstance(s, str):
                s = str(s)
            s = s.lower().strip()
            s = re.sub(r"\b(a|an|the)\b", " ", s)
            s = "".join(ch for ch in s if ch.isalnum() or ch in " .-_")
            return " ".join(s.split())
        def parse_boxed_answers(text: str) -> List[str]:
            text_without_reason = re.sub(
                r'<reason>.*?</reason>', '', text, flags=re.DOTALL | re.IGNORECASE)
            return [a.strip() for a in
                    re.findall(r"\\boxed\{([^}]*)\}", text_without_reason)
                    if a.strip()]
        return calculate_f1, normalize, parse_boxed_answers
    except ImportError:
        raise ImportError("Cannot import calculate_f1 from plug_v10 or rl.plugin")


# ===========================================================================
# Helpers
# ===========================================================================

def _load_results(path: str) -> List[Dict]:
    with open(path) as f:
        data = json.load(f)
    return data


def _save_json(obj, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def _print_header(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ===========================================================================
# Stage score extraction
# ===========================================================================

def _stage_f1(result: Dict, stage: str) -> float:
    ss = result.get("stage_scores", {})
    nested = ss.get(stage, {})
    if isinstance(nested, dict) and "f1" in nested:
        return nested["f1"]
    return ss.get(f"{stage}_f1", 0.0)


# ===========================================================================
# SUBCOMMAND: run
# ===========================================================================

def cmd_run(args: argparse.Namespace) -> int:
    """Launch a batch experiment via run_skill_enhanced_test.py."""
    cmd_parts = [
        sys.executable,
        str(ROOT / "scripts" / "run_skill_enhanced_test.py"),
        "--data-path", str(args.data_path),
        "--train-data-path", str(args.train_data_path),
        "--skills-root", str(args.skills_root),
        "--max-concurrency", str(args.max_concurrency),
        "--skill-top-k", str(args.skill_top_k),
    ]
    if args.label:
        cmd_parts += ["--label", args.label]
    if args.limit_cases:
        cmd_parts += ["--limit-cases", str(args.limit_cases)]
    if args.case_id:
        for cid in args.case_id:
            cmd_parts += ["--case-id", cid]
    if args.max_turns:
        cmd_parts += ["--max-turns", str(args.max_turns)]
    if args.variant:
        cmd_parts += ["--variant", args.variant]
    if args.stage5_prompt_variant:
        cmd_parts += ["--stage5-prompt-variant", args.stage5_prompt_variant]
    if args.no_skills:
        cmd_parts.append("--no-skills")
    if args.output_dir:
        cmd_parts += ["--output-dir", args.output_dir]
    if args.negative_skills_root:
        cmd_parts += ["--negative-skills-root", args.negative_skills_root]

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'src'}"

    print(f"[experiment_agent] Launching: {' '.join(cmd_parts)}")
    return os.execvpe(sys.executable, cmd_parts, env)


# ===========================================================================
# SUBCOMMAND: score
# ===========================================================================

def cmd_score(args: argparse.Namespace) -> int:
    """Retroactively compute all RL reward scores for an existing results.json."""
    rewards_mod = _import_reward_functions()
    compute_all_rewards = rewards_mod["compute_all_rewards"]
    REWARD_NAMES = rewards_mod["REWARD_NAMES"]

    results = _load_results(args.results)
    _print_header(f"RL Reward Scoring: {len(results)} cases")

    scored = []
    for i, case in enumerate(results):
        cid = case.get("case_id", f"case_{i}")
        try:
            rewards = compute_all_rewards(case)
        except Exception as exc:
            print(f"  [{i+1}/{len(results)}] {cid}: ERROR - {exc}")
            rewards = {}
        scored.append({"case_id": cid, "rewards": rewards})
        if (i + 1) % 50 == 0 or i == len(results) - 1:
            print(f"  [{i+1}/{len(results)}] {cid}: scored")

    # Aggregate
    reward_avgs: Dict[str, float] = {}
    for name in REWARD_NAMES:
        vals = [s["rewards"].get(name, {}).get("score", 0.0) for s in scored if name in s.get("rewards", {})]
        reward_avgs[name] = mean(vals) if vals else 0.0

    # Print summary
    print("\n--- Reward Averages ---")
    print(f"{'Reward':<15s} {'Avg':>8s} {'Median':>8s} {'Std':>8s}")
    print("-" * 42)
    for name in REWARD_NAMES:
        vals = [s["rewards"].get(name, {}).get("score", 0.0) for s in scored if name in s.get("rewards", {})]
        if vals:
            avg = mean(vals)
            med = median(vals)
            sd = stdev(vals) if len(vals) > 1 else 0.0
            print(f"{name:<15s} {avg:>8.4f} {med:>8.4f} {sd:>8.4f}")

    # Also print overall F1/Hit1
    f1_vals = [c.get("f1", 0.0) for c in results]
    hit1_vals = [c.get("hit_at_1", 0.0) for c in results]
    print(f"\n{'F1':<15s} {mean(f1_vals):>8.4f} {median(f1_vals):>8.4f}")
    print(f"{'Hit@1':<15s} {mean(hit1_vals):>8.4f} {median(hit1_vals):>8.4f}")

    # Save
    out_dir = Path(args.results).parent
    out_path = out_dir / "rl_reward_scores.json"
    _save_json(scored, str(out_path))
    print(f"\nSaved: {out_path}")

    # Markdown summary
    _save_reward_summary_md(scored, REWARD_NAMES, results, out_dir)
    return 0


def _save_reward_summary_md(
    scored: List[Dict],
    reward_names: List[str],
    results: List[Dict],
    out_dir: Path,
) -> None:
    lines = ["# RL Reward Score Summary", ""]
    lines.append("| Reward | Avg | Median | Std |")
    lines.append("|--------|-----|--------|-----|")
    for name in reward_names:
        vals = [s["rewards"].get(name, {}).get("score", 0.0) for s in scored if name in s.get("rewards", {})]
        if vals:
            avg = mean(vals)
            med = median(vals)
            sd = stdev(vals) if len(vals) > 1 else 0.0
            lines.append(f"| {name} | {avg:.4f} | {med:.4f} | {sd:.4f} |")
    f1_vals = [c.get("f1", 0.0) for c in results]
    hit1_vals = [c.get("hit_at_1", 0.0) for c in results]
    lines.append(f"| **F1** | **{mean(f1_vals):.4f}** | **{median(f1_vals):.4f}** | - |")
    lines.append(f"| **Hit@1** | **{mean(hit1_vals):.4f}** | **{median(hit1_vals):.4f}** | - |")
    lines.append("")
    Path(out_dir / "rl_reward_summary.md").write_text("\n".join(lines))
    print(f"Saved: {out_dir / 'rl_reward_summary.md'}")


# ===========================================================================
# SUBCOMMAND: analyze
# ===========================================================================

def cmd_analyze(args: argparse.Namespace) -> int:
    """Deep analysis of one or more cases."""
    rewards_mod = _import_reward_functions()
    compute_all_rewards = rewards_mod["compute_all_rewards"]
    reconstruct_rollout_info = rewards_mod["reconstruct_rollout_info"]
    REWARD_NAMES = rewards_mod["REWARD_NAMES"]

    results = _load_results(args.results)
    result_map = {r["case_id"]: r for r in results}

    # Select cases
    targets = []
    if args.case_id:
        for cid in args.case_id:
            if cid in result_map:
                targets.append(result_map[cid])
            else:
                print(f"WARNING: {cid} not found in results")
    elif args.worst:
        targets = sorted(results, key=lambda x: x.get("f1", 0.0))[:args.worst]
    elif args.best:
        targets = sorted(results, key=lambda x: x.get("f1", 0.0), reverse=True)[:args.best]
    elif args.random:
        import random as rnd
        targets = rnd.sample(results, min(args.random, len(results)))
    else:
        targets = sorted(results, key=lambda x: x.get("f1", 0.0))[:10]

    for case in targets:
        _print_case_analysis(case, rewards_mod, verbose=args.verbose)

    # Aggregate summary if multiple cases
    if len(targets) > 1:
        _print_header("Aggregate Summary for Selected Cases")
        _print_stage_summary_table(targets)
        _print_reward_aggregate(targets, REWARD_NAMES, rewards_mod)

    return 0


def _print_case_analysis(case: Dict, rewards_mod: Dict, verbose: bool = False) -> None:
    """Print full case analysis."""
    cid = case.get("case_id", "?")
    question = case.get("question", "?")[:80]
    f1_val = case.get("f1", 0.0)
    hit1 = case.get("hit_at_1", 0.0)
    predicted = case.get("predicted", [])
    gt = case.get("ground_truth", [])
    turns = case.get("turns", 0)
    fe = case.get("frontend_errors", 0)

    _print_header(f"Case: {cid}")
    print(f"Question:    {question}")
    print(f"F1:          {f1_val:.4f}")
    print(f"Hit@1:       {hit1:.4f}")
    print(f"Predicted:   {predicted[:5]}")
    print(f"Ground Truth:{gt[:5]}")
    print(f"Turns:       {turns}  |  Frontend Errors: {fe}")

    # Stage scores
    stage_scores = case.get("stage_scores", {})
    print(f"\n--- Stage Scores ---")
    for stage in ["entity_linking", "plan", "action", "filter", "final_reasoning"]:
        ss = stage_scores.get(stage, {})
        if isinstance(ss, dict):
            print(f"  {stage:<20s}  P={ss.get('precision',0):.3f}  R={ss.get('recall',0):.3f}  F1={ss.get('f1',0):.3f}")

    # RL Rewards
    try:
        rewards = rewards_mod["compute_all_rewards"](case)
        print(f"\n--- RL Reward Breakdown ---")
        for name in rewards_mod["REWARD_NAMES"]:
            r = rewards.get(name, {})
            score = r.get("score", 0.0)
            detail_str = ""
            if name == "plan":
                detail_str = f"  f1_gen={r.get('f1_generator',0):.3f}  coverage={r.get('coverage',0):.3f}  useful={r.get('useful_generators',0)}/{r.get('total_generators',0)}  valid_actions={r.get('valid_actions',0)}/{r.get('total_actions',0)}"
            elif name == "action":
                detail_str = f"  f1_action={r.get('f1_action',0):.3f}  coverage={r.get('coverage',0):.3f}  valid_exec={r.get('valid_executed',0)}/{r.get('total_executed',0)}  reason={r.get('reason','')}"
            elif name == "reasoning":
                detail_str = f"  r_exec={r.get('r_exec',0):.3f}  f1_final={r.get('f1_final',0):.3f}  compress={r.get('compression',0):.3f}  pool={r.get('pool_size',0)} -> answer={r.get('answer_size',0)}"
            elif name == "behavior":
                violations = r.get("violations", [])
                detail_str = f"  violations={violations}  turns={r.get('turn_count',0)}  queries={r.get('total_queries',0)}"
            elif name == "compliance":
                detail_str = f"  details={r.get('details', [])[:3]}"
            elif name == "env":
                detail_str = f"  success={r.get('tool_successes',0)}  fail={r.get('tool_failures',0)}  total={r.get('total_tools',0)}"
            elif name == "lazy":
                detail_str = f"  early_answer={r.get('early_answer',False)}  insufficient={r.get('insufficient_answer',False)}"
            print(f"  {name:<12s}  score={score:>8.4f}{detail_str}")
    except Exception as exc:
        print(f"  Reward computation error: {exc}")

    # Turn-by-turn summary
    if verbose:
        trajectory = case.get("trajectory", [])
        if trajectory:
            print(f"\n--- Turn-by-Turn Trajectory ({len(trajectory)} turns) ---")
            for turn in trajectory:
                turn_num = turn.get("turn", "?")
                parsed = turn.get("parsed_output", {})
                queries = parsed.get("queries", [])
                candidates = parsed.get("candidates", [])
                final_answer = parsed.get("final_answer", [])
                fe_turn = turn.get("frontend_errors", [])

                print(f"\n  Turn {turn_num}:")
                for q in queries:
                    tool = q.get("tool_name", "?")
                    args = q.get("arguments", {})
                    if tool == "plan":
                        rels = args.get("related", []) + args.get("maybe_related", [])
                        print(f"    plan: related={rels[:5]}  constraint={args.get('constraint_relations', [])[:3]}")
                    elif tool == "action":
                        anchor = args.get("anchor", "?")
                        path = args.get("path", [])
                        rels = [p.get("relation", "?") for p in path if isinstance(p, dict)]
                        print(f"    action: anchor={anchor[:30]}  path={rels}")
                    else:
                        print(f"    {tool}: {json.dumps(args, ensure_ascii=False)[:100]}")

                if candidates:
                    print(f"    candidates: {candidates[:5]}")
                if final_answer:
                    print(f"    FINAL ANSWER: {final_answer}")
                if fe_turn:
                    for e in fe_turn:
                        print(f"    FE: [{e.get('code','')}] {e.get('message','')[:80]}")

                # Backend results summary
                backend_results = turn.get("backend_results", [])
                for br in backend_results:
                    status = br.get("status", "")
                    tool = br.get("tool_name", "?")
                    resp = br.get("response_text", "")[:150].replace("\n", " ")
                    icon = "OK" if "SUCCESS" in status else "FAIL"
                    print(f"    [{icon}] {tool}: {resp}...")

    # Skill bundle info
    skill_bundle = case.get("skill_bundle", {})
    retrieved = skill_bundle.get("retrieved_case_ids", [])
    if retrieved:
        print(f"\n--- Skill Bundle ---")
        print(f"  Retrieved: {retrieved[:5]}")
        print(f"  Shortlisted: {skill_bundle.get('shortlisted_case_ids', [])[:5]}")
        if skill_bundle.get("aggregated_skill"):
            agg = skill_bundle["aggregated_skill"]
            print(f"  Aggregated: answer_type={agg.get('answer_type_guidance','')[:50]}")
            print(f"  Pitfalls: {agg.get('pitfalls','')[:100]}")
        if skill_bundle.get("audit_agent"):
            audit = skill_bundle["audit_agent"]
            print(f"  Audit: passed={audit.get('passed')} issues={audit.get('issues',[])}")


def _print_stage_summary_table(cases: List[Dict]) -> None:
    """Print stage score summary table."""
    print(f"\n{'Stage':<20s} {'Avg F1':>8s} {'Avg P':>8s} {'Avg R':>8s} {'Median':>8s}")
    print("-" * 55)
    for stage in ["entity_linking", "plan", "action", "filter", "final_reasoning"]:
        f1_vals, p_vals, r_vals = [], [], []
        for c in cases:
            ss = c.get("stage_scores", {}).get(stage, {})
            if isinstance(ss, dict):
                f1_vals.append(ss.get("f1", 0.0))
                p_vals.append(ss.get("precision", 0.0))
                r_vals.append(ss.get("recall", 0.0))
        if f1_vals:
            print(f"  {stage:<18s} {mean(f1_vals):>8.4f} {mean(p_vals):>8.4f} {mean(r_vals):>8.4f} {median(f1_vals):>8.4f}")

    # Overall
    f1_vals = [c.get("f1", 0.0) for c in cases]
    hit1_vals = [c.get("hit_at_1", 0.0) for c in cases]
    print(f"\n  {'OVERALL F1':<18s} {mean(f1_vals):>8.4f}")
    print(f"  {'Hit@1':<18s} {mean(hit1_vals):>8.4f}")


def _print_reward_aggregate(cases: List[Dict], reward_names: List[str], rewards_mod: Dict) -> None:
    """Print RL reward aggregate for a list of cases."""
    print(f"\n--- RL Reward Aggregate (n={len(cases)}) ---")
    all_rewards = []
    for c in cases:
        try:
            r = rewards_mod["compute_all_rewards"](c)
            all_rewards.append(r)
        except Exception:
            pass

    print(f"{'Reward':<15s} {'Avg':>8s} {'Median':>8s} {'Min':>8s} {'Max':>8s}")
    print("-" * 50)
    for name in reward_names:
        vals = [r.get(name, {}).get("score", 0.0) for r in all_rewards if name in r]
        if vals:
            print(f"  {name:<13s} {mean(vals):>8.4f} {median(vals):>8.4f} {min(vals):>8.4f} {max(vals):>8.4f}")


# ===========================================================================
# SUBCOMMAND: compare
# ===========================================================================

def cmd_compare(args: argparse.Namespace) -> int:
    """Compare two experiment runs with stage-level attribution."""
    from scripts.compare_test_runs import compare_runs

    report = compare_runs(
        args.baseline,
        args.experiment,
        output_path=args.output,
        show_hints=args.show_hints,
        top_n=args.top_n,
    )

    if not args.output:
        print(report)

    # If --score flag, also compute RL reward comparison
    if args.score:
        rewards_mod = _import_reward_functions()
        REWARD_NAMES = rewards_mod["REWARD_NAMES"]
        baseline = _load_results(args.baseline)
        experiment = _load_results(args.experiment)
        b_map = {r["case_id"]: r for r in baseline}
        e_map = {r["case_id"]: r for r in experiment}
        common = sorted(set(b_map.keys()) & set(e_map.keys()))

        _print_header(f"RL Reward Comparison ({len(common)} common cases)")
        print(f"{'Reward':<15s} {'Base Avg':>9s} {'Exp Avg':>9s} {'Delta':>9s}")
        print("-" * 45)
        for name in REWARD_NAMES:
            b_vals, e_vals = [], []
            for cid in common:
                try:
                    br = rewards_mod["compute_all_rewards"](b_map[cid])
                    b_vals.append(br.get(name, {}).get("score", 0.0))
                except Exception:
                    b_vals.append(0.0)
                try:
                    er = rewards_mod["compute_all_rewards"](e_map[cid])
                    e_vals.append(er.get(name, {}).get("score", 0.0))
                except Exception:
                    e_vals.append(0.0)
            if b_vals:
                b_avg = mean(b_vals)
                e_avg = mean(e_vals)
                delta = e_avg - b_avg
                marker = " **" if abs(delta) > 0.02 else ""
                print(f"  {name:<13s} {b_avg:>9.4f} {e_avg:>9.4f} {delta:>+9.4f}{marker}")

    return 0


# ===========================================================================
# SUBCOMMAND: cases — filter / list cases from results
# ===========================================================================

def cmd_cases(args: argparse.Namespace) -> int:
    """List and filter cases from a results file."""
    results = _load_results(args.results)

    # Parse --where conditions
    conditions = []
    if args.where:
        for cond in args.where:
            conditions.append(_parse_condition(cond))

    filtered = results
    for cond in conditions:
        filtered = [r for r in filtered if cond(r)]

    if args.sort_by:
        reverse = args.sort_by.startswith("-")
        key = args.sort_by.lstrip("-")
        filtered.sort(key=lambda r: _extract_field(r, key), reverse=reverse)

    limit = args.limit or len(filtered)
    filtered = filtered[:limit]

    # Output
    if args.format == "json":
        _save_json(filtered, args.output or "/dev/stdout")
    elif args.format == "ids":
        for r in filtered:
            print(r.get("case_id", ""))
    else:
        # Table
        print(f"{'Case ID':<20s} {'F1':>6s} {'Hit1':>6s} {'Plan':>6s} {'Action':>6s} {'Reason':>7s} {'Turns':>6s} {'FE':>4s} {'Question':<40s}")
        print("-" * 105)
        for r in filtered:
            cid = r.get("case_id", "?")
            f1 = r.get("f1", 0.0)
            hit1 = r.get("hit_at_1", 0.0)
            plan_f1 = _stage_f1(r, "plan")
            action_f1 = _stage_f1(r, "action")
            reason_f1 = _stage_f1(r, "final_reasoning")
            turns = r.get("turns", 0)
            fe = r.get("frontend_errors", 0)
            q = r.get("question", "")[:40]
            print(f"{cid:<20s} {f1:>6.2f} {hit1:>6.2f} {plan_f1:>6.2f} {action_f1:>6.2f} {reason_f1:>7.2f} {turns:>6d} {fe:>4d} {q:<40s}")

    print(f"\nTotal: {len(filtered)} cases")
    return 0


def _parse_condition(cond_str: str):
    """Parse a condition like 'plan_f1<0.3' or 'f1>=0.8'."""
    match = re.match(r'(\w+)\s*(>=|<=|!=|>|<|=)\s*([0-9.-]+)', cond_str)
    if not match:
        raise ValueError(f"Invalid condition: {cond_str}")
    field, op, val = match.group(1), match.group(2), float(match.group(3))

    ops = {
        ">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        "=": lambda a, b: abs(a - b) < 1e-9,
        "!=": lambda a, b: abs(a - b) >= 1e-9,
    }
    op_fn = ops[op]

    def check(result: Dict) -> bool:
        v = _extract_field(result, field)
        return op_fn(v, val)

    return check


def _extract_field(result: Dict, field: str) -> float:
    """Extract numeric field from result, supporting stage_score paths."""
    if field in ("f1", "hit_at_1", "turns", "frontend_errors"):
        return float(result.get(field, 0.0))
    # plan_f1 -> stage_scores.plan.f1
    for stage in ["entity_linking", "plan", "action", "filter", "final_reasoning"]:
        if field == f"{stage}_f1":
            return _stage_f1(result, stage)
        if field == f"{stage}_precision":
            ss = result.get("stage_scores", {}).get(stage, {})
            return ss.get("precision", 0.0) if isinstance(ss, dict) else 0.0
        if field == f"{stage}_recall":
            ss = result.get("stage_scores", {}).get(stage, {})
            return ss.get("recall", 0.0) if isinstance(ss, dict) else 0.0
    return 0.0


# ===========================================================================
# SUBCOMMAND: summary — quick aggregate stats
# ===========================================================================

def cmd_summary(args: argparse.Namespace) -> int:
    """Print aggregate stats for one or more results files."""
    for path in args.results:
        results = _load_results(path)
        label = Path(path).parent.name

        f1_vals = [r.get("f1", 0.0) for r in results]
        hit1_vals = [r.get("hit_at_1", 0.0) for r in results]
        plan_f1 = [_stage_f1(r, "plan") for r in results]
        action_f1 = [_stage_f1(r, "action") for r in results]
        reason_f1 = [_stage_f1(r, "final_reasoning") for r in results]
        turns_vals = [r.get("turns", 0) for r in results]
        fe_vals = [r.get("frontend_errors", 0) for r in results]

        _print_header(f"Summary: {label}")
        print(f"  Cases:    {len(results)}")
        print(f"  F1:       {mean(f1_vals):.4f}  (median={median(f1_vals):.4f})")
        print(f"  Hit@1:    {mean(hit1_vals):.4f}")
        print(f"  Hit@0.8:  {mean(1.0 if v >= 0.8 else 0.0 for v in f1_vals):.4f}")
        print(f"  Exact:    {mean(1.0 if v >= 0.95 else 0.0 for v in f1_vals):.4f}")
        print(f"  Plan F1:  {mean(plan_f1):.4f}")
        print(f"  Action F1:{mean(action_f1):.4f}")
        print(f"  Reason F1:{mean(reason_f1):.4f}")
        print(f"  Avg Turns:{mean(turns_vals):.2f}")
        print(f"  Total FE: {sum(fe_vals)}")

    return 0


# ===========================================================================
# Main
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment Agent — unified KGQA experiment entry point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", help="Subcommand")

    # --- run ---
    p_run = sub.add_parser("run", help="Launch batch experiment")
    p_run.add_argument("--data-path", required=True)
    p_run.add_argument("--train-data-path", default="data/webqsp/webqsp_train.jsonl")
    p_run.add_argument("--skills-root", default="skills/webqsp_train_case_skills_v3")
    p_run.add_argument("--label", default=None)
    p_run.add_argument("--limit-cases", type=int, default=None)
    p_run.add_argument("--case-id", action="append", default=[])
    p_run.add_argument("--max-turns", type=int, default=8)
    p_run.add_argument("--max-concurrency", type=int, default=32)
    p_run.add_argument("--skill-top-k", type=int, default=3)
    p_run.add_argument("--variant", default="original")
    p_run.add_argument("--stage5-prompt-variant", default=None)
    p_run.add_argument("--no-skills", action="store_true")
    p_run.add_argument("--output-dir", default=None)
    p_run.add_argument("--negative-skills-root", default=None)

    # --- score ---
    p_score = sub.add_parser("score", help="Retroactively score results with RL rewards")
    p_score.add_argument("--results", required=True, help="Path to results.json")

    # --- analyze ---
    p_analyze = sub.add_parser("analyze", help="Deep case analysis")
    p_analyze.add_argument("--results", required=True, help="Path to results.json")
    p_analyze.add_argument("--case-id", action="append", default=[], help="Specific case IDs")
    p_analyze.add_argument("--worst", type=int, default=None, help="Analyze N worst cases")
    p_analyze.add_argument("--best", type=int, default=None, help="Analyze N best cases")
    p_analyze.add_argument("--random", type=int, default=None, help="Analyze N random cases")
    p_analyze.add_argument("-v", "--verbose", action="store_true", help="Show turn-by-turn details")

    # --- compare ---
    p_cmp = sub.add_parser("compare", help="Compare two experiment runs")
    p_cmp.add_argument("baseline", help="Path to baseline results.json")
    p_cmp.add_argument("experiment", help="Path to experiment results.json")
    p_cmp.add_argument("-o", "--output", default=None, help="Output markdown path")
    p_cmp.add_argument("--show-hints", action="store_true")
    p_cmp.add_argument("--top-n", type=int, default=20)
    p_cmp.add_argument("--score", action="store_true", help="Also compute RL reward comparison")

    # --- cases ---
    p_cases = sub.add_parser("cases", help="List/filter cases")
    p_cases.add_argument("--results", required=True)
    p_cases.add_argument("--where", action="append", default=[], help="Filter conditions, e.g. 'plan_f1<0.3'")
    p_cases.add_argument("--sort-by", default="-f1", help="Sort field (prefix - for desc)")
    p_cases.add_argument("--limit", type=int, default=None)
    p_cases.add_argument("--format", choices=["table", "json", "ids"], default="table")
    p_cases.add_argument("--output", default=None)

    # --- summary ---
    p_sum = sub.add_parser("summary", help="Quick aggregate stats")
    p_sum.add_argument("results", nargs="+", help="One or more results.json paths")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    dispatch = {
        "run": cmd_run,
        "score": cmd_score,
        "analyze": cmd_analyze,
        "compare": cmd_compare,
        "cases": cmd_cases,
        "summary": cmd_summary,
    }

    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
