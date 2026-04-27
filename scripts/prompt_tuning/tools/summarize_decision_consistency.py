#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "skill_enhanced_test" / "prompt_tuning_v1"


def load_results(label: str) -> List[Dict[str, Any]]:
    path = OUT / label / "results.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    return {
        "avg_f1": mean(r["f1"] for r in rows) if rows else 0.0,
        "hit1": mean(1.0 if r["f1"] >= 0.5 else 0.0 for r in rows) if rows else 0.0,
        "em": mean(1.0 if r["f1"] >= 0.95 else 0.0 for r in rows) if rows else 0.0,
        "cons_used_cases": mean(
            1.0 if r.get("consistency", {}).get("used_turns", 0) > 0 else 0.0 for r in rows
        )
        if rows
        else 0.0,
        "cons_disagreement_turns": mean(
            float(r.get("consistency", {}).get("disagreement_turns", 0)) for r in rows
        )
        if rows
        else 0.0,
    }


def fmt(m: Dict[str, float]) -> str:
    return (
        f"Avg F1={m['avg_f1']:.4f}, Hit@1={m['hit1']:.4f}, EM={m['em']:.4f}, "
        f"cons_used={m['cons_used_cases']:.4f}, cons_disagree_turns={m['cons_disagreement_turns']:.4f}"
    )


def main() -> int:
    old_bad_a = metrics(load_results("decision_consistency_old_bad100_armA"))
    old_bad_b = metrics(load_results("decision_consistency_old_bad100_armB"))

    diag_a_runs = [metrics(load_results(f"decision_consistency_diag12_armA_r{i}")) for i in range(1, 4)]
    diag_b_runs = [metrics(load_results(f"decision_consistency_diag12_armB_r{i}")) for i in range(1, 4)]

    diag_a = {k: mean(run[k] for run in diag_a_runs) for k in diag_a_runs[0]}
    diag_b = {k: mean(run[k] for run in diag_b_runs) for k in diag_b_runs[0]}

    lines = [
        "# Decision Consistency Summary",
        "",
        "## old_bad_100",
        "",
        f"- Arm A (no consistency): {fmt(old_bad_a)}",
        f"- Arm B (consistency): {fmt(old_bad_b)}",
        f"- Delta (B - A): Avg F1={old_bad_b['avg_f1'] - old_bad_a['avg_f1']:.4f}, "
        f"Hit@1={old_bad_b['hit1'] - old_bad_a['hit1']:.4f}, EM={old_bad_b['em'] - old_bad_a['em']:.4f}",
        "",
        "## diag12 mean over 3 repeats",
        "",
        f"- Arm A (no consistency): {fmt(diag_a)}",
        f"- Arm B (consistency): {fmt(diag_b)}",
        f"- Delta (B - A): Avg F1={diag_b['avg_f1'] - diag_a['avg_f1']:.4f}, "
        f"Hit@1={diag_b['hit1'] - diag_a['hit1']:.4f}, EM={diag_b['em'] - diag_a['em']:.4f}",
        "",
    ]
    out_path = OUT / "decision_consistency_summary.md"
    out_path.write_text("\n".join(lines))
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
