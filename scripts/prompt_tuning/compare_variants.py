#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "reports" / "skill_enhanced_test" / "prompt_tuning_v1"


def _load_results(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    rows = []
    for results_path in sorted(REPORT_ROOT.glob("*/results.json")):
        label = results_path.parent.name
        if "_r" not in label:
            continue
        parts = label.rsplit("_r", 1)[0].split("_")
        if len(parts) < 3:
            continue
        split = parts[-1]
        variant = "_".join(parts[:-1])
        data = _load_results(results_path)
        avg_f1 = mean(float(item.get("f1", 0.0)) for item in data) if data else 0.0
        hit05 = mean(1.0 if float(item.get("f1", 0.0)) >= 0.5 else 0.0 for item in data) if data else 0.0
        hit095 = mean(1.0 if float(item.get("f1", 0.0)) >= 0.95 else 0.0 for item in data) if data else 0.0
        fe = sum(int(item.get("frontend_errors", 0)) for item in data)
        turns = mean(int(item.get("turns", 0)) for item in data) if data else 0.0
        rows.append(
            {
                "variant": variant,
                "split": split,
                "label": label,
                "avg_f1": avg_f1,
                "hit05": hit05,
                "hit095": hit095,
                "frontend_errors": fe,
                "avg_turns": turns,
            }
        )

    grouped = {}
    for row in rows:
        grouped.setdefault((row["variant"], row["split"]), []).append(row)

    lines = [
        "# Prompt Variant Comparison",
        "",
        "| Variant | Split | Runs | Mean F1 | Mean Hit@0.5 | Mean Hit@0.95 | Mean FE | Mean Turns |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    summary = {}
    for (variant, split), items in sorted(grouped.items()):
        summary.setdefault(variant, {})[split] = {
            "mean_f1": mean(item["avg_f1"] for item in items),
            "mean_hit05": mean(item["hit05"] for item in items),
            "mean_hit095": mean(item["hit095"] for item in items),
            "mean_frontend_errors": mean(item["frontend_errors"] for item in items),
            "mean_turns": mean(item["avg_turns"] for item in items),
            "runs": len(items),
        }
        lines.append(
            f"| {variant} | {split} | {len(items)} | "
            f"{summary[variant][split]['mean_f1']:.4f} | "
            f"{summary[variant][split]['mean_hit05']:.4f} | "
            f"{summary[variant][split]['mean_hit095']:.4f} | "
            f"{summary[variant][split]['mean_frontend_errors']:.2f} | "
            f"{summary[variant][split]['mean_turns']:.2f} |"
        )

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    (REPORT_ROOT / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
