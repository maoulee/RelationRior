#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[2]
TEST_DATA = ROOT / "data" / "webqsp" / "webqsp_test.jsonl"
OLD_RESULTS = ROOT / "reports" / "skill_enhanced_test" / "fulltest_skill_top3_protocol_guard_20260330" / "results.json"
WRONG600_RUNS = [
    ROOT / "reports" / "skill_enhanced_test" / "actionspace_v1_wrong600_per_skill_r1" / "results.json",
    ROOT / "reports" / "skill_enhanced_test" / "actionspace_v1_wrong600_per_skill_r2" / "results.json",
    ROOT / "reports" / "skill_enhanced_test" / "actionspace_v1_wrong600_per_skill_r3" / "results.json",
]
OUT_DIR = ROOT / "tmp" / "prompt_tuning"


def _load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open() as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    rng_good = random.Random(42)
    rng_bad = random.Random(43)

    dataset_rows = _load_jsonl(TEST_DATA)
    by_id = {str(row["id"]): row for row in dataset_rows}

    old_results = json.loads(OLD_RESULTS.read_text(encoding="utf-8"))
    old_good_ids = sorted(str(row["case_id"]) for row in old_results if float(row.get("f1", 0.0)) >= 0.95)
    old_bad_ids = sorted(str(row["case_id"]) for row in old_results if float(row.get("f1", 0.0)) < 1.0)

    run_scores: List[Dict[str, float]] = []
    for path in WRONG600_RUNS:
        data = json.loads(path.read_text(encoding="utf-8"))
        run_scores.append({str(row["case_id"]): float(row.get("f1", 0.0)) for row in data})
    unstable_ids = sorted(
        case_id
        for case_id in run_scores[0]
        if len({scores.get(case_id, 0.0) for scores in run_scores}) > 1
    )

    good_rows = [by_id[case_id] for case_id in old_good_ids if case_id in by_id]
    bad_rows = [by_id[case_id] for case_id in old_bad_ids if case_id in by_id]
    unstable_rows = [by_id[case_id] for case_id in unstable_ids if case_id in by_id]

    sampled_good = rng_good.sample(good_rows, min(50, len(good_rows)))
    sampled_bad = rng_bad.sample(bad_rows, min(100, len(bad_rows)))

    _write_jsonl(OUT_DIR / "old_good_50.jsonl", sampled_good)
    _write_jsonl(OUT_DIR / "old_bad_100.jsonl", sampled_bad)
    _write_jsonl(OUT_DIR / "unstable_all.jsonl", unstable_rows)

    manifest = {
        "generated_from": {
            "test_data": str(TEST_DATA),
            "old_results": str(OLD_RESULTS),
            "wrong600_runs": [str(path) for path in WRONG600_RUNS],
        },
        "sizes": {
            "old_good_50": len(sampled_good),
            "old_bad_100": len(sampled_bad),
            "unstable_all": len(unstable_rows),
        },
        "files": {
            "old_good_50": str(OUT_DIR / "old_good_50.jsonl"),
            "old_bad_100": str(OUT_DIR / "old_bad_100.jsonl"),
            "unstable_all": str(OUT_DIR / "unstable_all.jsonl"),
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
