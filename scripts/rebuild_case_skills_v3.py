#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subgraph_kgqa.skill_mining.case_skill import (
    synthesize_case_skill_from_dataset_case,
    upsert_case_skill_outputs,
)


def _load_cases(data_path: Path, limit: int | None = None) -> list[dict]:
    cases: list[dict] = []
    with data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            cases.append(json.loads(line))
            if limit is not None and len(cases) >= limit:
                break
    return cases


async def _run_case(
    case: dict,
    *,
    out_root: Path,
    use_llm: bool,
    semaphore: asyncio.Semaphore,
) -> dict:
    case_id = str(case.get("id", "unknown"))
    t0 = time.time()
    async with semaphore:
        try:
            card = await synthesize_case_skill_from_dataset_case(case, use_llm=use_llm)
            paths = upsert_case_skill_outputs(out_root, card)
            return {
                "case_id": case_id,
                "status": "ok",
                "seconds": round(time.time() - t0, 2),
                "core_relations": list(card.core_relations or []),
                "json_path": paths["json_path"],
                "md_path": paths["md_path"],
            }
        except Exception as exc:
            return {
                "case_id": case_id,
                "status": "error",
                "seconds": round(time.time() - t0, 2),
                "error": str(exc),
            }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild webqsp_train_case_skills_v3 with concurrency.")
    parser.add_argument("--data-path", default="data/webqsp/webqsp_train.jsonl")
    parser.add_argument("--out-root", default="skills/webqsp_train_case_skills_v3")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--report-path", default="reports/skill_v3_test/rebuild_case_skills_v3_report.json")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    out_root = Path(args.out_root)
    report_path = Path(args.report_path)
    use_llm = not args.skip_llm
    limit = args.limit if args.limit > 0 else None

    cases = _load_cases(data_path, limit=limit)
    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    tasks = [
        asyncio.create_task(
            _run_case(case, out_root=out_root, use_llm=use_llm, semaphore=semaphore)
        )
        for case in cases
    ]

    results: list[dict] = []
    total = len(tasks)
    completed = 0
    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)
        completed += 1
        status = result["status"]
        core = ",".join(result.get("core_relations", []))
        print(f"[{completed}/{total}] {result['case_id']} {status} {core}")
        if completed % 25 == 0 or completed == total:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for r in results if r["status"] == "ok")
    err = total - ok
    print(f"done ok={ok} err={err} report={report_path}")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
