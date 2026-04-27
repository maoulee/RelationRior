from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subgraph_kgqa.skill_mining.case_skill import (
    render_case_skill_markdown,
    synthesize_case_skill_from_dataset_case,
)
from subgraph_kgqa.skill_mining.schemas import read_json, write_json


def _load_dataset_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def _case_sort_key(case_id: str) -> Tuple[int, str]:
    m = re.search(r"(\d+)$", str(case_id))
    if m:
        return int(m.group(1)), str(case_id)
    return 10**12, str(case_id)


async def _build_one(
    dataset_case: Dict[str, Any],
    *,
    output_dir: Path,
    use_llm: bool,
    skip_existing: bool,
    sem: asyncio.Semaphore,
) -> Tuple[str, str]:
    case_id = str(dataset_case.get("id"))
    json_path = output_dir / f"{case_id}.json"
    md_path = output_dir / f"{case_id}.md"
    if skip_existing and json_path.exists() and md_path.exists():
        return case_id, "skipped"

    async with sem:
        try:
            card = await synthesize_case_skill_from_dataset_case(dataset_case, use_llm=use_llm)
            status = "llm" if use_llm else "fallback"
        except Exception:
            card = await synthesize_case_skill_from_dataset_case(dataset_case, use_llm=False)
            status = "fallback_after_error"

    write_json(json_path, card)
    md_path.write_text(render_case_skill_markdown(card), encoding="utf-8")
    return case_id, status


def _build_index(output_dir: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for path in sorted(output_dir.glob("WebQTrn-*.json"), key=lambda p: _case_sort_key(p.stem)):
        payload = read_json(path)
        entries.append(
            {
                "case_id": payload.get("case_id", path.stem),
                "question": payload.get("question", ""),
                "question_type": payload.get("question_type", ""),
                "retrieval_fields": payload.get("retrieval_fields", {}),
                "core_relation_domains": payload.get("core_relation_domains", []),
                "core_relations": payload.get("core_relations", []),
                "action_space_experience": payload.get("action_space_experience", ""),
                "final_selection_experience": payload.get("final_selection_experience", []),
                "candidate_constraint_relations": payload.get("candidate_constraint_relations", []),
                "candidate_constraint_entities": payload.get("candidate_constraint_entities", []),
                "json_path": str(path),
                "md_path": str(path.with_suffix(".md")),
            }
        )
    return entries


def _build_master_markdown(output_dir: Path, entries: List[Dict[str, Any]], *, title: str) -> Path:
    parts: List[str] = [
        f"# {title}",
        "",
        f"- Total case skills: {len(entries)}",
        "",
    ]
    for idx, entry in enumerate(sorted(entries, key=lambda item: _case_sort_key(str(item.get("case_id", ""))))):
        md_path = Path(str(entry["md_path"]))
        if not md_path.exists():
            continue
        if idx > 0:
            parts.extend(["", "---", ""])
        parts.append(md_path.read_text(encoding="utf-8").rstrip())
    master_path = output_dir / "all_case_skills.md"
    master_path.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
    return master_path


async def main() -> int:
    parser = argparse.ArgumentParser(description="Build English case skills for the full WebQTrn training set.")
    parser.add_argument("--data-path", default="data/webqsp/webqsp_train.jsonl")
    parser.add_argument("--skills-root", default="skills")
    parser.add_argument("--output-subdir", default="webqsp_train_case_skills_en")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    skills_root = Path(args.skills_root)
    output_dir = skills_root / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_dataset_rows(data_path)
    if args.case_id:
        wanted = set(args.case_id)
        rows = [row for row in rows if str(row.get("id")) in wanted]
    if args.start_index > 0:
        rows = rows[args.start_index :]
    if args.limit > 0:
        rows = rows[: args.limit]

    sem = asyncio.Semaphore(max(1, args.concurrency))
    tasks = [
        _build_one(
            row,
            output_dir=output_dir,
            use_llm=not args.skip_llm,
            skip_existing=args.skip_existing,
            sem=sem,
        )
        for row in rows
    ]

    completed = 0
    status_counter: Dict[str, int] = {}
    for coro in asyncio.as_completed(tasks):
        case_id, status = await coro
        completed += 1
        status_counter[status] = status_counter.get(status, 0) + 1
        if completed % 50 == 0 or completed == len(tasks):
            print(f"[progress] {completed}/{len(tasks)} last={case_id} status={status}")

    index_entries = _build_index(output_dir)
    index_path = output_dir / "index.json"
    write_json(index_path, index_entries)
    master_md_path = _build_master_markdown(
        output_dir,
        index_entries,
        title="WebQTrn Training Set Case Skills",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "index_path": str(index_path),
                "master_md_path": str(master_md_path),
                "built_rows": len(rows),
                "status_counter": status_counter,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
