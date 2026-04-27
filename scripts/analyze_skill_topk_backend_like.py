#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subgraph_kgqa.skill_mining.retriever import (
    filter_skills_post_plan,
    load_dataset_lookup,
    select_similar_case_ids,
)
from subgraph_kgqa.skill_mining.schemas import CaseSkillCard
from scripts.analyze_relation_connectivity_backend_like import (
    _evaluate_relation,
    _extract_final_successful_plan,
    _load_backend_module,
)


def _load_skill_card(skills_root: Path, skill_id: str) -> Optional[CaseSkillCard]:
    path = skills_root / f"{skill_id}.json"
    if not path.exists():
        return None
    return CaseSkillCard(**json.loads(path.read_text(encoding="utf-8")))


async def analyze_results(
    *,
    results_path: Path,
    skills_root: Path,
    train_data_path: Path,
    query_data_path: Path,
    top_k: int,
    report_json: Path,
    max_hops: int,
    path_limit: int,
) -> Dict[str, Any]:
    backend = _load_backend_module()
    manager = backend.DataManager()
    manager.load_data("/zhaoshu/SubgraphRAG-main", "webqsp", "test")

    dataset_lookup = load_dataset_lookup(train_data_path)
    records = json.loads(results_path.read_text(encoding="utf-8"))
    relation_cache: Dict[tuple, Dict[str, Any]] = {}

    rows: List[Dict[str, Any]] = []
    for idx, record in enumerate(records, 1):
        case_id = str(record.get("case_id"))
        matcher = manager.get_matcher(case_id)
        if matcher is None:
            continue
        anchor, plan_relations = _extract_final_successful_plan(record)
        top_ids, retrieval_note = await select_similar_case_ids(
            target_question=str(record.get("question", "")),
            dataset_lookup=dataset_lookup,
            data_path=train_data_path,
            cache_root=ROOT / "skills",
            exclude_case_id=None,
            query_case_id=case_id,
            query_data_path=query_data_path,
            candidate_limit=max(top_k * 2, 20),
            top_k=top_k,
        )
        top_cards = [_load_skill_card(skills_root, sid) for sid in top_ids]
        top_cards = [card for card in top_cards if card is not None]

        ground_truth = [str(x) for x in (record.get("ground_truth", []) or [])]

        vector_skill_rows = []
        for card in top_cards:
            relation_evals = []
            for rel in card.core_relations or []:
                rel_eval = _evaluate_relation(
                    matcher,
                    backend.collect_expanded_entities,
                    backend_module=backend,
                    anchor=anchor,
                    relation=rel,
                    ground_truth=ground_truth,
                    cache=relation_cache,
                    max_hops=max_hops,
                    path_limit=path_limit,
                )
                relation_evals.append({"relation": rel, **rel_eval})
            vector_skill_rows.append(
                {
                    "skill_id": card.case_id,
                    "question": card.question,
                    "core_relations": list(card.core_relations or []),
                    "effective": any(item["ground_truth_covered"] for item in relation_evals),
                    "relation_evaluations": relation_evals,
                }
            )

        posterior_cards = filter_skills_post_plan(top_cards, plan_relations, max_cards=top_k) if plan_relations else []
        posterior_skill_rows = []
        for card in posterior_cards:
            relation_evals = []
            for rel in card.core_relations or []:
                rel_eval = _evaluate_relation(
                    matcher,
                    backend.collect_expanded_entities,
                    backend_module=backend,
                    anchor=anchor,
                    relation=rel,
                    ground_truth=ground_truth,
                    cache=relation_cache,
                    max_hops=max_hops,
                    path_limit=path_limit,
                )
                relation_evals.append({"relation": rel, **rel_eval})
            posterior_skill_rows.append(
                {
                    "skill_id": card.case_id,
                    "question": card.question,
                    "core_relations": list(card.core_relations or []),
                    "effective": any(item["ground_truth_covered"] for item in relation_evals),
                    "relation_evaluations": relation_evals,
                }
            )

        rows.append(
            {
                "case_id": case_id,
                "f1": float(record.get("f1", 0.0)),
                "anchor": anchor,
                "ground_truth": ground_truth,
                "plan_relations": plan_relations,
                "retrieval_note": retrieval_note,
                "vector_topk_skills": vector_skill_rows,
                "posterior_skills": posterior_skill_rows,
            }
        )
        if idx % 100 == 0:
            print(f"progress {idx}/{len(records)}")

    def summarize(case_rows: List[Dict[str, Any]], *, correct_only: bool) -> Dict[str, Any]:
        subset = [r for r in case_rows if (r["f1"] >= 0.95)] if correct_only else case_rows
        if not subset:
            return {}
        return {
            "cases": len(subset),
            "vector_any_effective": sum(1 for r in subset if any(s["effective"] for s in r["vector_topk_skills"])),
            "vector_avg_effective": round(
                sum(sum(1 for s in r["vector_topk_skills"] if s["effective"]) for r in subset) / len(subset), 3
            ),
            "posterior_any_effective": sum(1 for r in subset if any(s["effective"] for s in r["posterior_skills"])),
            "posterior_avg_effective": round(
                sum(sum(1 for s in r["posterior_skills"] if s["effective"]) for r in subset) / len(subset), 3
            ),
            "posterior_avg_count": round(
                sum(len(r["posterior_skills"]) for r in subset) / len(subset), 3
            ),
        }

    report = {
        "summary_all": summarize(rows, correct_only=False),
        "summary_correct_only": summarize(rows, correct_only=True),
        "rows": rows,
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze skill top-k effectiveness with backend-like relation connectivity.")
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--skills-root", required=True)
    parser.add_argument("--train-data-path", default="data/webqsp/webqsp_train.jsonl")
    parser.add_argument("--query-data-path", default="data/webqsp/webqsp_test.jsonl")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--path-limit", type=int, default=5)
    args = parser.parse_args()

    report = asyncio.run(
        analyze_results(
            results_path=Path(args.results_path),
            skills_root=Path(args.skills_root),
            train_data_path=Path(args.train_data_path),
            query_data_path=Path(args.query_data_path),
            top_k=args.top_k,
            report_json=Path(args.report_json),
            max_hops=args.max_hops,
            path_limit=args.path_limit,
        )
    )
    print(json.dumps(report["summary_all"], ensure_ascii=False, indent=2))
    print(json.dumps(report["summary_correct_only"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
