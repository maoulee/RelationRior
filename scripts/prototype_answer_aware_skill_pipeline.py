#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subgraph_kgqa.skill_mining.case_skill import (
    _available_full_relations,
    _call_llm,
    _expand_relation_to_full,
    _extract_json_object,
    _load_local_raw_attempts_for_source,
    _load_local_source_card,
    _rank_core_relations_by_reward,
    _shortest_path_candidates_for_case,
    _trajectory_relation_candidates,
    _to_list,
)


def _load_case(data_path: Path, case_id: str) -> Dict[str, Any]:
    with data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row.get("id")) == case_id:
                return row
    raise ValueError(f"Case not found: {case_id}")


async def _decompose_with_answers(case: Dict[str, Any]) -> Dict[str, Any]:
    gt = case.get("ground_truth", {}) or {}
    question = str(case.get("question", "") or "")
    if not question:
        messages = case.get("messages", []) or []
        question = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")

    prompt_payload = {
        "task": "Decompose this training example into one or more answer-aware subproblems.",
        "rules": [
            "Return JSON only.",
            "Use the ground-truth answers as supervision.",
            "For WebQSP-like single-question cases, usually return one subproblem.",
            "For multi-answer / compositional cases, group answers by the subproblem that most naturally produces them.",
            "Each subproblem must include: subquestion, anchor, answers_for_this_subproblem, related, maybe_related.",
            "related and maybe_related should contain relation candidates that a model would plausibly plan with.",
            "Prefer concise relation lists; do not dump all nearby relations.",
        ],
        "json_schema": {
            "subproblems": [
                {
                    "subquestion": "string",
                    "anchor": ["string"],
                    "answers_for_this_subproblem": ["string"],
                    "related": ["string"],
                    "maybe_related": ["string"],
                }
            ]
        },
        "question": question,
        "ground_truth_answers": _to_list(gt.get("global_truth_answers")),
        "oracle_relations": _to_list(gt.get("oracle_relations")),
        "core_relations": _to_list(gt.get("core_relations")),
    }

    raw = await _call_llm(
        [
            {
                "role": "system",
                "content": "You decompose KGQA training examples into answer-aware subproblems. Return JSON only.",
            },
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False, indent=2)},
        ]
    )
    payload = _extract_json_object(raw) or {}
    return {
        "raw_llm_response": raw,
        "subproblems": payload.get("subproblems", []) if isinstance(payload.get("subproblems"), list) else [],
    }


def _normalize_relation_candidates(
    relations: List[str],
    available_relations: List[str],
) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for rel in relations:
        full = _expand_relation_to_full(rel, available_relations)
        if full and full not in seen:
            normalized.append(full)
            seen.add(full)
    return normalized


async def _generate_strategy(
    *,
    question: str,
    subquestion: str,
    answers: List[str],
    core_relation: str,
    path_candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    prompt_payload = {
        "task": "Generate answer strategy for a KGQA skill once core_relation is fixed.",
        "rules": [
            "Return JSON only.",
            "Keep action_space_mode fixed to one of: keep_whole_action_space, filter_within_action_space, collapse_within_action_space.",
            "Write filter_likely_attributes and selection_rule in natural language based on the question and the answer-bearing relation.",
            "Do not invent a new relation. Use the provided core_relation and path evidence.",
            "selection_rule should describe how to resolve answers after the action space is retrieved.",
        ],
        "json_schema": {
            "action_space_mode": "keep_whole_action_space|filter_within_action_space|collapse_within_action_space",
            "filter_likely_attributes": ["string"],
            "selection_rule": "string",
        },
        "question": question,
        "subquestion": subquestion,
        "answers_for_this_subproblem": answers,
        "core_relation": core_relation,
        "shortest_path_candidates": path_candidates,
    }
    raw = await _call_llm(
        [
            {
                "role": "system",
                "content": "You write answer strategy for a KGQA skill. Return JSON only.",
            },
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False, indent=2)},
        ]
    )
    payload = _extract_json_object(raw) or {}
    payload["raw_llm_response"] = raw
    return payload


async def run_case(case: Dict[str, Any]) -> Dict[str, Any]:
    case_id = str(case.get("id"))
    source_card = _load_local_source_card(case_id)
    raw_attempts = _load_local_raw_attempts_for_source(source_card)
    question = str(case.get("question", "") or "")
    if not question:
        messages = case.get("messages", []) or []
        question = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")

    decomposition = await _decompose_with_answers(case)
    available_relations = _available_full_relations(source_card or {}, case)
    shortest_path_candidates = _shortest_path_candidates_for_case(case_id)
    shortest_path_relations = {
        rel
        for item in shortest_path_candidates
        for rel in item.get("relations", [])
    }
    trajectory_candidates = _trajectory_relation_candidates(
        question,
        source_card=source_card,
        raw_attempts=raw_attempts,
        dataset_case=case,
    )

    enriched_subproblems: List[Dict[str, Any]] = []
    for sp in decomposition.get("subproblems", []):
        related = _normalize_relation_candidates(_to_list(sp.get("related")), available_relations)
        maybe_related = _normalize_relation_candidates(_to_list(sp.get("maybe_related")), available_relations)
        proposed = []
        seen = set()
        for rel in related + maybe_related:
            if rel not in seen:
                proposed.append(rel)
                seen.add(rel)

        on_shortest = [rel for rel in proposed if rel in shortest_path_relations]
        reward_pool = on_shortest or proposed or trajectory_candidates or list(shortest_path_relations)
        ranked = _rank_core_relations_by_reward(
            case_id,
            question,
            reward_pool,
            preferred_order=trajectory_candidates + proposed,
        )
        core_relation = ranked[0] if ranked else ""
        relevant_paths = [
            item for item in shortest_path_candidates if core_relation and core_relation in item.get("relations", [])
        ]
        strategy = await _generate_strategy(
            question=question,
            subquestion=str(sp.get("subquestion", "")).strip(),
            answers=_to_list(sp.get("answers_for_this_subproblem")),
            core_relation=core_relation,
            path_candidates=relevant_paths or shortest_path_candidates[:3],
        )

        enriched_subproblems.append(
            {
                "subquestion": sp.get("subquestion", ""),
                "anchor": _to_list(sp.get("anchor")),
                "answers_for_this_subproblem": _to_list(sp.get("answers_for_this_subproblem")),
                "related": related,
                "maybe_related": maybe_related,
                "trajectory_candidates": trajectory_candidates,
                "shortest_path_relations": sorted(shortest_path_relations),
                "relations_on_shortest_path": on_shortest,
                "reward_ranked_relations": ranked,
                "core_relation": core_relation,
                "strategy": {
                    "action_space_mode": strategy.get("action_space_mode", ""),
                    "filter_likely_attributes": _to_list(strategy.get("filter_likely_attributes")),
                    "selection_rule": strategy.get("selection_rule", ""),
                },
            }
        )

    return {
        "case_id": case_id,
        "question": question,
        "ground_truth_answers": _to_list((case.get("ground_truth", {}) or {}).get("global_truth_answers")),
        "decomposition": decomposition,
        "subproblems": enriched_subproblems,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Prototype answer-aware skill pipeline.")
    parser.add_argument("--data-path", default="data/webqsp/webqsp_train.jsonl")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--out", default="reports/skill_v3_test/prototype_answer_aware_skill_pipeline.json")
    args = parser.parse_args()

    if not args.case_id:
        raise SystemExit("--case-id is required")

    data_path = Path(args.data_path)
    out_path = Path(args.out)

    results = []
    for case_id in args.case_id:
        t0 = time.time()
        case = _load_case(data_path, case_id)
        result = await run_case(case)
        result["seconds"] = round(time.time() - t0, 2)
        results.append(result)
        print(f"{case_id}: done in {result['seconds']}s")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
