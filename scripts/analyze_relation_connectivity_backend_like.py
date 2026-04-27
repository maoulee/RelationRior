#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]


def _load_backend_module():
    path = Path("/zhaoshu/SubgraphRAG-main/retrieve/graph_server.py")
    spec = importlib.util.spec_from_file_location("gs_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load backend module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_final_successful_plan(record: Dict[str, Any]) -> Tuple[str, List[str]]:
    final_anchor = ""
    final_relations: List[str] = []

    for turn in record.get("trajectory", []) or []:
        parsed = turn.get("parsed_output", {}) or {}
        queries = parsed.get("queries", []) or []
        backend_results = turn.get("backend_results", []) or []

        plan_query = None
        for query in queries:
            if query.get("tool_name") == "plan":
                plan_query = query
                break
        if plan_query is None:
            continue

        plan_success = False
        for result in backend_results:
            if result.get("tool_name") == "plan" and result.get("is_success"):
                plan_success = True
                break
        if not plan_success:
            continue

        args = plan_query.get("arguments", {}) or {}
        anchor = ""
        raw_anchor = args.get("anchor", [])
        if isinstance(raw_anchor, list) and raw_anchor:
            anchor = str(raw_anchor[0])
        elif isinstance(raw_anchor, str):
            anchor = raw_anchor

        relations = list(args.get("related", []) or []) + list(args.get("maybe_related", []) or [])
        deduped: List[str] = []
        seen = set()
        for rel in relations:
            r = str(rel).strip()
            if r and r not in seen:
                deduped.append(r)
                seen.add(r)
        final_anchor = anchor
        final_relations = deduped

    if final_anchor:
        return final_anchor, final_relations
    return "", []


def _evaluate_relation(
    matcher: Any,
    collect_expanded_entities: Any,
    *,
    anchor: str,
    relation: str,
    ground_truth: Iterable[str],
    backend_module: Any,
    cache: Dict[Tuple[str, str, int, int], Dict[str, Any]],
    max_hops: int,
    path_limit: int,
) -> Dict[str, Any]:
    cache_key = (anchor, relation, max_hops, path_limit)
    if cache_key in cache:
        cached = dict(cache[cache_key])
        expanded_set = set(cached.get("expanded_entities", []))
        gt_set = set(ground_truth)
        hits = sorted(gt_set & expanded_set)
        cached["ground_truth_hits"] = hits
        cached["ground_truth_hit_count"] = len(hits)
        cached["ground_truth_covered"] = bool(hits)
        cached["ground_truth_covered_all"] = gt_set.issubset(expanded_set)
        return cached

    try:
        start_res = matcher.find_entities(anchor, limit=5)
        if not start_res or start_res[0]["score"] < 100:
            raise ValueError(f"Unresolved anchor: {anchor}")
        valid_anchor = start_res[0]["entity"]
    except Exception:
        valid_anchor = ""

    action_hints: List[Dict[str, Any]] = []
    if valid_anchor:
        try:
            plan_resp = backend_module._sync_find_logical_path(
                matcher,
                backend_module.PathRequest(
                    sample_id="unused",
                    start_entity=valid_anchor,
                    contains_relation=relation,
                    max_hops=max_hops,
                    limit=path_limit,
                ),
            )
            action_hints = list(plan_resp.action_hints or []) if getattr(plan_resp, "status", "") == "KG_SUCCESS" else []
        except Exception:
            action_hints = []

    expanded_set = set()
    action_rows: List[Dict[str, Any]] = []
    for hint in action_hints:
        steps = list(hint.get("steps", []) or [])
        if not steps:
            continue
        try:
            _, paths = matcher.execute_match_pattern(valid_anchor, steps)
        except Exception:
            paths = []
        ends = collect_expanded_entities(matcher, valid_anchor, paths) if paths else []
        ends_set = set(ends)
        expanded_set.update(ends_set)
        action_rows.append(
            {
                "steps": steps,
                "num_paths": len(paths),
                "expanded_entities": sorted(ends_set),
            }
        )

    gt_set = set(ground_truth)
    hits = sorted(gt_set & expanded_set)
    result = {
        "num_paths": sum(row["num_paths"] for row in action_rows),
        "num_action_hints": len(action_rows),
        "expanded_entities": sorted(expanded_set),
        "ground_truth_hits": hits,
        "ground_truth_hit_count": len(hits),
        "ground_truth_covered": bool(hits),
        "ground_truth_covered_all": gt_set.issubset(expanded_set),
        "action_evaluations": action_rows,
    }
    cache[cache_key] = dict(result)
    return result


def _load_skill_card(skills_root: Path, skill_id: str) -> Optional[Dict[str, Any]]:
    path = skills_root / f"{skill_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_results(
    *,
    results_path: Path,
    split: str,
    dataset_name: str,
    skills_root: Optional[Path] = None,
    report_json: Optional[Path] = None,
    max_hops: int = 3,
    path_limit: int = 5,
) -> Dict[str, Any]:
    backend = _load_backend_module()
    manager = backend.DataManager()
    manager.load_data("/zhaoshu/SubgraphRAG-main", dataset_name, split)

    records = json.loads(results_path.read_text(encoding="utf-8"))

    connectivity_cache: Dict[Tuple[str, str, int, int], Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    summary = {
        "total_cases": 0,
        "cases_with_plan_relations": 0,
        "cases_with_any_effective_plan_relation": 0,
        "correct_cases": 0,
        "correct_cases_with_plan_relations": 0,
        "correct_cases_with_any_effective_plan_relation": 0,
        "total_plan_relations": 0,
        "total_effective_plan_relations": 0,
        "cases_with_selected_skills": 0,
        "cases_with_any_effective_skill": 0,
        "correct_cases_with_selected_skills": 0,
        "correct_cases_with_any_effective_skill": 0,
        "total_selected_skills": 0,
        "total_effective_skills": 0,
    }

    for idx, record in enumerate(records, 1):
        case_id = str(record.get("case_id"))
        matcher = manager.get_matcher(case_id)
        if matcher is None:
            continue
        ground_truth = [str(x) for x in (record.get("ground_truth", []) or [])]
        correct = float(record.get("f1", 0.0)) >= 0.95
        summary["total_cases"] += 1
        if correct:
            summary["correct_cases"] += 1

        anchor, plan_relations = _extract_final_successful_plan(record)
        plan_relation_rows: List[Dict[str, Any]] = []
        if plan_relations and anchor:
            summary["cases_with_plan_relations"] += 1
            summary["total_plan_relations"] += len(plan_relations)
            for rel in plan_relations:
                rel_eval = _evaluate_relation(
                    matcher,
                    backend.collect_expanded_entities,
                    backend_module=backend,
                    anchor=anchor,
                    relation=rel,
                    ground_truth=ground_truth,
                    cache=connectivity_cache,
                    max_hops=max_hops,
                    path_limit=path_limit,
                )
                plan_relation_rows.append({"relation": rel, **rel_eval})
            effective_plan_count = sum(1 for row in plan_relation_rows if row["ground_truth_covered"])
            summary["total_effective_plan_relations"] += effective_plan_count
            if effective_plan_count > 0:
                summary["cases_with_any_effective_plan_relation"] += 1
                if correct:
                    summary["correct_cases_with_any_effective_plan_relation"] += 1
            if correct:
                summary["correct_cases_with_plan_relations"] += 1
        else:
            effective_plan_count = 0

        skill_rows: List[Dict[str, Any]] = []
        selected_skill_ids = (record.get("skill_bundle") or {}).get("selected_case_ids", []) or []
        if skills_root is not None and selected_skill_ids:
            summary["cases_with_selected_skills"] += 1
            summary["total_selected_skills"] += len(selected_skill_ids)
            if correct:
                summary["correct_cases_with_selected_skills"] += 1
            for skill_id in selected_skill_ids:
                card = _load_skill_card(skills_root, skill_id)
                if not card:
                    continue
                core_relations = list(card.get("core_relations") or [])
                relation_evals = []
                for rel in core_relations:
                    rel_eval = _evaluate_relation(
                        matcher,
                        backend.collect_expanded_entities,
                        backend_module=backend,
                        anchor=anchor,
                        relation=rel,
                        ground_truth=ground_truth,
                        cache=connectivity_cache,
                        max_hops=max_hops,
                        path_limit=path_limit,
                    )
                    relation_evals.append({"relation": rel, **rel_eval})
                effective = any(item["ground_truth_covered"] for item in relation_evals)
                if effective:
                    summary["total_effective_skills"] += 1
                skill_rows.append(
                    {
                        "skill_id": skill_id,
                        "question": card.get("question", ""),
                        "core_relations": core_relations,
                        "effective": effective,
                        "relation_evaluations": relation_evals,
                    }
                )
            if any(item["effective"] for item in skill_rows):
                summary["cases_with_any_effective_skill"] += 1
                if correct:
                    summary["correct_cases_with_any_effective_skill"] += 1

        rows.append(
            {
                "case_id": case_id,
                "f1": float(record.get("f1", 0.0)),
                "anchor": anchor,
                "ground_truth": ground_truth,
                "plan_relations": plan_relation_rows,
                "effective_plan_count": effective_plan_count,
                "selected_skills": skill_rows,
            }
        )

        if idx % 250 == 0:
            print(f"progress {idx}/{len(records)}")

    report = {"summary": summary, "rows": rows}
    if report_json is not None:
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze plan/skill relation connectivity with backend-like expansion.")
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-name", default="webqsp")
    parser.add_argument("--skills-root", default="")
    parser.add_argument("--report-json", default="")
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--path-limit", type=int, default=5)
    args = parser.parse_args()

    report = analyze_results(
        results_path=Path(args.results_path),
        split=args.split,
        dataset_name=args.dataset_name,
        skills_root=Path(args.skills_root) if args.skills_root else None,
        report_json=Path(args.report_json) if args.report_json else None,
        max_hops=args.max_hops,
        path_limit=args.path_limit,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
