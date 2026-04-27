from __future__ import annotations

import asyncio
import importlib.util
import json
import pickle
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


from ._helpers import (
    _extract_json_object,
    _extract_question_surface as extract_question_surface,
    _to_list,
)
from ..llm_client import call_llm
from .schemas import CaseSkillCard, read_json, write_json


QUESTION_TYPE_RULES = """
Question type must follow a stable relation-oriented rule:

1. Format:
   <source_semantic>_to_<target_semantic>.<relation_family>

2. The question type must describe stable semantic structure, not transient scope.
   Put current/latest/first/year-specific/cardinality details into answer_strategy or retrieval_fields.scope_mode, not into question_type.

3. Good examples:
   - person_to_team.affiliation
   - actor_to_character.portrayal
   - jurisdiction_to_role_holder.assignment
   - entity_to_country.location
   - person_to_person.kinship

4. Bad examples:
   - person_to_team.current_affiliation
   - current_prime_minister
   - latest_team_of_person
""".strip()

TRAIN_PROCESSED_PATH = Path("/zhaoshu/SubgraphRAG-main/retrieve/data_files/webqsp/processed/train.pkl")
TRAIN_TRIPLE_SCORE_PATH = Path("/zhaoshu/SubgraphRAG-main/retrieve/data_files/webqsp/triple_scores_gt/train.pth")
TRAIN_PATH_RELATION_CACHE_PATH = Path("/zhaoshu/subgraph/skills/_cache/webqsp_train_shortest_path_relations.json")
TRAIN_RELATION_REWARD_CACHE_PATH = Path("/zhaoshu/subgraph/skills/_cache/webqsp_train_relation_reward_stats.json")


async def _call_llm(messages: List[Dict[str, str]]) -> str:
    return await call_llm(messages, max_tokens=1400)


@lru_cache(maxsize=1)
def _load_train_graph_backend():
    path = Path("/zhaoshu/SubgraphRAG-main/retrieve/graph_server.py")
    spec = importlib.util.spec_from_file_location("gs_runtime_case_skill", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load backend module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manager = module.DataManager()
    manager.load_data("/zhaoshu/SubgraphRAG-main", "webqsp", "train")
    return module, manager


@lru_cache(maxsize=1)
def _load_train_processed_case_index() -> Dict[str, Dict[str, Any]]:
    if not TRAIN_PROCESSED_PATH.exists():
        return {}
    with TRAIN_PROCESSED_PATH.open("rb") as handle:
        rows = pickle.load(handle)
    return {str(row.get("id", "")): row for row in rows if row.get("id")}


@lru_cache(maxsize=1)
def _load_train_path_data_index() -> Dict[str, Any]:
    if not TRAIN_TRIPLE_SCORE_PATH.exists():
        return {}
    try:
        import torch
    except Exception:
        return {}
    return torch.load(TRAIN_TRIPLE_SCORE_PATH, map_location="cpu", weights_only=False)


@lru_cache(maxsize=1)
def _load_all_local_source_cards() -> Dict[str, Dict[str, Any]]:
    root = Path("/zhaoshu/subgraph/skills/source_cards")
    cards: Dict[str, Dict[str, Any]] = {}
    if not root.exists():
        return cards
    for path in root.glob("WebQTrn-*.json"):
        try:
            payload = read_json(path)
        except Exception:
            continue
        case_id = str(payload.get("case_id", "")).strip() or path.stem
        if case_id:
            cards[case_id] = payload
    return cards


@lru_cache(maxsize=1)
def _load_all_local_raw_attempts_by_record_id() -> Dict[str, Dict[str, Any]]:
    root = Path("/zhaoshu/subgraph/skills/raw_materials")
    records: Dict[str, Dict[str, Any]] = {}
    if not root.exists():
        return records
    for path in root.rglob("*.json"):
        try:
            payload = read_json(path)
        except Exception:
            continue
        record_id = str(payload.get("record_id", "")).strip()
        if record_id:
            records[record_id] = payload
    return records


def _available_full_relations(source_card: Dict[str, Any], dataset_case: Optional[Dict[str, Any]] = None) -> List[str]:
    relations: List[str] = []
    relations.extend(_to_list(source_card.get("planned_relations_seen")))
    prompt_context = source_card.get("prompt_context", {}) or {}
    relations.extend(_to_list(prompt_context.get("core_relations")))
    relations.extend(_to_list(source_card.get("constraint_relations_seen")))
    if dataset_case:
        gt = dataset_case.get("ground_truth", {}) or {}
        relations.extend(_to_list(gt.get("oracle_relations")))
        relations.extend(_to_list(gt.get("core_relations")))
    deduped: List[str] = []
    seen = set()
    for relation in relations:
        if relation and relation not in seen:
            deduped.append(relation)
            seen.add(relation)
    return deduped


def _train_question_anchor_entities(case_id: str, source_card: Optional[Dict[str, Any]] = None) -> List[str]:
    sample = _load_train_processed_case_index().get(str(case_id), {})
    anchors: List[str] = []
    if sample:
        entities = (sample.get("text_entity_list") or []) + (sample.get("non_text_entity_list") or [])
        for idx in sample.get("q_entity_id_list", []) or []:
            if isinstance(idx, int) and 0 <= idx < len(entities):
                anchor = str(entities[idx]).strip()
                if anchor:
                    anchors.append(anchor)
    if source_card:
        question_fields = source_card.get("question_fields", {}) or {}
        anchors.extend(_to_list(question_fields.get("core_entities")))
    deduped: List[str] = []
    seen = set()
    for anchor in anchors:
        item = str(anchor or "").strip()
        if item and item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped[:5]


def _extract_relation_hints_from_text(text: str) -> List[str]:
    raw = str(text or "")
    relations: List[str] = []
    seen = set()
    patterns = [
        r"- \[[0-9.]+\]\s*([A-Za-z0-9_#\.]+)",
        r"-\s+([A-Za-z0-9_#\.]+\.[A-Za-z0-9_#\.]+)",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, raw):
            relation = str(match).strip()
            if relation and relation not in seen and "." in relation:
                relations.append(relation)
                seen.add(relation)
    return relations


def _dataset_prompt_relation_candidates(dataset_case: Optional[Dict[str, Any]]) -> List[str]:
    if not dataset_case:
        return []
    messages = dataset_case.get("messages", []) or []
    user_text = "\n".join(
        str(message.get("content", ""))
        for message in messages
        if str(message.get("role", "")) == "user"
    )
    return _extract_relation_hints_from_text(user_text)


def _expand_relation_to_full(relation: str, available: List[str]) -> str:
    relation = str(relation or "").strip()
    if not relation:
        return relation
    if relation.count(".") >= 2:
        return relation
    candidates = [item for item in available if item == relation or item.endswith(f".{relation}")]
    if len(candidates) == 1:
        return candidates[0]
    candidates = [item for item in available if item.split(".", 1)[-1] == relation]
    if len(candidates) == 1:
        return candidates[0]
    return relation


def _normalize_full_relations(relations: List[str], source_card: Dict[str, Any], dataset_case: Optional[Dict[str, Any]] = None) -> List[str]:
    available = _available_full_relations(source_card, dataset_case)
    normalized: List[str] = []
    seen = set()
    for relation in relations:
        full = _expand_relation_to_full(relation, available)
        if full and full not in seen:
            normalized.append(full)
            seen.add(full)
    return normalized


def _derive_domains_from_relations(relations: List[str]) -> List[str]:
    return sorted({relation.split(".", 1)[0] for relation in relations if "." in relation})


def _limit_answer_bearing_relations(relations: List[str], *, max_relations: int = 1) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for relation in relations:
        rel = str(relation or "").strip()
        if not rel or rel in seen:
            continue
        deduped.append(rel)
        seen.add(rel)
        if len(deduped) >= max_relations:
            break
    return deduped


@lru_cache(maxsize=1)
def _load_train_shortest_path_relation_index() -> Dict[str, List[str]]:
    if TRAIN_PATH_RELATION_CACHE_PATH.exists():
        try:
            payload = read_json(TRAIN_PATH_RELATION_CACHE_PATH)
            if isinstance(payload, dict):
                return {
                    str(case_id): _to_list(relations)
                    for case_id, relations in payload.items()
                }
        except Exception:
            pass

    if not (TRAIN_PROCESSED_PATH.exists() and TRAIN_TRIPLE_SCORE_PATH.exists()):
        return {}

    try:
        import torch
    except Exception:
        return {}

    with TRAIN_PROCESSED_PATH.open("rb") as handle:
        processed_rows = pickle.load(handle)
    path_data_dict = torch.load(TRAIN_TRIPLE_SCORE_PATH, map_location="cpu", weights_only=False)

    relation_index: Dict[str, List[str]] = {}
    for sample in processed_rows:
        case_id = str(sample.get("id", "")).strip()
        if not case_id:
            continue
        path_data = path_data_dict.get(case_id)
        if not path_data:
            continue

        relation_list = sample.get("relation_list") or []
        relation_ids = sample.get("r_id_list") or []
        triple_scores = path_data.get("triple_scores")
        if triple_scores is None:
            continue
        if hasattr(triple_scores, "tolist"):
            triple_scores = triple_scores.tolist()

        path_relations: List[str] = []
        seen = set()
        for triple_idx, score in enumerate(triple_scores[: len(relation_ids)]):
            try:
                score_value = float(score)
            except Exception:
                continue
            if score_value <= 0:
                continue
            rel_id = relation_ids[triple_idx]
            if not isinstance(rel_id, int) or rel_id < 0 or rel_id >= len(relation_list):
                continue
            relation = str(relation_list[rel_id]).strip()
            if relation and relation not in seen:
                path_relations.append(relation)
                seen.add(relation)

        if path_relations:
            relation_index[case_id] = path_relations

    write_json(TRAIN_PATH_RELATION_CACHE_PATH, relation_index)
    return relation_index


def _ordered_shortest_path_relations(case_id: str, preferred_order: Optional[List[str]] = None) -> List[str]:
    path_relations = list(_load_train_shortest_path_relation_index().get(str(case_id), []))
    if not path_relations:
        return []
    if not preferred_order:
        return path_relations

    ordered: List[str] = []
    seen = set()
    for relation in preferred_order:
        if relation in path_relations and relation not in seen:
            ordered.append(relation)
            seen.add(relation)
    for relation in path_relations:
        if relation not in seen:
            ordered.append(relation)
            seen.add(relation)
    return ordered


def _validate_core_relations_with_shortest_path(
    case_id: str,
    relations: List[str],
    *,
    preferred_order: Optional[List[str]] = None,
) -> List[str]:
    normalized = _limit_answer_bearing_relations(relations, max_relations=max(2, len(relations) or 2))
    path_relations = _ordered_shortest_path_relations(case_id, preferred_order=(preferred_order or []) + normalized)
    if not path_relations:
        return normalized

    path_relation_set = set(path_relations)
    kept = [relation for relation in normalized if relation in path_relation_set]
    if kept:
        return kept
    return path_relations


@lru_cache(maxsize=1)
def _load_train_relation_reward_index() -> Dict[str, Dict[str, Dict[str, float]]]:
    if TRAIN_RELATION_REWARD_CACHE_PATH.exists():
        try:
            payload = read_json(TRAIN_RELATION_REWARD_CACHE_PATH)
            if (
                isinstance(payload, dict)
                and payload.get("__meta__", {}).get("version") == 2
                and isinstance(payload.get("cases"), dict)
            ):
                return payload["cases"]
        except Exception:
            pass

    if not (TRAIN_PROCESSED_PATH.exists() and TRAIN_TRIPLE_SCORE_PATH.exists()):
        return {}

    try:
        import torch
    except Exception:
        return {}

    with TRAIN_PROCESSED_PATH.open("rb") as handle:
        processed_rows = pickle.load(handle)
    path_data_dict = torch.load(TRAIN_TRIPLE_SCORE_PATH, map_location="cpu", weights_only=False)

    reward_index: Dict[str, Dict[str, Dict[str, float]]] = {}
    for sample in processed_rows:
        case_id = str(sample.get("id", "")).strip()
        if not case_id:
            continue
        path_data = path_data_dict.get(case_id)
        if not path_data:
            continue

        relation_list = sample.get("relation_list") or []
        relation_ids = sample.get("r_id_list") or []
        head_ids = sample.get("h_id_list") or []
        tail_ids = sample.get("t_id_list") or []
        anchor_ids = set(sample.get("q_entity_id_list") or [])
        answer_ids = set(sample.get("a_entity_id_list") or [])
        triple_scores = path_data.get("triple_scores")
        max_path_length = int(path_data.get("max_path_length", 1) or 1)
        if triple_scores is None:
            continue
        if hasattr(triple_scores, "tolist"):
            triple_scores = triple_scores.tolist()

        positive_adjacency: Dict[int, List[int]] = {}
        relation_stats: Dict[str, Dict[str, float]] = {}
        total = min(len(relation_ids), len(head_ids), len(tail_ids), len(triple_scores))
        for idx in range(total):
            rel_id = relation_ids[idx]
            if not isinstance(rel_id, int) or rel_id < 0 or rel_id >= len(relation_list):
                continue
            relation = str(relation_list[rel_id]).strip()
            if not relation:
                continue
            score = float(triple_scores[idx])
            if score > 0:
                positive_adjacency.setdefault(head_ids[idx], []).append(tail_ids[idx])
            stats = relation_stats.setdefault(
                relation,
                {
                    "total_sum": 0.0,
                    "total_max": 0.0,
                    "anchor_sum": 0.0,
                    "anchor_max": 0.0,
                    "answer_sum": 0.0,
                    "answer_max": 0.0,
                    "anchor_or_answer_sum": 0.0,
                    "anchor_or_answer_max": 0.0,
                    "anchor_out_sum": 0.0,
                    "anchor_out_max": 0.0,
                    "anchor_reaches_answer_sum": 0.0,
                    "anchor_reaches_answer_count": 0.0,
                },
            )
            stats["total_sum"] += score
            stats["total_max"] = max(stats["total_max"], score)
            head_id = head_ids[idx]
            tail_id = tail_ids[idx]
            touches_anchor = head_id in anchor_ids or tail_id in anchor_ids
            touches_answer = head_id in answer_ids or tail_id in answer_ids
            if touches_anchor:
                stats["anchor_sum"] += score
                stats["anchor_max"] = max(stats["anchor_max"], score)
            if head_id in anchor_ids:
                stats["anchor_out_sum"] += score
                stats["anchor_out_max"] = max(stats["anchor_out_max"], score)
            if touches_answer:
                stats["answer_sum"] += score
                stats["answer_max"] = max(stats["answer_max"], score)
            if touches_anchor or touches_answer:
                stats["anchor_or_answer_sum"] += score
                stats["anchor_or_answer_max"] = max(stats["anchor_or_answer_max"], score)

        if answer_ids and anchor_ids and relation_stats:
            remaining_steps = max(0, max_path_length - 1)

            def reaches_answer(start_node: int) -> bool:
                if start_node in answer_ids:
                    return True
                frontier = {start_node}
                visited = {start_node}
                for _ in range(remaining_steps):
                    next_frontier = set()
                    for node in frontier:
                        for nxt in positive_adjacency.get(node, []):
                            if nxt in answer_ids:
                                return True
                            if nxt not in visited:
                                visited.add(nxt)
                                next_frontier.add(nxt)
                    frontier = next_frontier
                    if not frontier:
                        break
                return False

            for idx in range(total):
                rel_id = relation_ids[idx]
                if not isinstance(rel_id, int) or rel_id < 0 or rel_id >= len(relation_list):
                    continue
                relation = str(relation_list[rel_id]).strip()
                score = float(triple_scores[idx])
                head_id = head_ids[idx]
                tail_id = tail_ids[idx]
                if score <= 0 or head_id not in anchor_ids:
                    continue
                if reaches_answer(tail_id):
                    stats = relation_stats[relation]
                    stats["anchor_reaches_answer_sum"] += score
                    stats["anchor_reaches_answer_count"] += 1.0

        if relation_stats:
            reward_index[case_id] = relation_stats

    write_json(
        TRAIN_RELATION_REWARD_CACHE_PATH,
        {
            "__meta__": {"version": 2},
            "cases": reward_index,
        },
    )
    return reward_index


def _question_relation_compatibility_score(question: str, relation: str) -> int:
    q = (question or "").lower()
    rel = (relation or "").lower()
    score = 0

    if any(token in q for token in [" brother", " sister", " sibling"]):
        if "sibling" in rel:
            score += 8
        if any(token in rel for token in ["children", "parents"]):
            score -= 4

    if any(token in q for token in [" child", " children", " son", " daughter"]):
        if "children" in rel:
            score += 8
        if "sibling" in rel:
            score -= 4

    if any(token in q for token in [" father", " mother", " parent", " parents"]):
        if "parents" in rel:
            score += 8
        if "sibling" in rel:
            score -= 4

    if any(token in q for token in [" wife", " husband", " spouse", " married"]):
        if any(token in rel for token in ["spouse", "married"]):
            score += 8

    if any(token in q for token in [" born", " birthplace", " where was"]) and "place_of_birth" in rel:
        score += 6

    if any(token in q for token in [" language", " speak", " spoken"]) and any(
        token in rel for token in ["language", "languages_spoken", "official_language"]
    ):
        score += 5

    if any(token in q for token in [" prime minister", " president", " governor", " mayor"]) and any(
        token in rel for token in ["office_holder", "governing", "position_to_holder"]
    ):
        score += 5

    return score


def _trajectory_relation_candidates(
    question: str,
    *,
    source_card: Optional[Dict[str, Any]] = None,
    raw_attempts: Optional[List[Dict[str, Any]]] = None,
    dataset_case: Optional[Dict[str, Any]] = None,
) -> List[str]:
    relations: List[str] = []
    if source_card:
        relations.extend(_to_list(source_card.get("planned_relations_seen")))
        prompt_context = source_card.get("prompt_context", {}) or {}
        relations.extend(_to_list(prompt_context.get("core_relations")))
    if raw_attempts:
        for attempt in raw_attempts:
            relations.extend(_to_list(attempt.get("planned_relations")))
    relations.extend(_dataset_prompt_relation_candidates(dataset_case))

    deduped: List[str] = []
    seen = set()
    for relation in relations:
        rel = str(relation or "").strip()
        if rel and rel not in seen and "." in rel:
            deduped.append(rel)
            seen.add(rel)
    return deduped


@lru_cache(maxsize=4096)
def _shortest_path_candidates_for_case(case_id: str) -> List[Dict[str, Any]]:
    sample = _load_train_processed_case_index().get(str(case_id))
    path_data = _load_train_path_data_index().get(str(case_id))
    if not sample or not path_data:
        return []
    graph = path_data.get("graph_obj")
    if graph is None:
        return []
    try:
        import graph_tool.all as gt
    except Exception:
        return []

    relation_list = sample.get("relation_list") or []
    q_ids = [qid for qid in sample.get("q_entity_id_list", []) if isinstance(qid, int)]
    a_ids = [aid for aid in sample.get("a_entity_id_list", []) if isinstance(aid, int)]
    if not q_ids or not a_ids:
        return []

    ep_relation_id = graph.edge_properties.get("relation_id")
    if ep_relation_id is None:
        return []

    def edge_path_to_steps(edges: List[Any], direction: str) -> List[Dict[str, str]]:
        if direction == "out":
            ordered_edges = edges
        else:
            ordered_edges = list(reversed(edges))
        steps: List[Dict[str, str]] = []
        for edge in ordered_edges:
            rel_id = ep_relation_id[edge]
            if not isinstance(rel_id, int) or rel_id < 0 or rel_id >= len(relation_list):
                continue
            relation = str(relation_list[rel_id]).strip()
            if relation:
                steps.append({"relation": relation, "direction": direction})
        return steps

    all_candidates: List[Dict[str, Any]] = []
    for q_id in q_ids:
        for a_id in a_ids:
            if not (0 <= q_id < graph.num_vertices() and 0 <= a_id < graph.num_vertices()):
                continue
            try:
                forward_paths = list(gt.all_shortest_paths(graph, q_id, a_id, edges=True))
            except Exception:
                forward_paths = []
            try:
                backward_paths = list(gt.all_shortest_paths(graph, a_id, q_id, edges=True))
            except Exception:
                backward_paths = []
            for edge_path in forward_paths:
                steps = edge_path_to_steps(list(edge_path), "out")
                if steps:
                    all_candidates.append({"steps": steps, "path_length": len(steps)})
            for edge_path in backward_paths:
                steps = edge_path_to_steps(list(edge_path), "in")
                if steps:
                    all_candidates.append({"steps": steps, "path_length": len(steps)})

    if not all_candidates:
        return []
    min_len = min(item["path_length"] for item in all_candidates)
    shortest = [item for item in all_candidates if item["path_length"] == min_len]

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in shortest:
        key = tuple((step["relation"], step["direction"]) for step in item["steps"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(
            {
                "steps": item["steps"],
                "path_length": item["path_length"],
                "relations": [step["relation"] for step in item["steps"]],
                "path_text": " -> ".join(
                    f"{step['relation']}[{step['direction']}]" for step in item["steps"]
                ),
            }
        )
    return deduped[:8]


@lru_cache(maxsize=4096)
def _shortest_path_core_relation_candidates_for_case(case_id: str) -> List[str]:
    sample = _load_train_processed_case_index().get(str(case_id))
    path_data = _load_train_path_data_index().get(str(case_id))
    if not sample or not path_data:
        return []
    graph = path_data.get("graph_obj")
    if graph is None:
        return []
    try:
        import graph_tool.all as gt
    except Exception:
        return []

    relation_list = sample.get("relation_list") or []
    q_ids = [qid for qid in sample.get("q_entity_id_list", []) if isinstance(qid, int)]
    a_ids = [aid for aid in sample.get("a_entity_id_list", []) if isinstance(aid, int)]
    cvt_ids = set(sample.get("non_text_entity_list", []) or [])
    entity_names = (sample.get("text_entity_list") or []) + (sample.get("non_text_entity_list") or [])
    ep_relation_id = graph.edge_properties.get("relation_id")
    if ep_relation_id is None or not q_ids or not a_ids:
        return []

    def _node_name(node_id: int) -> str:
        if 0 <= node_id < len(entity_names):
            return str(entity_names[node_id])
        return ""

    def _extract_from_edge_path(start_id: int, edge_path: List[Any]) -> Optional[str]:
        if not edge_path:
            return None
        node_ids: List[int] = [start_id]
        rel_ids: List[int] = []
        current = start_id
        for edge in edge_path:
            rel_ids.append(int(ep_relation_id[edge]))
            src = int(edge.source())
            tgt = int(edge.target())
            nxt = tgt if src == current else src
            node_ids.append(nxt)
            current = nxt
        if not rel_ids:
            return None
        predecessor_id = node_ids[-2] if len(node_ids) >= 2 else None
        rel_idx = -2 if predecessor_id is not None and _node_name(predecessor_id) in cvt_ids and len(rel_ids) >= 2 else -1
        rel_id = rel_ids[rel_idx]
        if not (0 <= rel_id < len(relation_list)):
            return None
        relation = str(relation_list[rel_id]).strip()
        return relation or None

    candidates: List[str] = []
    seen = set()
    for q_id in q_ids:
        for a_id in a_ids:
            if not (0 <= q_id < graph.num_vertices() and 0 <= a_id < graph.num_vertices()):
                continue
            try:
                forward_paths = list(gt.all_shortest_paths(graph, q_id, a_id, edges=True))
            except Exception:
                forward_paths = []
            try:
                backward_paths = list(gt.all_shortest_paths(graph, a_id, q_id, edges=True))
            except Exception:
                backward_paths = []
            for edge_path in forward_paths:
                relation = _extract_from_edge_path(q_id, list(edge_path))
                if relation and relation not in seen:
                    candidates.append(relation)
                    seen.add(relation)
            for edge_path in backward_paths:
                relation = _extract_from_edge_path(a_id, list(edge_path))
                if relation and relation not in seen:
                    candidates.append(relation)
                    seen.add(relation)
    return candidates[:8]


async def _llm_pick_winning_shortest_path(
    *,
    question: str,
    ground_truth_answers: List[str],
    path_candidates: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not path_candidates:
        return None
    prompt_payload = {
        "task": "Choose the most suitable shortest connected path between the anchor entity and the ground-truth answer, then identify the single core relation from that path.",
        "rules": [
            "Return JSON only.",
            "You must choose exactly one winning_path_index from the provided shortest path candidates.",
            "The winning path should best match the question's intended answer slot, not just graph adjacency.",
            "Then choose exactly one core_relation from relations that appear inside the chosen path.",
            "Prefer the relation in the chosen path that most directly carries the final answer semantics.",
            "Do not invent relations outside the chosen path.",
        ],
        "json_schema": {
            "winning_path_index": "integer",
            "core_relation": "string",
            "why": "string",
        },
        "question": question,
        "ground_truth_answers": ground_truth_answers,
        "path_candidates": [
            {
                "index": idx,
                "path_text": item["path_text"],
                "relations": item["relations"],
                "steps": item["steps"],
            }
            for idx, item in enumerate(path_candidates)
        ],
    }
    text = await _call_llm(
        [
            {"role": "system", "content": "You choose the best shortest KG path and its single core relation. Return JSON only."},
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False, indent=2)},
        ]
    )
    payload = _extract_json_object(text)
    if not payload:
        return None
    try:
        index = int(payload.get("winning_path_index"))
    except Exception:
        return None
    if index < 0 or index >= len(path_candidates):
        return None
    chosen = path_candidates[index]
    core_relation = str(payload.get("core_relation", "")).strip()
    if core_relation not in chosen["relations"]:
        return None
    return {
        "winning_path": chosen,
        "core_relation": core_relation,
        "why": str(payload.get("why", "")).strip(),
    }


def _load_local_source_card(case_id: str) -> Optional[Dict[str, Any]]:
    return _load_all_local_source_cards().get(str(case_id))


def _load_local_raw_attempts_for_source(source_card: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not source_card:
        return []
    record_ids = set(_to_list(source_card.get("raw_attempt_ids")))
    if not record_ids:
        return []
    record_map = _load_all_local_raw_attempts_by_record_id()
    attempts: List[Dict[str, Any]] = []
    for record_id in record_ids:
        payload = record_map.get(str(record_id))
        if payload:
            attempts.append(payload)
    attempts.sort(key=lambda item: str(item.get("created_at", "")))
    return attempts


def _reward_sort_key(stats: Dict[str, float]) -> Tuple[float, ...]:
    return (
        float(stats.get("anchor_reaches_answer_sum", 0.0)),
        float(stats.get("anchor_reaches_answer_count", 0.0)),
        float(stats.get("anchor_out_sum", 0.0)),
        float(stats.get("anchor_out_max", 0.0)),
        float(stats.get("anchor_or_answer_sum", 0.0)),
        float(stats.get("anchor_sum", 0.0)),
        float(stats.get("answer_sum", 0.0)),
        float(stats.get("total_sum", 0.0)),
        float(stats.get("anchor_or_answer_max", 0.0)),
        float(stats.get("anchor_max", 0.0)),
        float(stats.get("answer_max", 0.0)),
        float(stats.get("total_max", 0.0)),
    )


async def _llm_pick_core_relation(
    *,
    question: str,
    candidate_relations: List[str],
    available_relations: List[str],
    ground_truth_answers: List[str],
) -> Optional[str]:
    if not candidate_relations:
        return None
    prompt_payload = {
        "task": "Pick the single most likely answer-bearing core relation for this case.",
        "rules": [
            "Return JSON only.",
            "Choose exactly one relation from candidate_relations.",
            "Prioritize the relation that most directly matches the answer slot asked by the question.",
            "Do not choose a relation just because it is graph-near if it is semantically off-target.",
            "Use the ground-truth answer only to infer which relation most directly reaches the requested slot.",
            "Prefer the relation that the model should plan around, not a backup or exploratory neighbor relation.",
        ],
        "json_schema": {
            "core_relation": "string",
            "why": "string",
        },
        "question": question,
        "ground_truth_answers": ground_truth_answers,
        "candidate_relations": candidate_relations,
        "available_relations": available_relations,
    }
    text = await _call_llm(
        [
            {"role": "system", "content": "You choose the single best KGQA core relation. Return JSON only."},
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False, indent=2)},
        ]
    )
    payload = _extract_json_object(text)
    if not payload:
        return None
    relation = str(payload.get("core_relation", "")).strip()
    if relation in candidate_relations:
        return relation
    expanded = _expand_relation_to_full(relation, available_relations)
    return expanded if expanded in candidate_relations else None


def _rank_core_relations_by_reward(
    case_id: str,
    question: str,
    relations: List[str],
    *,
    preferred_order: Optional[List[str]] = None,
) -> List[str]:
    normalized = _limit_answer_bearing_relations(relations, max_relations=max(2, len(relations) or 2))
    if not normalized:
        return []

    reward_stats = _load_train_relation_reward_index().get(str(case_id), {})
    if not reward_stats:
        return sorted(
            normalized,
            key=lambda relation: (
                _question_relation_compatibility_score(question, relation),
                -(preferred_order or []).index(relation) if relation in (preferred_order or []) else -10**6,
            ),
            reverse=True,
        )

    preferred_rank = {relation: idx for idx, relation in enumerate(preferred_order or [])}
    ranked = sorted(
        normalized,
        key=lambda relation: (
            _reward_sort_key(reward_stats.get(relation, {})),
            _question_relation_compatibility_score(question, relation),
            -preferred_rank.get(relation, 10**6),
        ),
        reverse=True,
    )

    if len(ranked) > 1:
        return ranked
    return ranked


async def _select_core_relations_by_reward(
    case_id: str,
    question: str,
    relations: List[str],
    *,
    preferred_order: Optional[List[str]] = None,
    available_relations: Optional[List[str]] = None,
    ground_truth_answers: Optional[List[str]] = None,
    use_llm_tiebreak: bool = False,
) -> List[str]:
    ranked = _rank_core_relations_by_reward(
        case_id,
        question,
        relations,
        preferred_order=preferred_order,
    )
    if not ranked:
        return ranked
    if not use_llm_tiebreak or len(ranked) <= 1:
        return ranked

    reward_stats = _load_train_relation_reward_index().get(str(case_id), {})
    top_key = _reward_sort_key(reward_stats.get(ranked[0], {}))
    tied = [relation for relation in ranked if _reward_sort_key(reward_stats.get(relation, {})) == top_key]
    if len(tied) <= 1:
        return ranked
    chosen = await _llm_pick_core_relation(
        question=question,
        candidate_relations=tied,
        available_relations=available_relations or ranked,
        ground_truth_answers=ground_truth_answers or [],
    )
    if chosen:
        return [chosen] + [relation for relation in ranked if relation != chosen]
    return ranked


def _relation_validity_score_key(
    *,
    case_id: str,
    question: str,
    relation: str,
    eval_row: Dict[str, Any],
) -> Tuple[Any, ...]:
    reward_stats = _load_train_relation_reward_index().get(str(case_id), {})
    return (
        1 if eval_row.get("ground_truth_covered_all") else 0,
        float(eval_row.get("coverage_ratio", 0.0)),
        int(eval_row.get("ground_truth_hit_count", 0)),
        _question_relation_compatibility_score(question, relation),
        _reward_sort_key(reward_stats.get(relation, {})),
        int(eval_row.get("num_action_hints", 0)),
    )


def _is_clear_relation_winner(sorted_rows: List[Dict[str, Any]]) -> bool:
    if len(sorted_rows) <= 1:
        return True
    first = sorted_rows[0]
    second = sorted_rows[1]
    if int(first.get("ground_truth_hit_count", 0)) > int(second.get("ground_truth_hit_count", 0)):
        return True
    if float(first.get("coverage_ratio", 0.0)) > float(second.get("coverage_ratio", 0.0)):
        return True
    return False


@lru_cache(maxsize=32768)
def _evaluate_train_relation_action_validity(
    case_id: str,
    relation: str,
    *,
    max_hops: int = 3,
    path_limit: int = 5,
) -> Dict[str, Any]:
    backend, manager = _load_train_graph_backend()
    matcher = manager.get_matcher(str(case_id))
    if matcher is None:
        return {
            "anchor": "",
            "num_action_hints": 0,
            "num_paths": 0,
            "expanded_entities": [],
            "ground_truth_hits": [],
            "ground_truth_hit_count": 0,
            "ground_truth_covered": False,
            "ground_truth_covered_all": False,
            "coverage_ratio": 0.0,
            "action_evaluations": [],
        }

    sample = _load_train_processed_case_index().get(str(case_id), {})
    entities = (sample.get("text_entity_list") or []) + (sample.get("non_text_entity_list") or [])
    answer_names = [
        str(entities[idx]).strip()
        for idx in sample.get("a_entity_id_list", []) or []
        if isinstance(idx, int) and 0 <= idx < len(entities)
    ]
    gt_set = {item for item in answer_names if item}
    anchors = _train_question_anchor_entities(str(case_id))

    best_result: Optional[Dict[str, Any]] = None
    for anchor in anchors:
        try:
            plan_resp = backend._sync_find_logical_path(
                matcher,
                backend.PathRequest(
                    sample_id=str(case_id),
                    start_entity=anchor,
                    contains_relation=relation,
                    max_hops=max_hops,
                    limit=path_limit,
                ),
            )
        except Exception:
            continue
        if getattr(plan_resp, "status", "") != "KG_SUCCESS":
            continue
        action_hints = list(getattr(plan_resp, "action_hints", []) or [])
        expanded_set = set()
        action_rows: List[Dict[str, Any]] = []
        total_paths = 0
        for hint in action_hints:
            steps = list(hint.get("steps", []) or [])
            if not steps:
                continue
            try:
                _, paths = matcher.execute_match_pattern(anchor, steps)
            except Exception:
                paths = []
            ends = backend.collect_expanded_entities(matcher, anchor, paths) if paths else []
            total_paths += len(paths)
            ends_set = set(ends)
            expanded_set.update(ends_set)
            action_rows.append(
                {
                    "steps": steps,
                    "num_paths": len(paths),
                    "expanded_entities": sorted(ends_set),
                }
            )
        hits = sorted(gt_set & expanded_set)
        result = {
            "anchor": anchor,
            "num_action_hints": len(action_rows),
            "num_paths": total_paths,
            "expanded_entities": sorted(expanded_set),
            "ground_truth_hits": hits,
            "ground_truth_hit_count": len(hits),
            "ground_truth_covered": bool(hits),
            "ground_truth_covered_all": gt_set.issubset(expanded_set) if gt_set else False,
            "coverage_ratio": (len(hits) / len(gt_set)) if gt_set else 0.0,
            "action_evaluations": action_rows,
        }
        if best_result is None:
            best_result = result
            continue
        current_key = (
            int(result["ground_truth_covered_all"]),
            float(result["coverage_ratio"]),
            int(result["ground_truth_hit_count"]),
            int(result["num_action_hints"]),
        )
        best_key = (
            int(best_result["ground_truth_covered_all"]),
            float(best_result["coverage_ratio"]),
            int(best_result["ground_truth_hit_count"]),
            int(best_result["num_action_hints"]),
        )
        if current_key > best_key:
            best_result = result

    if best_result is not None:
        return best_result
    return {
        "anchor": anchors[0] if anchors else "",
        "num_action_hints": 0,
        "num_paths": 0,
        "expanded_entities": [],
        "ground_truth_hits": [],
        "ground_truth_hit_count": 0,
        "ground_truth_covered": False,
        "ground_truth_covered_all": False,
        "coverage_ratio": 0.0,
        "action_evaluations": [],
    }


async def _llm_pick_best_scored_relation(
    *,
    question: str,
    ground_truth_answers: List[str],
    candidate_rows: List[Dict[str, Any]],
) -> Optional[str]:
    if not candidate_rows:
        return None
    prompt_payload = {
        "task": "Choose the single best answer-bearing relation from the scored candidates.",
        "rules": [
            "Return JSON only.",
            "Primary evidence is action-space validity against the ground-truth answers.",
            "Prefer higher coverage_ratio and higher ground_truth_hit_count.",
            "Use question semantics only to break ties or reject semantically off-target relations.",
            "Do not invent new relations.",
        ],
        "json_schema": {
            "core_relation": "string",
            "why": "string",
        },
        "question": question,
        "ground_truth_answers": ground_truth_answers,
        "candidate_relations": [
            {
                "relation": row["relation"],
                "coverage_ratio": row["coverage_ratio"],
                "ground_truth_hit_count": row["ground_truth_hit_count"],
                "ground_truth_hits": row["ground_truth_hits"],
                "anchor": row["anchor"],
                "num_action_hints": row["num_action_hints"],
                "action_preview": [
                    " -> ".join(f"{step['relation']}[{step['direction']}]" for step in action.get("steps", []))
                    for action in row.get("action_evaluations", [])[:3]
                ],
            }
            for row in candidate_rows
        ],
    }
    text = await _call_llm(
        [
            {"role": "system", "content": "Choose the best scored answer-bearing relation. Return JSON only."},
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False, indent=2)},
        ]
    )
    payload = _extract_json_object(text)
    if not payload:
        return None
    relation = str(payload.get("core_relation", "")).strip()
    return relation if any(row["relation"] == relation for row in candidate_rows) else None


async def _choose_core_relation_v5(
    *,
    case_id: str,
    question: str,
    ground_truth_answers: List[str],
    plan_relation_candidates: List[str],
    shortest_path_relation_candidates: List[str],
) -> Dict[str, Any]:
    def _score_rows(relations: List[str]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        seen = set()
        for relation in relations:
            rel = str(relation or "").strip()
            if not rel or rel in seen:
                continue
            seen.add(rel)
            eval_row = _evaluate_train_relation_action_validity(case_id, rel)
            rows.append({"relation": rel, **eval_row})
        rows.sort(
            key=lambda row: _relation_validity_score_key(
                case_id=case_id,
                question=question,
                relation=row["relation"],
                eval_row=row,
            ),
            reverse=True,
        )
        return rows

    plan_rows = _score_rows(plan_relation_candidates)
    positive_plan_rows = [row for row in plan_rows if row.get("ground_truth_covered")]
    if positive_plan_rows:
        top_rows = positive_plan_rows[:3]
        if _is_clear_relation_winner(top_rows):
            chosen = top_rows[0]["relation"]
        else:
            chosen = await _llm_pick_best_scored_relation(
                question=question,
                ground_truth_answers=ground_truth_answers,
                candidate_rows=top_rows,
            ) or top_rows[0]["relation"]
        return {
            "selected_relation": chosen,
            "selection_source": "plan_scored",
            "top_candidates": top_rows,
        }

    fallback_rows = _score_rows(shortest_path_relation_candidates)
    positive_fallback_rows = [row for row in fallback_rows if row.get("ground_truth_covered")]
    if positive_fallback_rows:
        top_rows = positive_fallback_rows[:3]
        if _is_clear_relation_winner(top_rows):
            chosen = top_rows[0]["relation"]
        else:
            chosen = await _llm_pick_best_scored_relation(
                question=question,
                ground_truth_answers=ground_truth_answers,
                candidate_rows=top_rows,
            ) or top_rows[0]["relation"]
        return {
            "selected_relation": chosen,
            "selection_source": "shortest_path_scored",
            "top_candidates": top_rows,
        }

    combined_rows = plan_rows or fallback_rows
    top_rows = combined_rows[:3]
    chosen = top_rows[0]["relation"] if top_rows else ""
    return {
        "selected_relation": chosen,
        "selection_source": "fallback_ranked",
        "top_candidates": top_rows,
    }


def _relation_oriented_question_type(
    primary_relations: List[str],
    *,
    answer_type: str,
    temporal_policy: str,
    fallback_type: str,
) -> str:
    joined = " ".join(primary_relations)
    if any(token in joined for token in ["pro_athlete.teams", "basketball_player_stats.team", "football_player_stats.team", "sports_team_roster.team"]):
        return "person_to_team.affiliation"
    if any(token in joined for token in ["performance.actor", "performance.character", "film.film.starring"]):
        return "actor_to_character.portrayal"
    if any(token in joined for token in ["governing", "office_holder", "position_to_holder", "jurisdiction_to_officials"]):
        return "jurisdiction_to_role_holder.assignment"
    if answer_type == "organization":
        return "entity_to_organization"
    if answer_type == "person":
        return "entity_to_person"
    return fallback_type


def _answer_strategy_dict(answer_policy: str, temporal_policy: str) -> Dict[str, Any]:
    policy = (answer_policy or "").strip()
    temporal = (temporal_policy or "").strip()
    if temporal in {"current_only", "current_or_recent"}:
        temporal_scope = "current_only"
    elif temporal:
        temporal_scope = temporal
    elif policy in {"current_only", "single_best"}:
        temporal_scope = "none_or_implicit_current"
    else:
        temporal_scope = "none"

    if policy in {"all_valid", "attribute_filtered_subset"}:
        cardinality = "multiple_or_filtered"
    else:
        cardinality = "single"

    if policy == "all_valid":
        action_space_mode = "keep_whole_action_space"
        answering_tendency = "usually_keep_non_conflicting_action_space_results"
        filter_likely_attributes: List[str] = []
    elif policy == "attribute_filtered_subset":
        action_space_mode = "filter_within_action_space"
        answering_tendency = "usually_filter_within_action_space_before_answering"
        filter_likely_attributes = ["time", "title", "type", "location_level"]
    else:
        action_space_mode = "collapse_within_action_space"
        answering_tendency = "usually_converges_to_one_best_supported_answer"
        filter_likely_attributes = []

    selection_rule = {
        "current_only": "Use the selected action space, then prefer the current or most recent valid answer when evidence distinguishes it.",
        "current_or_recent": "Use the selected action space, then prefer the current or most recent valid holder or affiliation.",
        "single_best": "Use the selected action space, then collapse only after comparing the best-supported candidate.",
        "all_valid": "Use the selected action space as the answer set when the surviving candidates all directly satisfy the question.",
        "attribute_filtered_subset": "Use the selected action space, then filter inside it by the key distinguishing attribute before answering.",
    }.get(policy or temporal, "Decide whether to keep, filter, or collapse within the selected action space from the question semantics and surviving candidates.")
    return {
        "answer_count": cardinality,
        "temporal_scope": temporal_scope,
        "action_space_mode": action_space_mode,
        "filter_likely_attributes": filter_likely_attributes,
        "answering_tendency": answering_tendency,
        "selection_rule": selection_rule,
    }


def _action_space_experience(
    question: str,
    primary_relations: List[str],
    candidate_constraint_entities: List[str],
) -> str:
    q = (question or "").lower()
    relation_text = " ".join(primary_relations)
    if any(token in relation_text for token in ["containedby", "contains", "partially_contains", "partially_containedby"]):
        return "Focus on containment-style relations, but choose the path whose whole relation chain best matches the asked location scope."
    if any(token in relation_text for token in ["languages_spoken", "official_language"]):
        return "Focus on language-bearing relations, but choose the path whose whole chain best matches who is speaking or officially using the language."
    if any(token in relation_text for token in ["performance.character", "performance.actor", "film.film.starring"]):
        return "Focus on portrayal relations, but choose the path whose whole chain keeps the work-role semantics aligned with the question."
    if any(token in relation_text for token in ["pro_athlete.teams", "basketball_player_stats.team", "football_player_stats.team", "sports_team_roster.team"]):
        return "Focus on team-affiliation relations, but choose the path whose whole chain best matches the asked role, roster, or time context."
    if any(token in relation_text for token in ["governing", "office_holder", "position_to_holder"]):
        return "Focus on role-holder relations, but choose the path whose whole chain best matches the asked office, title, and governing context."
    if any(token in q for token in ["wife", "husband", "spouse", "married"]):
        return "Focus on spouse and family relations, but choose the path whose whole chain preserves the asked relationship rather than drifting into biography facts."
    if candidate_constraint_entities:
        return "Focus on the most relevant core relations, and prefer paths that preserve the named secondary context across the whole relation chain."
    return "Focus on the most relevant core relations, but judge them within the logic of the whole relation chain and choose the path that best matches the question intent."


def _final_selection_experience(
    question: str,
    primary_relations: List[str],
    answer_policy: str,
    temporal_policy: str,
    candidate_constraint_entities: List[str],
) -> List[str]:
    q = (question or "").lower()
    relation_text = " ".join(primary_relations)

    filter_attribute = ""
    if any(token in relation_text for token in ["pro_athlete.teams", "basketball_player_stats.team", "football_player_stats.team"]):
        filter_attribute = "date or recency evidence"
    elif any(token in relation_text for token in ["performance.character", "performance.actor", "film.film.starring"]):
        filter_attribute = "the named work, role, or character context"
    elif any(token in relation_text for token in ["governing", "office_holder", "position_to_holder"]):
        filter_attribute = "office title or temporal evidence"
    elif candidate_constraint_entities:
        filter_attribute = "the named secondary context"

    if answer_policy == "all_valid":
        experiences: List[str] = ["This kind of question often keeps the selected action space as the answer set when the surviving results are all valid."]
    elif answer_policy == "attribute_filtered_subset" or filter_attribute:
        experiences = [f"This kind of question often filters within the selected action space using {filter_attribute or 'the key distinguishing attribute'} before answering."]
    else:
        experiences = ["This kind of question often collapses within the selected action space to one best-supported answer."]

    if temporal_policy not in {"", "none"} or any(
        token in q for token in [" current", " now", " latest", " most recent", " first", " earliest"]
    ) or re.search(r"\b(19|20)\d{2}\b", q):
        experiences.append(
            "Check temporal evidence first; prefer current, latest, first, or year-specific answers only when the graph actually distinguishes them."
        )

    if candidate_constraint_entities:
        joined = ", ".join(candidate_constraint_entities[:3])
        experiences.append(
            f"Use the named secondary context ({joined}) to judge which survivors in the chosen action space still match the question."
        )

    if any(token in relation_text for token in ["containedby", "contains"]) or any(
        token in q for token in ["where ", " located", " county", " city", " country"]
    ):
        experiences.append(
            "Match the kept answer level to the question phrasing instead of returning every nearby location in the same space."
        )

    if any(token in relation_text for token in ["languages_spoken", "official_language"]):
        experiences.append(
            "Decide whether to keep the whole language inventory or narrow it by the question wording, not by raw relation overlap alone."
        )

    deduped: List[str] = []
    seen = set()
    for item in experiences:
        text = str(item or "").strip()
        if text and text not in seen:
            deduped.append(text)
            seen.add(text)
    return deduped[:4]


def _looks_too_explanatory(text: str, *, max_words: int) -> bool:
    tokens = str(text or "").strip().split()
    lowered = str(text or "").lower()
    if not tokens:
        return True
    if len(tokens) > max_words:
        return True
    if any(marker in lowered for marker in [" because ", " which ", " implying ", " the question asks", "this means"]):
        return True
    return False


def _coerce_action_space_experience(text: str, fallback: str) -> str:
    candidate = " ".join(str(text or "").strip().split())
    return fallback if _looks_too_explanatory(candidate, max_words=32) else candidate


def _coerce_final_selection_experience(items: List[str], fallback: List[str]) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    for item in items:
        text = " ".join(str(item or "").strip().split())
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    if len(cleaned) < 2 or len(cleaned) > 4:
        return fallback[:4]
    if any(_looks_too_explanatory(text, max_words=22) for text in cleaned):
        return fallback[:4]
    return cleaned[:4]


def _guess_constraint_relations(question: str, primary_relations: List[str]) -> List[str]:
    q = (question or "").lower()
    joined = " ".join(primary_relations)
    guessed: List[str] = []
    if any(token in q for token in [" brother", " sister", " sibling"]) and "people.person.sibling_s" in joined:
        guessed.append("people.person.gender")
    if "character" in q and "star wars" in q:
        guessed.append("film.film.title")
    if any(token in q for token in ["prime minister", "president", "mayor", "governor"]):
        guessed.append("government.government_position_held.basic_title")
    return guessed


def _guess_constraint_entities(question: str) -> List[str]:
    q = (question or "").strip()
    lowered = q.lower()
    guessed: List[str] = []
    if "star wars" in lowered:
        guessed.append("Star Wars")
    return guessed


def _stable_constraint_entities_from_question(source_card: Dict[str, Any]) -> List[str]:
    explicit = _to_list(source_card.get("constraint_entities_seen"))
    if explicit:
        return explicit
    return _guess_constraint_entities(source_card.get("question_text", ""))


async def _generate_skill_card_from_fixed_relation(
    *,
    question: str,
    ground_truth_answers: List[str],
    chosen_relation: str,
    relation_selection_source: str,
    top_relation_candidates: List[Dict[str, Any]],
    fallback: CaseSkillCard,
    candidate_constraint_entities: List[str],
    available_relations: List[str],
) -> Dict[str, Any]:
    prompt_payload = {
        "task": "Generate a single-case KGQA skill card after the core relation has already been fixed by graph validation.",
        "rules": [
            "Return JSON only.",
            "The core relation is already fixed. Do not replace it.",
            "Use the scored relation candidates only as context for why this relation is the reusable answer-bearing one.",
            "question_type should follow the provided relation-oriented canonical rule.",
            "retrieval_fields should stay semantic and reusable.",
            "constraint_guidance should describe concise filtering guidance in English.",
            "candidate_constraint_relations are optional and should be conservative. Leave them empty if backend dynamic filtering is more appropriate.",
            "candidate_constraint_entities should list only stable named secondary context from the question. Do not include anchors or answers.",
            "action_space_experience should explain how to focus on the selected relation family without turning it into a hard rule.",
            "final_selection_experience should be 2-4 short imperative-style English statements.",
            "answer_strategy must keep action_space_mode as one of three fixed modes: keep_whole_action_space, filter_within_action_space, collapse_within_action_space.",
            "filter_likely_attributes and selection_rule should be grounded in the fixed relation and the question wording.",
            "Do not add extra core_relations beyond the fixed one.",
            "Use concise English.",
        ],
        "json_schema": {
            "question_type": "string",
            "retrieval_fields": {
                "semantic_intent": "string",
                "target_answer_type": "string",
                "scope_mode": "string",
                "anchor_pattern": "string",
                "domains": ["string"],
            },
            "constraint_guidance": ["string"],
            "candidate_constraint_relations": ["string"],
            "candidate_constraint_entities": ["string"],
            "action_space_experience": "string",
            "final_selection_experience": ["string"],
            "answer_strategy": {
                "answer_count": "single|multiple_or_filtered",
                "temporal_scope": "string",
                "action_space_mode": "keep_whole_action_space|filter_within_action_space|collapse_within_action_space",
                "filter_likely_attributes": ["string"],
                "answering_tendency": "string",
                "selection_rule": "string",
            },
            "common_pitfalls": ["string"],
            "notes": "string",
        },
        "question": question,
        "ground_truth_answers": ground_truth_answers,
        "fixed_core_relation": chosen_relation,
        "fixed_core_relation_domain": chosen_relation.split(".", 1)[0] if "." in chosen_relation else "",
        "relation_selection_source": relation_selection_source,
        "top_relation_candidates": [
            {
                "relation": row.get("relation"),
                "coverage_ratio": row.get("coverage_ratio"),
                "ground_truth_hit_count": row.get("ground_truth_hit_count"),
                "ground_truth_hits": row.get("ground_truth_hits"),
                "anchor": row.get("anchor"),
                "num_action_hints": row.get("num_action_hints"),
                "action_preview": [
                    " -> ".join(f"{step['relation']}[{step['direction']}]" for step in action.get("steps", []))
                    for action in row.get("action_evaluations", [])[:3]
                ],
            }
            for row in top_relation_candidates
        ],
        "candidate_constraint_entities_hint": candidate_constraint_entities,
        "available_full_relations_from_case": available_relations,
        "question_type_rules": QUESTION_TYPE_RULES,
        "fallback": {
            "question_type": fallback.question_type,
            "retrieval_fields": fallback.retrieval_fields,
            "constraint_guidance": fallback.constraint_guidance,
            "candidate_constraint_relations": fallback.candidate_constraint_relations,
            "candidate_constraint_entities": fallback.candidate_constraint_entities,
            "action_space_experience": fallback.action_space_experience,
            "final_selection_experience": fallback.final_selection_experience,
            "answer_strategy": fallback.answer_strategy,
            "common_pitfalls": fallback.common_pitfalls,
            "notes": fallback.notes,
        },
    }
    text = await _call_llm(
        [
            {"role": "system", "content": "You are curating KGQA case skills with fixed core relations. Return JSON only."},
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False, indent=2)},
        ]
    )
    return _extract_json_object(text) or {}


def _infer_temporal_policy_from_question(question: str) -> str:
    q = (question or "").lower()
    if re.search(r"\b(19|20)\d{2}\b", q):
        return "year_specific"
    if any(token in q for token in [" current ", " now", " today", "present", "currently"]):
        return "current_only"
    if " first " in q:
        return "first_only"
    if any(token in q for token in [" latest ", " newest ", " most recent "]):
        return "latest_only"
    return "none"


def _infer_answer_count_from_question(question: str) -> str:
    q = (question or "").lower()
    if any(token in q for token in ["how many", "which teams", "what teams", "list", "all "]):
        return "multiple_or_filtered"
    return "single"


def _fallback_relation_candidates(
    *,
    source_card: Optional[Dict[str, Any]] = None,
    dataset_case: Optional[Dict[str, Any]] = None,
) -> List[str]:
    relations: List[str] = []
    if dataset_case:
        gt = dataset_case.get("ground_truth", {}) or {}
        relations.extend(_to_list(gt.get("oracle_relations")))
        relations.extend(_to_list(gt.get("core_relations")))
    if source_card:
        prompt_context = source_card.get("prompt_context", {}) or {}
        relations.extend(_to_list(prompt_context.get("core_relations")))
    deduped: List[str] = []
    seen = set()
    for relation in relations:
        rel = str(relation or "").strip()
        if rel and rel not in seen:
            deduped.append(rel)
            seen.add(rel)
    return deduped


def _fallback_case_skill_from_dataset_case(dataset_case: Dict[str, Any]) -> CaseSkillCard:
    gt = dataset_case.get("ground_truth", {}) or {}
    case_id = str(dataset_case.get("id", ""))
    question = extract_question_surface(
        next((msg.get("content", "") for msg in dataset_case.get("messages", []) if msg.get("role") == "user"), "")
    )
    local_source_card = _load_local_source_card(case_id)
    local_raw_attempts = _load_local_raw_attempts_for_source(local_source_card)
    trajectory_candidates = _trajectory_relation_candidates(
        question,
        source_card=local_source_card,
        raw_attempts=local_raw_attempts,
        dataset_case=dataset_case,
    )
    model_candidates = trajectory_candidates or _dataset_prompt_relation_candidates(dataset_case)
    available_relations = model_candidates or _fallback_relation_candidates(source_card=local_source_card, dataset_case=dataset_case)
    preferred_relations = model_candidates or (_to_list(gt.get("core_relations")) + _to_list(gt.get("oracle_relations")))
    primary_relations = _rank_core_relations_by_reward(
        case_id,
        question,
        available_relations,
        preferred_order=preferred_relations,
    )
    primary_relations = _limit_answer_bearing_relations(primary_relations, max_relations=1)
    answer_type = "entity"
    answers = _to_list(gt.get("global_truth_answers"))
    if answers:
        answer_type = "organization" if "team" in question.lower() else "string"
    temporal_policy = _infer_temporal_policy_from_question(question)
    answer_count = _infer_answer_count_from_question(question)
    answer_policy = temporal_policy if temporal_policy != "none" else ("all_valid" if answer_count == "multiple_or_filtered" else "single_best")
    question_type_label = _relation_oriented_question_type(
        primary_relations,
        answer_type=answer_type,
        temporal_policy=temporal_policy,
        fallback_type="entity_to_entity.lookup",
    )
    core_relation_domains = _derive_domains_from_relations(primary_relations)
    candidate_constraint_relations: List[str] = []
    candidate_constraint_entities = _guess_constraint_entities(question)
    action_space_experience = _action_space_experience(question, primary_relations, candidate_constraint_entities)
    final_selection_experience = _final_selection_experience(
        question,
        primary_relations,
        answer_policy,
        temporal_policy,
        candidate_constraint_entities,
    )
    relation_text = " ".join(primary_relations)
    intent_clarification = "This question is asking for the answer that directly fits the requested slot, not every related entity reachable through nearby paths."
    common_misreadings: List[str] = []
    wrong_families: List[str] = []
    if any(token in relation_text for token in ["containedby", "contains", "partially_contains", "partially_containedby"]):
        common_misreadings.append("Confusing the asked containing location with member locations from the same geographic action space.")
        wrong_families.append("member locations or nearby geographic entities instead of the intended location level")
    if any(token in relation_text for token in ["languages_spoken", "official_language"]):
        common_misreadings.append("Treating the retrieved language inventory as a forced single-answer question without checking the wording.")
        wrong_families.append("broader country-language facts instead of the intended language answer")
    return CaseSkillCard(
        case_id=str(dataset_case.get("id", "unknown")),
        question=question,
        question_type=question_type_label,
        retrieval_fields={
            "semantic_intent": question_type_label,
            "target_answer_type": answer_type,
            "scope_mode": temporal_policy if temporal_policy != "none" else answer_count,
            "anchor_pattern": "single_anchor" if len(_to_list(gt.get("core_entities"))) <= 1 else "multi_anchor",
            "domains": core_relation_domains,
        },
        core_relation_domains=core_relation_domains,
        core_relations=primary_relations,
        constraint_guidance=["If too many candidates survive, prefer discriminative time, title, or type attributes."],
        answer_strategy=_answer_strategy_dict(answer_policy, temporal_policy),
        action_space_experience=action_space_experience,
        final_selection_experience=final_selection_experience,
        candidate_constraint_relations=candidate_constraint_relations,
        candidate_constraint_entities=candidate_constraint_entities,
        common_pitfalls=[],
        intent_clarification=intent_clarification,
        common_misreadings=common_misreadings,
        wrong_but_related_answer_families=wrong_families,
        notes="dataset-only fallback skill",
    )


def _fallback_case_skill(source_card: Dict[str, Any], atomic_card: Dict[str, Any], dataset_case: Optional[Dict[str, Any]] = None) -> CaseSkillCard:
    relation_profile = atomic_card.get("relation_profile", {})
    constraint_profile = atomic_card.get("constraint_profile", {})
    answer_profile = atomic_card.get("answer_profile", {})
    question_type = atomic_card.get("question_type", {})

    raw_type = str(question_type.get("candidate_subtype") or question_type.get("parent_type") or "untyped").strip()
    temporal_policy = str(constraint_profile.get("temporal_policy", "")).strip()
    answer_policy = str(answer_profile.get("answer_scope_policy", "")).strip()
    raw_attempts = []
    record_ids = set(_to_list(source_card.get("raw_attempt_ids")))
    if record_ids:
        raw_root = Path("/zhaoshu/subgraph/skills/raw_materials")
        for path in raw_root.rglob("*.json"):
            try:
                payload = read_json(path)
            except Exception:
                continue
            if payload.get("record_id") in record_ids:
                raw_attempts.append(payload)
    primary_relations = _normalize_full_relations(_to_list(relation_profile.get("primary_relations")), source_card, dataset_case)
    trajectory_candidates = _normalize_full_relations(
        _trajectory_relation_candidates(
            source_card["question_text"],
            source_card=source_card,
            raw_attempts=raw_attempts,
            dataset_case=dataset_case,
        ),
        source_card,
        dataset_case,
    )
    if trajectory_candidates:
        primary_relations = trajectory_candidates
    elif not primary_relations and dataset_case:
        primary_relations = _normalize_full_relations(
            _fallback_relation_candidates(source_card=source_card, dataset_case=dataset_case),
            source_card,
            dataset_case,
        )
    if dataset_case:
        gt = dataset_case.get("ground_truth", {}) or {}
        primary_relations = _rank_core_relations_by_reward(
            str(dataset_case.get("id", source_card.get("case_id", ""))),
            source_card["question_text"],
            primary_relations,
            preferred_order=_to_list(gt.get("core_relations")) + _to_list(gt.get("oracle_relations")) + primary_relations,
        )
    answer_type = str(answer_profile.get("answer_type", "entity")).strip() or "entity"
    question_type_label = _relation_oriented_question_type(
        primary_relations,
        answer_type=answer_type,
        temporal_policy=temporal_policy,
        fallback_type=raw_type,
    )

    retrieval_fields = {
        "semantic_intent": question_type_label,
        "target_answer_type": answer_type,
        "scope_mode": temporal_policy or answer_policy or "undetermined",
        "anchor_pattern": str((atomic_card.get("question_fields", {}) or {}).get("anchor_pattern", "unknown")),
        "domains": _to_list(relation_profile.get("primary_relation_domains")),
    }
    primary_relations = _limit_answer_bearing_relations(primary_relations, max_relations=1)
    core_relation_domains = _derive_domains_from_relations(primary_relations)

    constraint_guidance = _to_list(constraint_profile.get("candidate_constraint_relations"))
    if not constraint_guidance:
        if temporal_policy in {"current_only", "current_or_recent"}:
            constraint_guidance = ["If the question does not explicitly request full history, prefer current or most recent time constraints."]
        else:
            constraint_guidance = ["If too many candidates survive, prefer discriminative attribute relations."]

    strategy = _answer_strategy_dict(answer_policy, temporal_policy)
    candidate_constraint_relations = _normalize_full_relations(
        _to_list(source_card.get("constraint_relations_seen")),
        source_card,
        dataset_case,
    )
    candidate_constraint_entities = _stable_constraint_entities_from_question(source_card)
    action_space_experience = _action_space_experience(
        source_card["question_text"],
        primary_relations,
        candidate_constraint_entities,
    )
    final_selection_experience = _final_selection_experience(
        source_card["question_text"],
        primary_relations,
        answer_policy,
        temporal_policy,
        candidate_constraint_entities,
    )
    relation_text = " ".join(primary_relations)
    intent_clarification = "This question is asking for the answer that best fits the requested slot under the selected action space."
    common_misreadings = _to_list(atomic_card.get("common_errors"))
    wrong_families: List[str] = []
    if any(token in relation_text for token in ["containedby", "contains", "partially_contains", "partially_containedby"]):
        common_misreadings.append("Confusing container answers with member entities from the same geographic space.")
        wrong_families.append("member locations or nearby geographic scopes")
    if any(token in relation_text for token in ["governing", "office_holder", "position_to_holder"]):
        common_misreadings.append("Keeping historical holders when the question is really asking for the currently relevant holder.")
        wrong_families.append("historical or adjacent office holders")
    if any(token in relation_text for token in ["languages_spoken", "official_language"]):
        common_misreadings.append("Returning the whole language inventory when the wording points to a narrower answer.")
        wrong_families.append("broader nationality or country attributes")

    return CaseSkillCard(
        case_id=source_card["case_id"],
        question=source_card["question_text"],
        question_type=question_type_label,
        retrieval_fields=retrieval_fields,
        core_relation_domains=core_relation_domains,
        core_relations=primary_relations,
        constraint_guidance=constraint_guidance,
        answer_strategy=strategy,
        action_space_experience=action_space_experience,
        final_selection_experience=final_selection_experience,
        candidate_constraint_relations=candidate_constraint_relations,
        candidate_constraint_entities=candidate_constraint_entities,
        common_pitfalls=_to_list(atomic_card.get("common_errors")),
        intent_clarification=intent_clarification,
        common_misreadings=_to_list(common_misreadings),
        wrong_but_related_answer_families=wrong_families,
        notes=str(atomic_card.get("extraction_notes", "")).strip(),
    )


def _load_raw_attempts(skills_root: Path, source_card: Dict[str, Any]) -> List[Dict[str, Any]]:
    record_ids = set(_to_list(source_card.get("raw_attempt_ids")))
    if not record_ids:
        return []
    attempts: List[Dict[str, Any]] = []
    raw_root = skills_root / "raw_materials"
    for path in raw_root.rglob("*.json"):
        payload = read_json(path)
        if payload.get("record_id") in record_ids:
            attempts.append(payload)
    attempts.sort(key=lambda item: str(item.get("created_at", "")))
    return attempts


async def synthesize_case_skill(
    source_card: Dict[str, Any],
    atomic_card: Dict[str, Any],
    *,
    skills_root: Path,
    dataset_case: Optional[Dict[str, Any]] = None,
    use_llm: bool = True,
) -> CaseSkillCard:
    fallback = _fallback_case_skill(source_card, atomic_card, dataset_case)
    if not use_llm:
        return fallback

    raw_attempts = _load_raw_attempts(skills_root, source_card)
    condensed_attempts = [
        {
            "record_id": item.get("record_id"),
            "variant": item.get("variant"),
            "success": item.get("success"),
            "ground_truth_answers": item.get("ground_truth_answers"),
            "predicted_answers": item.get("predicted_answers"),
            "planned_relations": item.get("planned_relations"),
            "candidate_constraint_relations": item.get("candidate_constraint_relations"),
            "candidate_constraint_entities": item.get("candidate_constraint_entities"),
            "error_text": item.get("error_text"),
            "smoke_excerpt": (item.get("smoke_excerpt") or "")[:2500],
        }
        for item in raw_attempts[-3:]
    ]

    prompt_payload = {
        "task": "Synthesize one single-case KGQA skill card for retrieval-time sampling. This is a reflection pass: use the question, final answer evidence, and answer-bearing relation evidence to decide the true reusable experience for this case.",
        "rules": [
            "Return JSON only.",
            "This is a case skill, not a cluster skill.",
            "Keep the original question string because later retrieval will compare question fields across case skills.",
            "Do not mention provenance, source_attempt_ids, or extraction confidence.",
            "question_type should follow the provided relation-oriented canonical rule.",
            "retrieval_fields should be semantic retrieval keys, not raw surface cues: semantic_intent, target_answer_type, scope_mode, anchor_pattern, domains.",
            "core_relations must keep only answer-bearing relations for this case.",
            "Prefer exactly one true answer-bearing relation.",
            "Use two relations only if both are necessary answer-bearing relations in the same semantic chain.",
            "Never dump all plausible or neighboring relations into core_relations.",
            "If one relation is clearly the true answer-bearing relation and others are only exploratory backups, keep only the true answer-bearing relation in core_relations.",
            "core_relations must be fully qualified relations with domain included, e.g. sports.pro_athlete.teams.",
            "Return exactly 1 core_relation whenever possible.",
            "Only return more than 1 core_relation if the answer-bearing signal is truly inseparable; otherwise keep the single best relation only.",
            "core_relation_domains must be only the domains of the kept core_relations. Do not add any extra domains.",
            "Do not keep graph-near but semantically off-target relations in core_relations.",
            "Use the final answer or ground truth to reflect which relation actually leads to the answer type. Do not just copy failed trajectory relations.",
            "constraint_guidance should describe what kind of constraint is usually needed here, not repeat tool syntax.",
            "candidate_constraint_relations are optional and should be conservative. Leave them empty if backend dynamic filtering is more appropriate.",
            "candidate_constraint_entities should list only second-entity/title/franchise reminders explicitly mentioned by the question. Do NOT include the anchor entity or answer candidates.",
            "action_space_experience should describe which core relations deserve attention and remind the model to judge them within the logic of the whole relation chain.",
            "Do not turn action_space_experience into a hard rule or a state machine.",
            "Prefer wording like: focus on the most relevant core relations, but choose the path whose whole chain best matches the question intent.",
            "final_selection_experience should be 2-4 short English statements (each up to ~15 words).",
            "Use imperative or preference style such as: Prefer..., Filter by..., Keep..., Do not force collapse....",
            "Do not write explanatory prose, chain-of-thought, or long because-style justifications.",
            "final_selection_experience should explain what this question is really asking for and how answers of this kind are usually resolved inside the selected action space.",
            "Describe whether this kind of question usually keeps the selected action space, filters within it, or collapses within it.",
            "final_selection_experience should mention temporal tendency (current/latest/year-specific) only when the case actually involves temporal scope.",
            "Negative or failed attempts are weak optional references only. Prioritize question, final answer evidence, and answer-bearing relations.",
            "answer_strategy must stay lightweight and structured for compatibility, but it should now emphasize action_space_mode, filter_likely_attributes, and answering_tendency in addition to the legacy keys.",
            "Use concise English only. No Chinese.",
        ],
        "json_schema": {
            "question_type": "string",
            "retrieval_fields": {
                "semantic_intent": "string",
                "target_answer_type": "string",
                "scope_mode": "string",
                "anchor_pattern": "string",
                "domains": ["string"],
            },
            "core_relation_domains": ["string"],
            "core_relations": ["string"],
            "constraint_guidance": ["string"],
            "candidate_constraint_relations": ["string"],
            "candidate_constraint_entities": ["string"],
            "action_space_experience": "string",
            "final_selection_experience": ["string"],
            "answer_strategy": {
                "answer_count": "single|multiple_or_filtered",
                "temporal_scope": "string",
                "action_space_mode": "keep_whole_action_space|filter_within_action_space|collapse_within_action_space",
                "filter_likely_attributes": ["string"],
                "answering_tendency": "string",
                "selection_rule": "string",
            },
            "common_pitfalls": ["string"],
            "notes": "string",
        },
        "source_card": source_card,
        "atomic_card": atomic_card,
        "dataset_case": dataset_case or {},
        "recent_raw_attempts": condensed_attempts,
        "available_full_relations_from_case": _available_full_relations(source_card, dataset_case),
        "question_type_rules": QUESTION_TYPE_RULES,
        "fallback": {
            "question_type": fallback.question_type,
            "retrieval_fields": fallback.retrieval_fields,
            "core_relation_domains": fallback.core_relation_domains,
            "core_relations": fallback.core_relations,
            "constraint_guidance": fallback.constraint_guidance,
            "candidate_constraint_relations": fallback.candidate_constraint_relations,
            "candidate_constraint_entities": fallback.candidate_constraint_entities,
            "action_space_experience": fallback.action_space_experience,
            "final_selection_experience": fallback.final_selection_experience,
            "answer_strategy": fallback.answer_strategy,
            "common_pitfalls": fallback.common_pitfalls,
            "notes": fallback.notes,
        },
    }
    text = await _call_llm(
        [
            {"role": "system", "content": "You are curating single-case KGQA skill cards. Return JSON only."},
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False, indent=2)},
        ]
    )
    payload = _extract_json_object(text)
    if not payload:
        return fallback

    trajectory_candidates = _normalize_full_relations(
        _trajectory_relation_candidates(
            source_card["question_text"],
            source_card=source_card,
            raw_attempts=raw_attempts,
            dataset_case=dataset_case,
        ),
        source_card,
        dataset_case,
    )
    normalized_core_relations = _normalize_full_relations(_to_list(payload.get("core_relations")), source_card, dataset_case)
    if dataset_case:
        gt = dataset_case.get("ground_truth", {}) or {}
        available_relations = _available_full_relations(source_card, dataset_case)
        selection = await _choose_core_relation_v5(
            case_id=str(dataset_case.get("id", source_card["case_id"])),
            question=source_card["question_text"],
            ground_truth_answers=_to_list(gt.get("global_truth_answers")),
            plan_relation_candidates=trajectory_candidates or normalized_core_relations,
            shortest_path_relation_candidates=_shortest_path_core_relation_candidates_for_case(str(dataset_case.get("id", source_card["case_id"]))),
        )
        chosen_relation = str(selection.get("selected_relation") or "").strip() or (
            fallback.core_relations[0] if fallback.core_relations else ""
        )
        fixed_payload = await _generate_skill_card_from_fixed_relation(
            question=source_card["question_text"],
            ground_truth_answers=_to_list(gt.get("global_truth_answers")),
            chosen_relation=chosen_relation,
            relation_selection_source=str(selection.get("selection_source") or "fixed"),
            top_relation_candidates=selection.get("top_candidates") or [],
            fallback=fallback,
            candidate_constraint_entities=fallback.candidate_constraint_entities,
            available_relations=available_relations or normalized_core_relations,
        )
        payload = fixed_payload or payload
        normalized_core_relations = [chosen_relation] if chosen_relation else fallback.core_relations
    else:
        normalized_core_relations = _limit_answer_bearing_relations(
            trajectory_candidates or normalized_core_relations or fallback.core_relations,
            max_relations=1,
        )

    card = CaseSkillCard(
        case_id=source_card["case_id"],
        question=source_card["question_text"],
        question_type=str(payload.get("question_type") or fallback.question_type),
        retrieval_fields=payload.get("retrieval_fields") or fallback.retrieval_fields,
        core_relation_domains=_derive_domains_from_relations(normalized_core_relations) or fallback.core_relation_domains,
        core_relations=normalized_core_relations,
        constraint_guidance=_to_list(payload.get("constraint_guidance")) or fallback.constraint_guidance,
        answer_strategy=payload.get("answer_strategy") or fallback.answer_strategy,
        action_space_experience=_coerce_action_space_experience(
            str(payload.get("action_space_experience") or fallback.action_space_experience).strip(),
            fallback.action_space_experience,
        ),
        final_selection_experience=_coerce_final_selection_experience(
            _to_list(payload.get("final_selection_experience")) or fallback.final_selection_experience,
            fallback.final_selection_experience,
        ),
        candidate_constraint_relations=fallback.candidate_constraint_relations,
        candidate_constraint_entities=_to_list(payload.get("candidate_constraint_entities")) or fallback.candidate_constraint_entities,
        intent_clarification="",
        common_misreadings=[],
        wrong_but_related_answer_families=[],
        common_pitfalls=_to_list(payload.get("common_pitfalls")) or fallback.common_pitfalls,
        notes=str(payload.get("notes") or fallback.notes).strip(),
    )
    return card


async def synthesize_case_skill_from_dataset_case(
    dataset_case: Dict[str, Any],
    *,
    use_llm: bool = True,
) -> CaseSkillCard:
    fallback = _fallback_case_skill_from_dataset_case(dataset_case)
    if not use_llm:
        return fallback

    gt = dataset_case.get("ground_truth", {}) or {}
    case_id = str(dataset_case.get("id", fallback.case_id))
    question = fallback.question
    local_source_card = _load_local_source_card(case_id)
    local_raw_attempts = _load_local_raw_attempts_for_source(local_source_card)
    trajectory_candidates = _trajectory_relation_candidates(
        question,
        source_card=local_source_card,
        raw_attempts=local_raw_attempts,
        dataset_case=dataset_case,
    )
    prompt_candidates = _dataset_prompt_relation_candidates(dataset_case)
    model_relation_candidates = _limit_answer_bearing_relations(
        trajectory_candidates or prompt_candidates,
        max_relations=4,
    )
    shortest_path_relation_candidates = _shortest_path_core_relation_candidates_for_case(case_id)
    selection = await _choose_core_relation_v5(
        case_id=case_id,
        question=question,
        ground_truth_answers=_to_list(gt.get("global_truth_answers")),
        plan_relation_candidates=model_relation_candidates,
        shortest_path_relation_candidates=shortest_path_relation_candidates,
    )
    chosen_relation = str(selection.get("selected_relation") or "").strip() or (
        fallback.core_relations[0] if fallback.core_relations else ""
    )
    top_relation_candidates = selection.get("top_candidates") or []
    available_relations = (
        model_relation_candidates
        or shortest_path_relation_candidates
        or _fallback_relation_candidates(source_card=local_source_card, dataset_case=dataset_case)
        or _to_list(gt.get("oracle_relations"))
        or _to_list(gt.get("core_relations"))
    )
    payload = await _generate_skill_card_from_fixed_relation(
        question=question,
        ground_truth_answers=_to_list(gt.get("global_truth_answers")),
        chosen_relation=chosen_relation,
        relation_selection_source=str(selection.get("selection_source") or "fixed"),
        top_relation_candidates=top_relation_candidates,
        fallback=fallback,
        candidate_constraint_entities=fallback.candidate_constraint_entities,
        available_relations=available_relations,
    )

    card = CaseSkillCard(
        case_id=str(dataset_case.get("id", fallback.case_id)),
        question=question,
        question_type=str(payload.get("question_type") or fallback.question_type),
        retrieval_fields=payload.get("retrieval_fields") or fallback.retrieval_fields,
        core_relation_domains=_derive_domains_from_relations([chosen_relation]) or fallback.core_relation_domains,
        core_relations=[chosen_relation] if chosen_relation else fallback.core_relations,
        constraint_guidance=_to_list(payload.get("constraint_guidance")) or fallback.constraint_guidance,
        answer_strategy=payload.get("answer_strategy") or fallback.answer_strategy,
        action_space_experience=_coerce_action_space_experience(
            str(payload.get("action_space_experience") or fallback.action_space_experience).strip(),
            fallback.action_space_experience,
        ),
        final_selection_experience=_coerce_final_selection_experience(
            _to_list(payload.get("final_selection_experience")) or fallback.final_selection_experience,
            fallback.final_selection_experience,
        ),
        candidate_constraint_relations=[
            _expand_relation_to_full(relation, available_relations) for relation in _to_list(payload.get("candidate_constraint_relations"))
            if _expand_relation_to_full(relation, available_relations)
        ] or fallback.candidate_constraint_relations,
        candidate_constraint_entities=_to_list(payload.get("candidate_constraint_entities")) or fallback.candidate_constraint_entities,
        intent_clarification="",
        common_misreadings=[],
        wrong_but_related_answer_families=[],
        common_pitfalls=_to_list(payload.get("common_pitfalls")) or fallback.common_pitfalls,
        notes="; ".join(
            part for part in [
                str(payload.get("notes") or fallback.notes).strip(),
                f"relation_selection_source={selection.get('selection_source')}",
            ] if part
        ),
    )
    card.core_relations = _limit_answer_bearing_relations(card.core_relations or fallback.core_relations, max_relations=1)
    card.core_relation_domains = _derive_domains_from_relations(card.core_relations) or fallback.core_relation_domains
    return card


def _resolve_case_skill_output_root(skills_root: Path) -> Path:
    """Resolve where case-skill JSON/MD files should be written.

    Supports both legacy roots like `skills/` and direct corpora roots like
    `skills/webqsp_train_case_skills_en/`.
    """
    if skills_root.name == "case_skills":
        return skills_root
    if (skills_root / "index.json").exists():
        return skills_root
    return skills_root / "case_skills"


def upsert_case_skill_outputs(skills_root: Path, card: CaseSkillCard) -> Dict[str, str]:
    case_skill_root = _resolve_case_skill_output_root(skills_root)
    case_skill_root.mkdir(parents=True, exist_ok=True)
    json_path = case_skill_root / f"{card.case_id}.json"
    md_path = case_skill_root / f"{card.case_id}.md"
    write_json(json_path, card)
    md_path.write_text(render_case_skill_markdown(card), encoding="utf-8")

    index_path = case_skill_root / "index.json"
    index_payload: List[Dict[str, Any]] = []
    if index_path.exists():
        raw_index = read_json(index_path)
        if isinstance(raw_index, list):
            index_payload = raw_index
    updated_entry = {
        "case_id": card.case_id,
        "question": card.question,
        "question_type": card.question_type,
        "retrieval_fields": card.retrieval_fields,
        "core_relation_domains": card.core_relation_domains,
        "core_relations": card.core_relations,
        "answer_strategy": card.answer_strategy,
        "action_space_experience": card.action_space_experience,
        "final_selection_experience": card.final_selection_experience,
        "candidate_constraint_relations": card.candidate_constraint_relations,
        "candidate_constraint_entities": card.candidate_constraint_entities,
        "intent_clarification": card.intent_clarification,
        "common_misreadings": card.common_misreadings,
        "wrong_but_related_answer_families": card.wrong_but_related_answer_families,
        "md_path": str(md_path),
        "json_path": str(json_path),
    }
    filtered = [item for item in index_payload if str(item.get("case_id")) != card.case_id]
    filtered.append(updated_entry)
    filtered.sort(key=lambda item: str(item.get("case_id", "")))
    write_json(index_path, filtered)
    return {"json_path": str(json_path), "md_path": str(md_path), "index_path": str(index_path)}


def render_case_skill_markdown(card: CaseSkillCard) -> str:
    strategy = card.answer_strategy or {}
    lines = [
        f"# {card.case_id}",
        "",
        "## Question",
        "",
        f"- Original question: `{card.question}`",
        f"- Question type: `{card.question_type}`",
        "",
        "## Core Relation",
        "",
    ]
    if card.core_relations:
        lines.extend([f"- `{relation}`" for relation in card.core_relations])
    else:
        lines.append("- Not stabilized")
    lines.extend(
        [
            "",
            "## Core Strategy",
            "",
            f"- Action-space mode: {strategy.get('action_space_mode', '')}",
            f"- Filter likely attributes: {', '.join(_to_list(strategy.get('filter_likely_attributes', []))) or 'none'}",
            f"- Selection rule: {strategy.get('selection_rule', '')}",
        ]
    )
    if card.notes:
        lines.extend(
            [
                "",
                "## Notes",
                "",
                f"- {card.notes}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


async def build_case_skill_outputs(
    skills_root: Path,
    *,
    use_llm: bool = True,
    case_ids: List[str] | None = None,
    dataset_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    concurrency: int = 4,
) -> Dict[str, Any]:
    source_root = skills_root / "source_cards"
    atomic_root = skills_root / "atomic_cards"
    case_skill_root = skills_root / "case_skills"
    case_skill_root.mkdir(parents=True, exist_ok=True)

    selected = set(case_ids or [])
    written: List[str] = []
    index: List[Dict[str, Any]] = []

    work_items: List[tuple[str, Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]] = []
    for atomic_path in sorted(atomic_root.glob("*.json")):
        atomic_card = read_json(atomic_path)
        case_id = atomic_card.get("case_id", "")
        if not case_id:
            continue
        if selected and case_id not in selected:
            continue
        source_path = source_root / f"{case_id}.json"
        if not source_path.exists():
            continue
        source_card = read_json(source_path)
        dataset_case = dataset_lookup.get(case_id) if dataset_lookup else None
        work_items.append((case_id, source_card, atomic_card, dataset_case))

    semaphore = asyncio.Semaphore(max(1, int(concurrency or 1)))

    async def _build_one(
        case_id: str,
        source_card: Dict[str, Any],
        atomic_card: Dict[str, Any],
        dataset_case: Optional[Dict[str, Any]],
    ) -> tuple[str, CaseSkillCard]:
        async with semaphore:
            card = await synthesize_case_skill(
                source_card,
                atomic_card,
                skills_root=skills_root,
                dataset_case=dataset_case,
                use_llm=use_llm,
            )
            return case_id, card

    built_cards = await asyncio.gather(
        *(
            _build_one(case_id, source_card, atomic_card, dataset_case)
            for case_id, source_card, atomic_card, dataset_case in work_items
        )
    )

    for case_id, card in built_cards:
        json_path = case_skill_root / f"{case_id}.json"
        md_path = case_skill_root / f"{case_id}.md"
        write_json(json_path, card)
        md_path.write_text(render_case_skill_markdown(card), encoding="utf-8")
        written.extend([str(json_path), str(md_path)])
        index.append(
            {
                "case_id": card.case_id,
                "question": card.question,
                "question_type": card.question_type,
                "retrieval_fields": card.retrieval_fields,
                "core_relation_domains": card.core_relation_domains,
                "core_relations": card.core_relations,
                "answer_strategy": card.answer_strategy,
                "action_space_experience": card.action_space_experience,
                "final_selection_experience": card.final_selection_experience,
                "candidate_constraint_relations": card.candidate_constraint_relations,
                "candidate_constraint_entities": card.candidate_constraint_entities,
                "intent_clarification": card.intent_clarification,
                "common_misreadings": card.common_misreadings,
                "wrong_but_related_answer_families": card.wrong_but_related_answer_families,
                "md_path": str(md_path),
                "json_path": str(json_path),
            }
        )

    index_path = case_skill_root / "index.json"
    write_json(index_path, index)
    return {
        "cases": len(index),
        "written": written,
        "index_path": str(index_path),
        "case_skill_root": str(case_skill_root),
    }


def build_case_skill_outputs_sync(
    skills_root: Path,
    *,
    use_llm: bool = True,
    case_ids: List[str] | None = None,
    dataset_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    concurrency: int = 4,
) -> Dict[str, Any]:
    return asyncio.run(
        build_case_skill_outputs(
            skills_root,
            use_llm=use_llm,
            case_ids=case_ids,
            dataset_lookup=dataset_lookup,
            concurrency=concurrency,
        )
    )


def sample_case_ids(skills_root: Path, *, sample_size: int, seed: int = 0) -> List[str]:
    atomic_root = skills_root / "atomic_cards"
    case_ids: List[str] = []
    for path in sorted(atomic_root.glob("*.json")):
        case_id = read_json(path).get("case_id", "")
        if case_id:
            case_ids.append(case_id)
    case_ids = sorted(set(case_ids))
    if sample_size >= len(case_ids):
        return case_ids
    rng = random.Random(seed)
    return sorted(rng.sample(case_ids, sample_size))
