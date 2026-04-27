#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp


ROOT = Path(__file__).resolve().parents[1]
LLM_API_URL = "http://localhost:8000/v1/chat/completions"
LLM_MODEL = "qwen35-9b-local"
GTE_API_URL = "http://localhost:8003"
DEFAULT_PILOT_RESULTS = ROOT / "reports" / "stage_pipeline_test" / "find_check_plan_pilot_10cases" / "results.json"
DEFAULT_CWQ_PKL = Path("/zhaoshu/SubgraphRAG-main/retrieve/data_files/cwq/processed/test.pkl")
DEFAULT_OUTPUT = ROOT / "reports" / "stage_pipeline_test" / "constrained_subgraph_pilot_10cases"


SYSTEM_PROMPT = """You analyze a question and decompose it into a structured reasoning chain.

Goal:
- Identify the anchor entity (starting point)
- Break down the question into a sequence of intent steps
- For each intent, suggest relation candidates that might fulfill it
- Identify the target entity if explicitly mentioned in the question

Rules:
- anchor: the explicit entity mentioned in the question to start from
- intent_chain: ordered list of reasoning steps, each with:
  - step: sequential number starting from 1
  - intent: short description of what this step aims to find/verify
  - relation_candidates: up to 3 short phrases describing the relation type needed
- target_entity: the final answer entity if explicitly named (e.g., "Central America"), otherwise null
- Keep relation candidates concise (2-4 words each)
- Output JSON only

Output JSON schema:
{
  "anchor": "entity name",
  "intent_chain": [
    {"step": 1, "intent": "what this step finds", "relation_candidates": ["rel1", "rel2", "rel3"]},
    {"step": 2, "intent": "what this step finds", "relation_candidates": ["rel1", "rel2", "rel3"]}
  ],
  "target_entity": "explicit target" or null
}

Examples:

Question: "Which nation has the Alta Verapaz Department and is in Central America?"
Output: {"anchor": "Alta Verapaz Department", "intent_chain": [{"step":1,"intent":"what nation contains this department","relation_candidates":["department to country","administrative division country","state province country"]},{"step":2,"intent":"verify it is in Central America","relation_candidates":["country continent","country location","country region"]}],"target_entity":"Central America"}

Question: "What language is spoken in the location that appointed Michelle Bachelet to a governmental position speak?"
Output: {"anchor": "Michelle Bachelet", "intent_chain": [{"step":1,"intent":"where was she appointed to governmental position","relation_candidates":["position jurisdiction","office location","government appointment country"]},{"step":2,"intent":"what language is spoken there","relation_candidates":["country languages spoken","country official language","language spoken in country"]}],"target_entity":null}

Question: "Lou Seal is the mascot for the team that last won the World Series when?"
Output: {"anchor": "Lou Seal", "intent_chain": [{"step":1,"intent":"what team has this mascot","relation_candidates":["mascot team","team mascot","mascot belongs to team"]},{"step":2,"intent":"what championship did the team win last","relation_candidates":["team championships","league season championship","world series champion"]}],"target_entity":null}

Question: "Which man is the leader of the country that uses Libya, Libya, Libya as its national anthem?"
Output: {"anchor": "Libya, Libya, Libya", "intent_chain": [{"step":1,"intent":"what country uses this as national anthem","relation_candidates":["anthem country","national anthem of country","country with anthem"]},{"step":2,"intent":"who is the leader of that country","relation_candidates":["country leader","government office holder","head of state"]}],"target_entity":null}

Question: "In which countries do the people speak Portuguese, where the child labor percentage was once 1.8?"
Output: {"anchor": "Portuguese", "intent_chain": [{"step":1,"intent":"which countries speak this language","relation_candidates":["language countries spoken in","official language country","language spoken in country"]},{"step":2,"intent":"filter by child labor percentage","relation_candidates":["country child labor percentage","statistical region child labor","labor statistics"]}],"target_entity":null}
"""


def is_cvt_like(name: str) -> bool:
    return bool(re.match(r"^[mg]\.[A-Za-z0-9_]+$", name))


def rel_to_text(rel: str) -> str:
    parts = rel.split(".")
    if len(parts) >= 2:
        return " ".join(p.replace("_", " ") for p in parts[-2:])
    return rel.replace("_", " ")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def candidate_hit(cands: List[str], targets: List[str]) -> bool:
    norm_cands = [normalize(c) for c in cands]
    for t in targets:
        nt = normalize(t)
        for c in norm_cands:
            if c == nt or nt in c or c in nt:
                return True
    return False


async def call_llm(session: aiohttp.ClientSession, question: str) -> Dict[str, Any]:
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"},
        ],
        "max_tokens": 500,
        "temperature": 0.0,
        "top_p": 0.8,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with session.post(LLM_API_URL, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
        data = await resp.json()
    raw = data["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"raw_output": raw, "anchor": None, "intent_chain": [], "target_entity": None}
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"raw_output": raw, "anchor": None, "intent_chain": [], "target_entity": None}
    parsed["raw_output"] = raw
    # Limit intents and candidates
    parsed["intent_chain"] = parsed.get("intent_chain", [])[:3]
    for intent in parsed["intent_chain"]:
        intent["relation_candidates"] = intent.get("relation_candidates", [])[:3]
    return parsed


async def gte_retrieve(
    session: aiohttp.ClientSession,
    query: str,
    candidates: List[str],
    candidate_texts: List[str] | None = None,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    payload = {
        "query": query,
        "candidates": candidates,
        "candidate_texts": candidate_texts,
        "top_k": top_k,
    }
    async with session.post(
        f"{GTE_API_URL}/retrieve",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=30),
    ) as resp:
        data = await resp.json()
    return data.get("results", [])


async def gte_resolve_anchor(
    session: aiohttp.ClientSession,
    anchor_name: str,
    entity_list: List[str],
    top_k: int = 5,
) -> Optional[int]:
    """Resolve anchor entity via GTE semantic search (defensive).

    Ensures the anchor is semantically verified, not just string-matched.
    Filters out CVT-like entities from candidates.
    """
    candidates = []
    candidate_indices = []
    for idx, name in enumerate(entity_list):
        if name and len(name) > 1 and not is_cvt_like(name):
            candidates.append(name)
            candidate_indices.append(idx)

    if not candidates:
        return None

    rows = await gte_retrieve(session, anchor_name, candidates, top_k=top_k)

    if rows:
        best_name = rows[0]["candidate"]
        for idx, name in zip(candidate_indices, candidates):
            if name == best_name:
                return idx

    return None


def expand_node_with_relations(
    node_idx: int,
    relation_indices: set,
    h_ids: List[int],
    r_ids: List[int],
    t_ids: List[int],
    reverse: bool = False,
) -> List[tuple]:
    """Find all children of node_idx via any of the given relations."""
    children = []
    for i in range(len(h_ids)):
        if reverse:
            if t_ids[i] == node_idx and r_ids[i] in relation_indices:
                children.append((h_ids[i], r_ids[i]))
        else:
            if h_ids[i] == node_idx and r_ids[i] in relation_indices:
                children.append((t_ids[i], r_ids[i]))
    return children


def expand_through_cvt(
    node_idx: int,
    h_ids: List[int],
    r_ids: List[int],
    t_ids: List[int],
    entity_list: List[str],
) -> List[tuple]:
    """Expand one hop through a CVT node to reach the actual entity."""
    node_name = entity_list[node_idx] if 0 <= node_idx < len(entity_list) else ""
    if not is_cvt_like(node_name):
        return []

    # Find all triples connecting through this CVT
    children = []
    seen = set()
    for i in range(len(h_ids)):
        # Forward: CVT -> target
        if h_ids[i] == node_idx and t_ids[i] not in seen:
            children.append((t_ids[i], r_ids[i]))
            seen.add(t_ids[i])
        # Reverse: source -> CVT
        if t_ids[i] == node_idx and h_ids[i] not in seen:
            children.append((h_ids[i], r_ids[i]))
            seen.add(h_ids[i])
    return children


def flexible_expand(
    anchor_idx: int,
    resolved_intents: List[Dict[str, Any]],
    h_ids: List[int],
    r_ids: List[int],
    t_ids: List[int],
    entity_list: List[str],
    max_hops: int = 5,
    beam_width: int = 100,
) -> tuple[List[Dict[str, Any]], int, int]:
    """
    Expand subgraph using ALL intent relations as a pooled set.
    Score paths by intent coverage (how many distinct intent steps are matched).
    If layer 1 fails, layer 2 can still fire — no strict ordering.

    Returns:
        - List of best paths (max intent coverage)
        - Max depth achieved
        - Max intent coverage achieved
    """
    # Build: rel_idx -> set of intent steps it belongs to
    rel_to_intents: Dict[int, set] = {}
    all_rel_indices: set = set()
    for intent in resolved_intents:
        for rel_idx in intent["resolved_relation_indices"]:
            all_rel_indices.add(rel_idx)
            rel_to_intents.setdefault(rel_idx, set()).add(intent["step"])

    if not all_rel_indices:
        return [{"nodes": [anchor_idx], "relations": [], "covered_intents": set(), "depth": 0}], 0, 0

    # BFS with beam search
    paths = [{"nodes": [anchor_idx], "relations": [], "covered_intents": set(), "depth": 0}]

    for hop in range(max_hops):
        new_paths = []
        for path in paths:
            current = path["nodes"][-1]

            # Expand with ALL pooled relations (forward + reverse)
            children = expand_node_with_relations(current, all_rel_indices, h_ids, r_ids, t_ids, reverse=False)
            rev_children = expand_node_with_relations(current, all_rel_indices, h_ids, r_ids, t_ids, reverse=True)
            all_children = children + rev_children

            if all_children:
                seen = set(path["nodes"])
                for child_idx, rel_idx in all_children:
                    if child_idx in seen:
                        continue
                    covered = set(path["covered_intents"])
                    if rel_idx in rel_to_intents:
                        covered |= rel_to_intents[rel_idx]

                    new_path = {
                        "nodes": path["nodes"] + [child_idx],
                        "relations": path["relations"] + [rel_idx],
                        "covered_intents": covered,
                        "depth": path["depth"] + 1,
                    }

                    child_name = entity_list[child_idx] if 0 <= child_idx < len(entity_list) else ""
                    if is_cvt_like(child_name):
                        cvt_children = expand_through_cvt(child_idx, h_ids, r_ids, t_ids, entity_list)
                        new_seen = set(new_path["nodes"])
                        for cvt_idx, cvt_rel in cvt_children:
                            if cvt_idx in new_seen:
                                continue
                            cvt_covered = set(covered)
                            if cvt_rel in rel_to_intents:
                                cvt_covered |= rel_to_intents[cvt_rel]
                            new_paths.append({
                                "nodes": new_path["nodes"] + [cvt_idx],
                                "relations": new_path["relations"] + [cvt_rel],
                                "covered_intents": cvt_covered,
                                "depth": new_path["depth"] + 1,
                            })
                    else:
                        new_paths.append(new_path)
            else:
                # Dead end — keep path as-is
                new_paths.append(path)

        # Beam search: keep top beam_width by coverage, then depth
        new_paths.sort(key=lambda p: (len(p["covered_intents"]), p["depth"]), reverse=True)
        paths = new_paths[:beam_width]

    # Final: keep paths with max intent coverage
    if not paths:
        return [{"nodes": [anchor_idx], "relations": [], "covered_intents": set(), "depth": 0}], 0, 0

    max_coverage = max(len(p["covered_intents"]) for p in paths)
    best = [p for p in paths if len(p["covered_intents"]) == max_coverage]
    max_depth = max(p["depth"] for p in best)

    return best, max_depth, max_coverage


def bidirectional_expand(
    anchor_idx: int,
    target_idx: int,
    prior_rel_indices: set,
    h_ids: List[int],
    r_ids: List[int],
    t_ids: List[int],
    entity_list: List[str],
    max_hops: int = 5,
    beam_width: int = 100,
) -> tuple[List[Dict[str, Any]], int, int]:
    """
    Bidirectional BFS from anchor and target entity.
    When frontiers meet, reconstruct paths and score by prior relation coverage.

    Returns:
        - Best paths (max prior coverage)
        - Max depth
        - Max prior coverage
    """
    # Build adjacency: node_idx -> list of (neighbor_idx, rel_idx)
    adj: Dict[int, List[tuple]] = {}
    for i in range(len(h_ids)):
        h, r, t = h_ids[i], r_ids[i], t_ids[i]
        adj.setdefault(h, []).append((t, r))
        adj.setdefault(t, []).append((h, r))

    # Forward BFS from anchor, backward BFS from target
    # Each frontier: dict of node_idx -> list of paths ending at that node (current hop only)
    fwd_frontier: Dict[int, List[Dict]] = {
        anchor_idx: [{"nodes": [anchor_idx], "relations": [], "depth": 0}]
    }
    bwd_frontier: Dict[int, List[Dict]] = {
        target_idx: [{"nodes": [target_idx], "relations": [], "depth": 0}]
    }
    # Persistent path storage for meeting reconstruction (accumulates across all hops)
    fwd_path_map: Dict[int, List[Dict]] = {
        anchor_idx: [{"nodes": [anchor_idx], "relations": [], "depth": 0}]
    }
    bwd_path_map: Dict[int, List[Dict]] = {
        target_idx: [{"nodes": [target_idx], "relations": [], "depth": 0}]
    }
    fwd_visited: Dict[int, int] = {anchor_idx: 0}  # node -> min depth
    bwd_visited: Dict[int, int] = {target_idx: 0}

    meeting_paths = []

    for hop in range(max_hops):
        # Expand the smaller frontier
        if len(fwd_frontier) <= len(bwd_frontier):
            # Expand forward
            new_frontier: Dict[int, List[Dict]] = {}
            for node_idx, paths in fwd_frontier.items():
                for neighbor, rel in adj.get(node_idx, []):
                    if neighbor in fwd_visited and fwd_visited[neighbor] < hop + 1:
                        continue  # Already visited at shorter depth
                    fwd_visited.setdefault(neighbor, hop + 1)
                    for path in paths:
                        if neighbor in set(path["nodes"]):
                            continue  # Cycle avoidance
                        new_path = {
                            "nodes": path["nodes"] + [neighbor],
                            "relations": path["relations"] + [rel],
                            "depth": path["depth"] + 1,
                        }
                        # Check meeting using persistent path map
                        if neighbor in bwd_visited:
                            for bwd_path in bwd_path_map.get(neighbor, []):
                                merged = _merge_paths(new_path, bwd_path, entity_list)
                                if merged:
                                    meeting_paths.append(merged)
                        new_frontier.setdefault(neighbor, []).append(new_path)
            # Beam prune per node
            for node in new_frontier:
                if len(new_frontier[node]) > beam_width:
                    new_frontier[node].sort(
                        key=lambda p: _prior_score(p, prior_rel_indices), reverse=True
                    )
                    new_frontier[node] = new_frontier[node][:beam_width]
            fwd_frontier = new_frontier
            # Accumulate into persistent path map
            for node, node_paths in new_frontier.items():
                fwd_path_map.setdefault(node, []).extend(node_paths)
                if len(fwd_path_map[node]) > beam_width:
                    fwd_path_map[node].sort(
                        key=lambda p: _prior_score(p, prior_rel_indices), reverse=True
                    )
                    fwd_path_map[node] = fwd_path_map[node][:beam_width]
        else:
            # Expand backward
            new_frontier: Dict[int, List[Dict]] = {}
            for node_idx, paths in bwd_frontier.items():
                for neighbor, rel in adj.get(node_idx, []):
                    if neighbor in bwd_visited and bwd_visited[neighbor] < hop + 1:
                        continue
                    bwd_visited.setdefault(neighbor, hop + 1)
                    for path in paths:
                        if neighbor in set(path["nodes"]):
                            continue
                        new_path = {
                            "nodes": path["nodes"] + [neighbor],
                            "relations": path["relations"] + [rel],
                            "depth": path["depth"] + 1,
                        }
                        # Check meeting using persistent path map
                        if neighbor in fwd_visited:
                            for fwd_path in fwd_path_map.get(neighbor, []):
                                merged = _merge_paths(fwd_path, new_path, entity_list)
                                if merged:
                                    meeting_paths.append(merged)
                        new_frontier.setdefault(neighbor, []).append(new_path)
            for node in new_frontier:
                if len(new_frontier[node]) > beam_width:
                    new_frontier[node].sort(
                        key=lambda p: _prior_score(p, prior_rel_indices), reverse=True
                    )
                    new_frontier[node] = new_frontier[node][:beam_width]
            bwd_frontier = new_frontier
            # Accumulate into persistent path map
            for node, node_paths in new_frontier.items():
                bwd_path_map.setdefault(node, []).extend(node_paths)
                if len(bwd_path_map[node]) > beam_width:
                    bwd_path_map[node].sort(
                        key=lambda p: _prior_score(p, prior_rel_indices), reverse=True
                    )
                    bwd_path_map[node] = bwd_path_map[node][:beam_width]

        # Don't early terminate — shorter meeting paths may not contain
        # the answer; allow BFS to explore deeper hops for better paths

    if not meeting_paths:
        return [], 0, 0

    # Score by prior coverage: paths with more prior relations win
    for p in meeting_paths:
        p["prior_coverage"] = _prior_score(p, prior_rel_indices)

    max_cov = max(p["prior_coverage"] for p in meeting_paths)
    best = [p for p in meeting_paths if p["prior_coverage"] == max_cov]
    max_depth = max(p["depth"] for p in best)
    return best, max_depth, max_cov


def _prior_score(path: Dict, prior_rel_indices: set) -> int:
    """Count how many relations in the path match prior relations."""
    return sum(1 for r in path.get("relations", []) if r in prior_rel_indices)


def _merge_paths(
    fwd_path: Dict, bwd_path: Dict, entity_list: List[str]
) -> Optional[Dict]:
    """Merge forward and backward paths at meeting point.
    Forward path: [anchor, ..., meeting_node]
    Backward path: [target, ..., meeting_node] (built by appending during BFS)
    """
    fwd_nodes = fwd_path["nodes"]
    bwd_nodes = bwd_path["nodes"]

    # Meeting point is at the END of both paths
    meeting_node = fwd_nodes[-1]
    if meeting_node != bwd_nodes[-1]:
        return None

    fwd_set = set(fwd_nodes[:-1])
    bwd_set = set(bwd_nodes[:-1])
    if fwd_set & bwd_set:
        return None  # Overlapping nodes besides meeting point

    # Reverse backward path to get: meeting_node -> ... -> target
    # Then merge: [anchor, ..., meeting_node] + [..., target]
    bwd_reversed = list(reversed(bwd_nodes[:-1]))
    bwd_rels_reversed = list(reversed(bwd_path["relations"]))

    merged_nodes = fwd_nodes + bwd_reversed
    merged_rels = fwd_path["relations"] + bwd_rels_reversed
    merged_depth = fwd_path["depth"] + bwd_path["depth"]

    return {
        "nodes": merged_nodes,
        "relations": merged_rels,
        "depth": merged_depth,
        "covered_intents": set(),
    }


def chain_expand(
    anchor_idx: int,
    resolved_intents: List[Dict[str, Any]],
    h_ids: List[int],
    r_ids: List[int],
    t_ids: List[int],
    entity_list: List[str],
) -> tuple[List[Dict[str, Any]], int]:
    """
    Ordered intent chain expansion (fallback).
    Expands step 1 → step 2 → ... in order. If step N fails, skips and tries step N+1.
    """
    paths = [{"nodes": [anchor_idx], "relations": [], "depth": 0}]

    for intent in resolved_intents:
        rel_indices = set(intent["resolved_relation_indices"])
        new_paths = []

        for path in paths:
            current_node = path["nodes"][-1]
            children = expand_node_with_relations(current_node, rel_indices, h_ids, r_ids, t_ids, reverse=False)
            reverse_children = expand_node_with_relations(current_node, rel_indices, h_ids, r_ids, t_ids, reverse=True)
            all_children = children + reverse_children

            if all_children:
                path_node_set = set(path["nodes"])
                for child_idx, rel_idx in all_children:
                    if child_idx in path_node_set:
                        continue
                    new_path = {
                        "nodes": path["nodes"] + [child_idx],
                        "relations": path["relations"] + [rel_idx],
                        "depth": path["depth"] + 1
                    }
                    child_name = entity_list[child_idx] if 0 <= child_idx < len(entity_list) else ""
                    if is_cvt_like(child_name):
                        cvt_children = expand_through_cvt(child_idx, h_ids, r_ids, t_ids, entity_list)
                        new_node_set = set(new_path["nodes"])
                        for cvt_idx, cvt_rel in cvt_children:
                            if cvt_idx in new_node_set:
                                continue
                            new_paths.append({
                                "nodes": new_path["nodes"] + [cvt_idx],
                                "relations": new_path["relations"] + [cvt_rel],
                                "depth": new_path["depth"] + 1
                            })
                    else:
                        new_paths.append(new_path)
            else:
                # Dead end — keep path, next intent will try from same node
                new_paths.append(path)

        paths = new_paths if new_paths else paths

    if not paths:
        return [], 0

    max_depth = max(p["depth"] for p in paths)
    deepest = [p for p in paths if p["depth"] == max_depth]
    return deepest, max_depth


def resolve_target_entity(target_name: Optional[str], entity_list: List[str]) -> Optional[int]:
    """Find target entity index in entity list."""
    if not target_name:
        return None

    target_norm = normalize(target_name)
    for idx, entity in enumerate(entity_list):
        if normalize(entity) == target_norm:
            return idx
        # Handle partial match for variations
        if target_norm in normalize(entity) or normalize(entity) in target_norm:
            return idx
    return None


async def run_case(
    session: aiohttp.ClientSession,
    sample: Dict[str, Any],
    pilot_row: Dict[str, Any],
) -> Dict[str, Any]:
    question = pilot_row["question"]

    # Layer 1: LLM Decomposition
    decomposition = await call_llm(session, question)

    # Prepare entity list - combine text and non-text entities
    # The indices in h_id_list, r_id_list, t_id_list reference this combined list
    text_entities = sample.get("text_entity_list", [])
    non_text_entities = sample.get("non_text_entity_list", [])
    all_entities = text_entities + non_text_entities

    # Build entity name to index mapping
    entity_name_to_idx = {}
    for idx, e in enumerate(all_entities):
        if e and len(e) > 1:
            entity_name_to_idx[e] = idx

    # Prepare relation list
    all_relations = list(sample.get("relation_list", []))
    relation_texts = [f"{r} ; {rel_to_text(r)}" for r in all_relations]

    # Resolve anchor entity: GTE against entity list, fallback to string match
    anchor_name = decomposition.get("anchor")
    anchor_idx = None
    anchor_source = None
    if anchor_name:
        # GTE: query=anchor_name, candidates=entity list
        anchor_idx = await gte_resolve_anchor(session, anchor_name, all_entities, top_k=5)
        anchor_source = "gte" if anchor_idx is not None else None
        # Fallback: exact string match
        if anchor_idx is None:
            anchor_idx = entity_name_to_idx.get(anchor_name)
            anchor_source = "exact" if anchor_idx is not None else None
        # Fallback: fuzzy string match
        if anchor_idx is None:
            anchor_norm = normalize(anchor_name)
            for entity_name, idx in entity_name_to_idx.items():
                if anchor_norm == normalize(entity_name) or anchor_norm in normalize(entity_name) or normalize(entity_name) in anchor_norm:
                    anchor_idx = idx
                    anchor_source = "fuzzy"
                    break

    # Layer 2: GTE Relation Resolution
    resolved_intents = []
    for intent in decomposition.get("intent_chain", []):
        resolved_candidates = []
        seen = set()
        for candidate in intent.get("relation_candidates", []):
            rows = await gte_retrieve(
                session,
                candidate,
                all_relations,
                candidate_texts=relation_texts,
                top_k=5
            )
            if rows:
                for row in rows:
                    rel_name = row["candidate"]
                    if rel_name in all_relations:
                        rel_idx = all_relations.index(rel_name)
                        if rel_idx not in seen:
                            resolved_candidates.append(rel_idx)
                            seen.add(rel_idx)

        resolved_intents.append({
            "step": intent["step"],
            "intent": intent["intent"],
            "relation_candidates": intent.get("relation_candidates", []),
            "resolved_relation_indices": resolved_candidates[:5],
        })

    # Layer 3: Graph Traversal
    paths = []
    max_depth = 0
    max_coverage = 0
    gt_hit = False
    answer_candidates = []

    if anchor_idx is not None:
        h_ids = sample.get("h_id_list", [])
        r_ids = sample.get("r_id_list", [])
        t_ids = sample.get("t_id_list", [])

        # Collect all prior relation indices
        prior_rel_indices = set()
        for intent in resolved_intents:
            prior_rel_indices.update(intent["resolved_relation_indices"])

        # Detect must_exist entities: entities in the question that are NOT the anchor
        # and NOT a substring/variant of the anchor
        must_exist_idx = None
        anchor_name_lower = normalize(anchor_name) if anchor_name else ""
        question_lower = question.lower()
        for entity_name, idx in entity_name_to_idx.items():
            if idx == anchor_idx:
                continue
            if len(entity_name) < 5 or is_cvt_like(entity_name):
                continue
            # Skip entities that are substrings of the anchor or vice versa
            ent_lower = normalize(entity_name)
            if anchor_name_lower and (ent_lower in anchor_name_lower or anchor_name_lower in ent_lower):
                continue
            # Check if entity name appears in question as proper noun (case-sensitive)
            if entity_name in question:
                must_exist_idx = idx
                break

        # Also check target_entity from decomposition
        target_entity = decomposition.get("target_entity")
        target_idx = None
        if target_entity:
            target_idx = entity_name_to_idx.get(target_entity)
            if target_idx is None:
                target_norm = normalize(target_entity)
                for entity_name, idx in entity_name_to_idx.items():
                    if target_norm == normalize(entity_name) or target_norm in normalize(entity_name) or normalize(entity_name) in target_norm:
                        target_idx = idx
                        break

        # Strategy selection:
        # Bidirectional BFS only for must_exist entities (NOT target_entity)
        # target_entity uses chain_expand + post-filtering (already works well)
        bi_target = None
        if must_exist_idx is not None and must_exist_idx != anchor_idx:
            # must_exist must be different from target to avoid double-routing
            if must_exist_idx != target_idx:
                bi_target = must_exist_idx

        if bi_target is not None:
            bi_paths, bi_depth, bi_cov = bidirectional_expand(
                anchor_idx, bi_target, prior_rel_indices,
                h_ids, r_ids, t_ids, all_entities,
            )
            # Only use bidirectional results if they have prior coverage
            # Otherwise shortest paths without priors are unreliable —
            # fall back to prior-guided chain_expand instead
            if bi_paths and bi_cov > 0:
                paths = bi_paths
                max_depth = bi_depth
                max_coverage = bi_cov

        # Fallback to chain_expand if bidirectional failed or not applicable
        if not paths:
            paths, max_depth = chain_expand(
                anchor_idx,
                resolved_intents,
                h_ids,
                r_ids,
                t_ids,
                all_entities
            )
            max_coverage = min(max_depth, len(resolved_intents))

        # Fallback to flexible if both failed
        if max_depth == 0:
            paths_fb, depth_fb, coverage_fb = flexible_expand(
                anchor_idx,
                resolved_intents,
                h_ids,
                r_ids,
                t_ids,
                all_entities
            )
            if depth_fb > 0 or coverage_fb > 0:
                paths = paths_fb
                max_depth = depth_fb
                max_coverage = coverage_fb

        # Layer 4: Target Filtering
        if target_idx is not None:
            target_paths = [p for p in paths if target_idx in p["nodes"]]
            paths = target_paths if target_paths else paths

        # Layer 5: Answer Extraction
        # Collect ALL non-CVT, non-anchor entities from deepest paths.
        # The answer may be at any position depending on question structure.
        for path in paths:
            for node_idx in path["nodes"]:
                # Skip anchor
                if node_idx == anchor_idx:
                    continue
                node_name = all_entities[node_idx] if 0 <= node_idx < len(all_entities) else ""
                # Expand CVT nodes to get actual entities
                if is_cvt_like(node_name):
                    cvt_children = expand_through_cvt(node_idx, h_ids, r_ids, t_ids, all_entities)
                    for cvt_idx, _ in cvt_children:
                        if cvt_idx != anchor_idx and 0 <= cvt_idx < len(all_entities):
                            cvt_name = all_entities[cvt_idx]
                            if not is_cvt_like(cvt_name):
                                answer_candidates.append(cvt_name)
                else:
                    answer_candidates.append(node_name)

        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for c in answer_candidates:
            norm_c = normalize(c)
            if norm_c not in seen:
                seen.add(norm_c)
                unique_candidates.append(c)
        answer_candidates = unique_candidates

    # Layer 6: Evaluation
    gt_answers = pilot_row.get("gt", [])
    gt_hit = candidate_hit(answer_candidates, gt_answers) if answer_candidates else False

    return {
        "case_id": pilot_row["case_id"],
        "question": question,
        "gt_answers": gt_answers,
        "decomposition": decomposition,
        "anchor_resolved": anchor_idx is not None,
        "anchor_idx": anchor_idx,
        "resolved_intents": resolved_intents,
        "num_paths": len(paths),
        "max_depth": max_depth,
        "max_coverage": max_coverage,
        "target_entity": decomposition.get("target_entity"),
        "target_resolved": resolve_target_entity(decomposition.get("target_entity"), all_entities) is not None,
        "answer_candidates": answer_candidates,
        "gt_hit": gt_hit,
    }


def write_report(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = []

    # Aggregate statistics
    cases = len(rows)
    gt_hits = sum(1 for r in rows if r["gt_hit"])
    anchor_resolved = sum(1 for r in rows if r["anchor_resolved"])
    avg_max_depth = sum(r["max_depth"] for r in rows) / cases if cases else 0
    target_entity_cases = sum(1 for r in rows if r["target_entity"])
    target_hit = sum(1 for r in rows if r["target_resolved"] and r["target_entity"])
    target_hit_rate = target_hit / target_entity_cases if target_entity_cases else 0
    avg_coverage = sum(r["max_coverage"] for r in rows) / cases if cases else 0

    lines.append("# Constrained Subgraph Test")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total cases: {cases}")
    lines.append(f"- GT answer recall: {gt_hits}/{cases} ({100*gt_hits/cases if cases else 0:.1f}%)")
    lines.append(f"- Anchor resolved: {anchor_resolved}/{cases} ({100*anchor_resolved/cases if cases else 0:.1f}%)")
    lines.append(f"- Avg max depth: {avg_max_depth:.2f}")
    lines.append(f"- Avg intent coverage: {avg_coverage:.2f}")
    lines.append(f"- Target entity cases: {target_entity_cases}/{cases}")
    lines.append(f"- Target resolved: {target_hit}/{target_entity_cases} ({100*target_hit_rate if target_entity_cases else 0:.1f}%)")
    lines.append("")

    # Per-case details
    lines.append("## Per-Case Results")
    lines.append("")

    for row in rows:
        lines.append(f"### {row['case_id']}")
        lines.append(f"**Question:** {row['question']}")
        lines.append("")
        lines.append(f"- GT answers: {row['gt_answers']}")
        lines.append(f"- Anchor resolved: {row['anchor_resolved']} (idx: {row['anchor_idx']})")
        lines.append(f"- Target entity: {row['target_entity']}")
        lines.append(f"- Target resolved: {row['target_resolved']}")
        lines.append(f"- Max depth: {row['max_depth']}")
        lines.append(f"- Intent coverage: {row['max_coverage']}/{len(row['resolved_intents'])}")
        lines.append(f"- Num paths: {row['num_paths']}")
        lines.append(f"- GT hit: {row['gt_hit']}")
        lines.append("")

        lines.append("**Decomposition:**")
        lines.append(f"- Anchor: {row['decomposition'].get('anchor')}")
        lines.append(f"- Target entity: {row['decomposition'].get('target_entity')}")
        lines.append("")

        lines.append("**Intent chain:**")
        for intent in row["resolved_intents"]:
            lines.append(f"- Step {intent['step']}: {intent['intent']}")
            lines.append(f"  - Candidates: {intent['relation_candidates']}")
            lines.append(f"  - Resolved indices: {intent['resolved_relation_indices']}")
        lines.append("")

        lines.append(f"**Answer candidates:** {row['answer_candidates']}")
        lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


async def amain(args: argparse.Namespace) -> None:
    pilot_rows = json.loads(Path(args.pilot_results).read_text())
    with Path(args.cwq_pkl).open("rb") as f:
        samples = pickle.load(f)
    sample_map = {s["id"]: s for s in samples if "id" in s}

    rows = []
    async with aiohttp.ClientSession() as session:
        for pilot_row in pilot_rows[: args.limit]:
            sample = sample_map[pilot_row["case_id"]]
            rows.append(await run_case(session, sample, pilot_row))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write results.json
    (out_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write summary.json
    cases = len(rows)
    gt_hits = sum(1 for r in rows if r["gt_hit"])
    anchor_resolved = sum(1 for r in rows if r["anchor_resolved"])
    avg_max_depth = sum(r["max_depth"] for r in rows) / cases if cases else 0
    target_entity_cases = sum(1 for r in rows if r["target_entity"])
    target_hit = sum(1 for r in rows if r["target_resolved"] and r["target_entity"])

    summary = {
        "cases": cases,
        "gt_answer_recall": gt_hits,
        "anchor_resolved": anchor_resolved,
        "avg_max_depth": avg_max_depth,
        "target_entity_cases": target_entity_cases,
        "target_hit": target_hit,
        "target_hit_rate": target_hit / target_entity_cases if target_entity_cases else 0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write report.md
    write_report(out_dir, rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-results", default=str(DEFAULT_PILOT_RESULTS))
    parser.add_argument("--cwq-pkl", default=str(DEFAULT_CWQ_PKL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
