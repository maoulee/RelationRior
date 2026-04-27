#!/usr/bin/env python3
"""
Per-Agent Independent Pipeline Test

Tests the decomposed 4-agent prompt system. Each agent has its own fixed
system prompt and only its stage-relevant tools. The orchestrator handles
data bridging between agents.

Pipeline:
  Agent 1 (Discovery)  → check_entities, explore_schema → entities + relations
  Agent 2 (Planning)   → plan → action space
  Agent 3 (Action+Filter) → action, filter → execution results
  Agent 4 (Answer)     → no tools → final \boxed{} output

Usage:
    python scripts/run_stage_pipeline_test.py \
        --data-path data/cwq/cwq_test.jsonl \
        --limit-cases 20 \
        --max-concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.subagent_prompts import (
    AGENT_SYSTEM_PROMPTS,
    AGENT_USER_TEMPLATES,
    AGENT_TOOLS,
)

# Backend config
KG_API_URL = os.getenv("KGQA_KG_API_URL", "http://localhost:8002").rstrip("/")
LLM_API_URL = os.getenv("KGQA_LLM_API_URL", "http://localhost:8000/v1").rstrip("/")
LLM_MODEL = os.getenv("KGQA_MODEL_NAME", "qwen35-9b-local")
GTE_API_URL = os.getenv("KGQA_GTE_API_URL", "http://localhost:8003").rstrip("/")

TOOL_ENDPOINTS = {
    "check_entities": f"{KG_API_URL}/v2/find_entities",
    "find_entities": f"{KG_API_URL}/v2/find_entities",
    "explore_schema": f"{KG_API_URL}/v2/explore_schema",
    "semantic_retrieve": f"{KG_API_URL}/v2/semantic_retrieve",
    "plan": f"{KG_API_URL}/v2/plan_subquestion",
    "action": f"{KG_API_URL}/v2/match_pattern",
    "filter": f"{KG_API_URL}/v2/filter",
}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default


# ── Utility functions ──────────────────────────────────────────────

def _normalize_text(value: str) -> str:
    if not isinstance(value, str):
        value = str(value)
    value = value.lower().strip()
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = "".join(ch for ch in value if ch.isalnum() or ch in " .-_")
    return " ".join(value.split())


def _entity_overlap(pred: List[str], gt: List[str]) -> Tuple[float, float, float]:
    if not pred or not gt:
        return (0.0, 0.0, 0.0)
    pred_set = {_normalize_text(x) for x in pred}
    gt_set = {_normalize_text(x) for x in gt}
    pred_set.discard("")
    gt_set.discard("")
    if not pred_set or not gt_set:
        return (0.0, 0.0, 0.0)
    common = pred_set & gt_set
    precision = len(common) / len(pred_set) if pred_set else 0.0
    recall = len(common) / len(gt_set) if gt_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return (precision, recall, f1)


def _extract_clean_question(user_message: str) -> str:
    if "Question:" in user_message:
        after_question = user_message.split("Question:", 1)[1]
        question = after_question.strip().split("\n")[0].strip()
        return question
    return user_message.strip()


def _extract_domains(user_message: str) -> List[str]:
    match = re.search(r"Available Domains in Subgraph:\s*\n(.*?)(?:\n\n|\[Retrieval)", user_message, re.S)
    if match:
        domains_text = match.group(1).strip()
        return [d.strip() for d in domains_text.split(",") if d.strip()]
    return []


def _is_relation_name(name: str) -> bool:
    parts = name.split(".")
    if len(parts) >= 3:
        return True
    if re.match(r'^m\.[a-z0-9]+$', name):
        return True
    return False


def _is_temporal_value(text: str) -> bool:
    """Check if a string is a temporal/numeric value, not a KG entity."""
    text = text.strip()
    if not text:
        return True
    # Pure year or date patterns
    if re.match(r'^\d{4}$', text):
        return True
    # Date patterns like "3-1-1983", "11/6/1962", "December 12, 1808"
    if re.match(r'^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$', text):
        return True
    if re.match(r'^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\s*$', text):
        return True
    # Pure number
    if re.match(r'^\d+$', text):
        return True
    return False


def _is_temporal_entity(name: str) -> bool:
    """Filter entities that are temporal events or date-noise from check_entities.

    IMPORTANT: Do NOT filter legitimate named entities that happen to contain a year,
    such as "2010 FIFA World Cup" or "1931 World Series". Only filter generic
    year+category combos that are likely date-noise (e.g., "2014 LPGA Tour season").
    """
    name_lower = name.lower()
    # Only filter entities that are SHORT and dominated by year+generic-category
    noise_patterns = [
        # Year + generic category words anywhere in name (catches "2014 LPGA Tour season")
        r'\b\d{4}\b.*\b(?:tour|season|wave|festival|election|primary|deadline|classic|shootings|cold wave)\b',
        # Generic category + year anywhere
        r'\b(?:tour|classic|season|festival|election|primary|deadline)\b.*\b\d{4}\b',
        # Hurricane names are usually noise in KGQA context
        r'^hurricane\s',
    ]
    for pat in noise_patterns:
        if re.search(pat, name_lower):
            return True
    return False


def _filter_answer_entities(entities: List[str]) -> List[str]:
    filtered = []
    for e in entities:
        e = e.strip()
        if not e or e.lower() in {"none", "entity name", "n/a", "unknown", "no entity found", "no_candidates_found"}:
            continue
        if _is_relation_name(e):
            continue
        # Filter obvious non-entity patterns
        if e.lower().endswith((".jpg", ".png", ".gif", ".svg")):
            continue
        if len(e) <= 1:
            continue
        filtered.append(e)
    return filtered


def _extract_entity_from_sentence(text: str, candidates: List[str]) -> List[str]:
    text_lower = text.lower()
    found = []
    for cand in candidates:
        if cand.lower() in text_lower:
            found.append(cand)
    return found


# ── Parsing ────────────────────────────────────────────────────────

def _parse_model_output(text: str, candidates: Optional[List[str]] = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "checkpoint": "",
        "tool_calls": [],
        "answer": [],
        "candidates_keyword": [],
    }

    # Extract checkpoint
    cp_match = re.search(r"<checkpoint>(.*?)</checkpoint>", text, re.S)
    if cp_match:
        result["checkpoint"] = cp_match.group(1).strip()

    # Extract answer from <answer> tag
    ans_match = re.search(r"<answer>(.*?)</answer>", text, re.S)
    if ans_match:
        ans_text = ans_match.group(1).strip()

        # Check for CANDIDATES: prefix (Stage 4 output)
        if ans_text.startswith("CANDIDATES:"):
            cand_text = ans_text[len("CANDIDATES:"):].strip()
            cands = [c.strip() for c in cand_text.split(",") if c.strip()]
            result["candidates_keyword"] = _filter_answer_entities(cands)
        else:
            # Extract \boxed{} entities
            boxed = re.findall(r"\\boxed\{([^}]+)\}", ans_text)
            if boxed:
                result["answer"] = _filter_answer_entities(boxed)
            else:
                if candidates:
                    matched = _extract_entity_from_sentence(ans_text, candidates)
                    if matched:
                        result["answer"] = matched
                    else:
                        result["answer"] = _filter_answer_entities([ans_text.strip()])
                else:
                    result["answer"] = _filter_answer_entities([ans_text.strip()])

    # Extract tool calls from <act> block
    act_match = re.search(r"<act>(.*?)</act>", text, re.S)
    if act_match:
        act_text = act_match.group(1)
        for qm in re.finditer(r"<query>(.*?)</query>", act_text, re.S):
            query_text = qm.group(1).strip()
            tc = _parse_single_tool_call(query_text)
            if tc:
                result["tool_calls"].append(tc)

    # Fallback: no <answer> and no <act> — might be a sentence answer
    if not result["answer"] and not result["tool_calls"] and not result["candidates_keyword"]:
        if candidates:
            matched = _extract_entity_from_sentence(text, candidates)
            if matched:
                result["answer"] = matched

    return result


def _parse_single_tool_call(text: str) -> Optional[Dict[str, Any]]:
    match = re.match(r"(\w+)\s*\((.*)\)\s*$", text.strip(), re.S)
    if not match:
        return None
    tool_name = match.group(1)
    args_text = match.group(2)

    args: Dict[str, Any] = {}
    for kv_match in re.finditer(r'(\w+)\s*=\s*(\[.*?\]|"[^"]*"|\'[^\']*\'|\S+)', args_text, re.S):
        key = kv_match.group(1)
        val_str = kv_match.group(2).strip()
        if val_str.startswith("[") and val_str.endswith("]"):
            items = re.findall(r'"([^"]*)"|\'([^\']*)\'|(\w+)', val_str)
            args[key] = [i[0] or i[1] or i[2] for i in items if any(i)]
        elif val_str.startswith('"') and val_str.endswith('"'):
            args[key] = val_str[1:-1]
        elif val_str.startswith("'") and val_str.endswith("'"):
            args[key] = val_str[1:-1]
        else:
            args[key] = val_str

    # Auto-fix path: convert flat ["relation","X","direction","out",...] → [{"relation":"X","direction":"out"},...]
    if "path" in args and isinstance(args["path"], list):
        raw_path = args["path"]
        # Check if it's a flat list of strings (model output) vs list of dicts (correct)
        if raw_path and isinstance(raw_path[0], str):
            # Flat format: ["relation","rel1","direction","out","relation","rel2","direction","out"]
            fixed_path = []
            i = 0
            while i < len(raw_path) - 1:
                step = {}
                while i < len(raw_path) - 1:
                    key = raw_path[i]
                    val = raw_path[i + 1]
                    if key in ("relation", "direction"):
                        step[key] = val
                        i += 2
                        if key == "direction":
                            break
                    else:
                        i += 1
                        break
                if "relation" in step:
                    if "direction" not in step:
                        step["direction"] = "out"
                    fixed_path.append(step)
            args["path"] = fixed_path
        elif raw_path and isinstance(raw_path[0], dict):
            # Already dicts, just ensure direction
            for step in raw_path:
                if "direction" not in step:
                    step["direction"] = "out"
            args["path"] = raw_path

    return {"tool_name": tool_name, "arguments": args}


# ── Frontend Validation ─────────────────────────────────────────────

def _fuzzy_match_anchor(anchor: str, verified_entities: List[str]) -> Optional[str]:
    """Find best fuzzy match for anchor in verified entities. Returns exact entity name or None."""
    if not anchor:
        return None
    anchor_lower = anchor.lower()
    # Exact match first
    for ve in verified_entities:
        if ve.lower() == anchor_lower:
            return ve
    # Fuzzy: word overlap >= 60%
    anchor_words = set(anchor_lower.split())
    scored = []
    for ve in verified_entities:
        ve_words = set(ve.lower().split())
        overlap = len(anchor_words & ve_words)
        total = max(len(anchor_words), len(ve_words))
        ratio = overlap / total if total > 0 else 0
        if overlap > 0:
            scored.append((ratio, overlap, ve))
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], -x[1]))
    if scored[0][0] >= 0.5:
        return scored[0][2]
    return None


def _validate_plan_args(args: Dict[str, Any], state: "StageState") -> List[str]:
    """Lightweight plan validation (ported from FrontendValidator). Returns error list."""
    errors = []
    verified = {e.lower() for e in state.verified_entities}
    schema_rels = set(state.discovered_relations)

    # Check 1: anchor must be verified (exact match preferred, fuzzy fallback with high overlap)
    anchor = args.get("anchor", "")
    if anchor:
        # Reject temporal/numeric anchors
        if _is_temporal_value(anchor):
            errors.append(
                f'INVALID_ANCHOR: "{anchor}" is a temporal/numeric value, not a KG entity. '
                f'Use a verified entity name as anchor. Verified: {state.verified_entities[:5]}'
            )
            return errors  # No point checking further

        anchor_lower = anchor.lower()
        # Exact match first
        exact_match = anchor_lower in verified
        if not exact_match:
            # Fuzzy fallback
            best_match = _fuzzy_match_anchor(anchor, state.verified_entities)
            if best_match:
                # Fuzzy match found — anchor will be corrected by caller
                pass
            else:
                suggestions = state.verified_entities[:5]
                errors.append(
                    f'UNVERIFIED_ENTITY: Anchor "{anchor}" was NOT matched in check_entities results. '
                    f'You MUST use an entity name from the verified list. '
                    f'Closest matches: {suggestions}. '
                    f'All verified: {state.verified_entities[:8]}'
                )

    # Check 2: relations must be in discovered schema
    all_rels = args.get("related", []) + args.get("maybe_related", [])
    if not all_rels:
        errors.append(
            "EMPTY_REQUIRED: related + maybe_related cannot both be empty. "
            "Select at least one relation from the Available Relations list."
        )
    else:
        for rel in all_rels:
            if rel not in schema_rels:
                # Find similar relations
                rel_parts = rel.split(".")
                similar = []
                for sr in schema_rels:
                    if rel_parts[-1] in sr or (len(rel_parts) >= 2 and rel_parts[-2] in sr):
                        similar.append(sr)
                similar = similar[:8] if similar else list(schema_rels)[:8]
                errors.append(
                    f'INVALID_RELATION: Relation "{rel}" was NOT found in explored schema. '
                    f'Use ONLY relations from the list above. Similar: {similar}'
                )

    # Check 3: constraint_entities must be EXACTLY verified and distinct from anchor
    for ent in args.get("constraint_entities", []):
        ent_lower = ent.lower()
        if anchor and ent_lower == anchor.lower():
            errors.append(
                f'INVALID_CONSTRAINT: Entity "{ent}" is already the anchor. '
                f'Do not use the same entity as both anchor and constraint.'
            )
        elif ent_lower not in verified:
            # Fuzzy check: accept high-overlap matches (spelling variants)
            ent_words = set(ent_lower.split())
            best_ratio = 0
            for ve in state.verified_entities:
                ve_words = set(ve.lower().split())
                overlap = len(ent_words & ve_words)
                total = max(len(ent_words), len(ve_words))
                ratio = overlap / total if total > 0 else 0
                best_ratio = max(best_ratio, ratio)
            if best_ratio < 0.6:
                candidates = [e for e in state.verified_entities[:5]]
                errors.append(
                    f'UNVERIFIED_ENTITY: Constraint entity "{ent}" is not verified. '
                    f'Only use entities from check_entities results. Verified: {candidates}'
                )

    # Check 4: constraint_relations must be in schema and distinct from path relations
    path_rel_set = set(all_rels)
    for rel in args.get("constraint_relations", []):
        if rel not in schema_rels:
            errors.append(
                f'INVALID_RELATION: Constraint relation "{rel}" was NOT found in explored schema. '
                f'Use ONLY verified relations.'
            )
        elif rel in path_rel_set:
            errors.append(
                f'INVALID_CONSTRAINT: Relation "{rel}" is already in the path. '
                f'Constraints should filter, not repeat path relations.'
            )

    return errors


def _validate_action_args(args: Dict[str, Any], state: "StageState") -> List[str]:
    """Validate action call args (anchor + path structure)."""
    errors = []
    anchor = args.get("anchor", "")
    if not anchor:
        errors.append("MISSING_FIELD: action requires an anchor entity.")

    path = args.get("path", [])
    if not path:
        errors.append("MISSING_FIELD: action requires a non-empty path.")

    return errors


# ── Backend calls ──────────────────────────────────────────────────

async def _call_backend(
    session: aiohttp.ClientSession,
    tool_name: str,
    args: Dict[str, Any],
    sample_id: str,
) -> Dict[str, Any]:
    url = TOOL_ENDPOINTS.get(tool_name)
    if not url:
        return {
            "tool_name": tool_name,
            "is_success": False,
            "status": "UNKNOWN_TOOL",
            "response_text": f"Unknown tool: {tool_name}",
            "found_end_entities": [],
            "action_hints": [],
        }

    payload = {**args, "sample_id": sample_id}
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            data = await resp.json()
            return {
                "tool_name": tool_name,
                "is_success": data.get("success", False),
                "status": data.get("status", ""),
                "response_text": data.get("response_text", ""),
                "found_end_entities": data.get("found_end_entities", []),
                "action_hints": data.get("action_hints", []),
                "structured_data": data.get("structured_data"),
            }
    except Exception as exc:
        return {
            "tool_name": tool_name,
            "is_success": False,
            "status": "BACKEND_ERROR",
            "response_text": str(exc),
            "found_end_entities": [],
            "action_hints": [],
        }


async def _gte_retrieve(
    session: aiohttp.ClientSession,
    query: str,
    candidates: List[str],
    candidate_texts: Optional[List[str]] = None,
    top_k: int = 5,
) -> List[dict]:
    """Use GTE-large to retrieve top-k candidates by semantic similarity."""
    if not candidates:
        return []
    payload = {
        "query": query,
        "candidates": candidates,
        "candidate_texts": candidate_texts,
        "top_k": top_k,
    }
    try:
        async with session.post(
            f"{GTE_API_URL}/retrieve", json=payload,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            data = await resp.json()
            return data.get("results", [])
    except Exception as exc:
        print(f"  [GTE retrieve error] {exc}")
        return []


async def _semantic_retrieve(
    session: aiohttp.ClientSession,
    sample_id: str,
    queries: List[str],
    top_k: int = 10,
) -> Dict[str, Any]:
    url = TOOL_ENDPOINTS["semantic_retrieve"]
    payload = {
        "sample_id": sample_id,
        "queries": queries,
        "top_k": top_k,
        "gte_url": GTE_API_URL,
    }
    try:
        async with session.post(
            url, json=payload,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            data = await resp.json()
            return data
    except Exception as exc:
        print(f"  [semantic_retrieve error] {exc}")
        return {"success": False, "entities_per_query": [], "relations_per_query": []}


async def _run_gte_ranking(
    session: aiohttp.ClientSession,
    state: StageState,
    question: str,
    sample_id: str,
) -> None:
    """Use GTE to recall relations per sub-question and rank entities.

    Outputs:
    - state.gte_relations_by_subq: [{sub_question, tag, relations, scores}]
    - state.gte_ranked_entities: entities ranked by GTE similarity to question
    - state.gte_recommended: flat set of all GTE-recalled relations (for ★ marking)
    """
    decomp = state.decomposition

    # Collect main + attribute questions
    sub_questions = []
    main_q = decomp.get("main_question") or decomp.get("reasoning_chain", "")
    attr_qs = decomp.get("attribute_questions", []) or []

    if main_q:
        sub_questions.append({"text": main_q, "tag": "MAIN"})
    for attr in attr_qs:
        sub_questions.append({"text": attr, "tag": "ATTR"})
    if not sub_questions:
        sub_questions.append({"text": question, "tag": "MAIN"})

    # Full-subgraph semantic retrieval per sub-question
    if not sub_questions:
        return
    sem = await _semantic_retrieve(
        session,
        sample_id,
        [sq["text"] for sq in sub_questions],
        top_k=10,
    )
    entities_per_query = sem.get("entities_per_query", []) or []
    relations_per_query = sem.get("relations_per_query", []) or []

    state.gte_entities_by_subq = []
    state.gte_relations_by_subq = []
    merged_entities = []
    seen_entities = set()
    all_gte_rels = []
    seen_rels = set()

    for idx, sq_info in enumerate(sub_questions):
        ent_rows = entities_per_query[idx] if idx < len(entities_per_query) else []
        rel_rows = relations_per_query[idx] if idx < len(relations_per_query) else []

        ranked_entities = [item["candidate"] for item in ent_rows if item.get("candidate")]
        state.gte_entities_by_subq.append({
            "sub_question": sq_info["text"],
            "tag": sq_info["tag"],
            "entities": ranked_entities,
        })
        for ent in ranked_entities:
            if ent not in seen_entities:
                merged_entities.append(ent)
                seen_entities.add(ent)
            if ent not in state.verified_entities and not _is_relation_name(ent) and not _is_temporal_entity(ent):
                state.verified_entities.append(ent)

        rels = []
        for item in rel_rows:
            rel = item.get("candidate", "")
            if not rel:
                continue
            rels.append({"relation": rel, "score": item.get("score", 0.0)})
            if rel not in seen_rels:
                all_gte_rels.append(rel)
                seen_rels.add(rel)
            if rel not in state.discovered_relations:
                state.discovered_relations.append(rel)
        state.gte_relations_by_subq.append({
            "sub_question": sq_info["text"],
            "tag": sq_info["tag"],
            "relations": rels,
        })

    state.gte_ranked_entities = merged_entities
    if merged_entities:
        print(f"  [GTE entities] top-3: {merged_entities[:3]}")
    if all_gte_rels:
        state.gte_recommended = set(all_gte_rels)
        print(f"  [GTE relations] sub-queries={len(sub_questions)}, unique_rels={len(all_gte_rels)}, top-3: {all_gte_rels[:3]}")
    else:
        state.gte_recommended = set()


async def _run_direct_discovery(
    *,
    session: aiohttp.ClientSession,
    state: StageState,
    entity_names: List[str],
    domains: List[str],
    sample_id: str,
) -> bool:
    """Direct API discovery without LLM. Uses Agent 0's extracted entity names
    to call check_entities, and case domains to call explore_schema."""
    # 1. Check entities using names from Agent 0
    for name in entity_names:
        if _is_temporal_value(name):
            continue
        result = await _call_backend(session, "check_entities", {"entity_substring": name}, sample_id)
        state.all_tool_calls.append({"tool_name": "check_entities", "arguments": {"entity_substring": name}})
        state.all_backend_results.append(result)

        if result["is_success"] and result["response_text"]:
            for line in result["response_text"].split("\n"):
                line = line.strip()
                if line.startswith("- ") and not line.startswith("- relation:"):
                    found_name = line[2:].split("[")[0].strip()
                    ctx_match = re.search(r'\[Context:\s*([^\]]+)\]', line)
                    ctx = ctx_match.group(1) if ctx_match else ""
                    if found_name and len(found_name) > 1 and not _is_relation_name(found_name):
                        if not _is_temporal_entity(found_name):
                            if found_name not in state.verified_entities:
                                state.verified_entities.append(found_name)
                                if ctx:
                                    state.entity_contexts[found_name] = ctx

    # 2. Explore schema for ALL domains in the subgraph
    for domain in domains:
        result = await _call_backend(session, "explore_schema", {"pattern": domain}, sample_id)
        state.all_tool_calls.append({"tool_name": "explore_schema", "arguments": {"pattern": domain}})
        state.all_backend_results.append(result)
        state.explored_domains.add(domain)

        if result["is_success"]:
            rels = result.get("found_end_entities", [])
            if not rels and result.get("response_text"):
                for line in result["response_text"].split("\n"):
                    line = line.strip()
                    if line.startswith("- ") and "." in line:
                        rel_name = line[2:].strip()
                        if _is_relation_name(rel_name):
                            rels.append(rel_name)
                if not rels:
                    rels = re.findall(r'([a-z_]+\.[a-z_]+\.[a-z_]+)', result["response_text"])
            for r in rels:
                if r not in state.discovered_relations:
                    state.discovered_relations.append(r)

    ok = len(state.verified_entities) > 0 or len(state.discovered_relations) > 0
    print(f"  [Direct discovery] entities={len(state.verified_entities)}, relations={len(state.discovered_relations)}, ok={ok}")
    return ok


async def _call_llm(messages: List[Dict[str, str]], max_tokens: int = 0) -> str:
    if max_tokens <= 0:
        max_tokens = _env_int("KGQA_LLM_MAX_TOKENS", 2048)
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": _env_float("KGQA_LLM_TEMPERATURE", 0.3),
        "top_p": _env_float("KGQA_LLM_TOP_P", 0.8),
        "top_k": _env_int("KGQA_LLM_TOP_K", 20),
        "repetition_penalty": _env_float("KGQA_LLM_REPETITION_PENALTY", 1.0),
        "presence_penalty": _env_float("KGQA_LLM_PRESENCE_PENALTY", 0.0),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    timeout = aiohttp.ClientTimeout(total=_env_float("KGQA_LLM_TIMEOUT_SEC", 120.0))

    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        for attempt in range(3):
            try:
                async with session.post(f"{LLM_API_URL}/chat/completions", headers=headers, json=payload) as resp:
                    if resp.status in {429, 500, 502, 503, 504}:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    resp.raise_for_status()
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt >= 2:
                    raise
                await asyncio.sleep(2 ** attempt)
    raise RuntimeError("LLM call failed after retries")


# ── Stage Runner ───────────────────────────────────────────────────

class StageState:
    """Track accumulated state across agents."""

    def __init__(self):
        self.verified_entities: List[str] = []
        self.entity_contexts: Dict[str, str] = {}  # entity → context hint
        self.discovered_relations: List[str] = []
        self.explored_domains: set = set()
        self.gte_recommended: set = set()  # GTE top relations (marked with ★)
        self.gte_relations_by_subq: List[Dict[str, Any]] = []  # [{sub_question, relations, scores}]
        self.gte_ranked_entities: List[str] = []  # entities ranked by GTE similarity to question
        self.gte_entities_by_subq: List[Dict[str, Any]] = []  # [{sub_question, tag, entities}]
        self.decomposition: Dict[str, Any] = {}  # Stage 0 output
        self.core_question: str = ""  # Rewritten question from decomposition
        self.constraints: Dict[str, Any] = {}  # Extracted constraints
        self.plan_response: str = ""
        self.plan_action_hints: List[Dict] = []
        self.action_id_map: Dict[str, Dict] = {}  # A1 → action hint dict
        self.plan_anchor: str = ""
        self.plan_relations: List[str] = []
        self.plan_constraint_entities: List[str] = []
        self.plan_constraint_relations: List[str] = []  # constraint_relations from plan
        self.leaf_entities: List[str] = []
        self.candidates: List[str] = []
        self.execution_results_text: str = ""  # full backend output for Agent 4
        self.filter_results_text: str = ""     # filter output for Agent 4
        self.filter_mode: str = ""             # entity_first / relation_fallback / skipped
        self.all_tool_calls: List[Dict] = []
        self.all_backend_results: List[Dict] = []
        self.stage_logs: List[Dict[str, str]] = []  # [{stage, system_prompt, user_message, model_response}]


async def _run_discovery(
    *,
    session: aiohttp.ClientSession,
    state: StageState,
    question: str,
    domains: List[str],
    sample_id: str,
    max_rounds: int = 3,
) -> bool:
    """Run Agent 1: Discovery. Uses preprocessing output for focused exploration."""
    system_prompt = AGENT_SYSTEM_PROMPTS[1]

    # Pass NL decomposition to discovery agent
    decomp_nl = state.decomposition.get("raw_nl", "")
    user_content = AGENT_USER_TEMPLATES[1].format(
        decomposition=decomp_nl or question,
        domains=", ".join(domains[:40]),
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    for round_idx in range(max_rounds):
        raw = await _call_llm(messages)
        messages.append({"role": "assistant", "content": raw})
        parsed = _parse_model_output(raw)

        state.stage_logs.append({
            "stage": 1,
            "round": round_idx + 1,
            "system_prompt": system_prompt,
            "user_message": messages[-2]["content"] if len(messages) >= 2 else user_content,
            "model_response": raw,
        })

        if not parsed["tool_calls"]:
            if state.verified_entities or state.discovered_relations:
                return True
            messages.append({"role": "user", "content": "Please call check_entities and explore_schema to verify entities and discover relations."})
            continue

        feedback_parts = []
        for tc in parsed["tool_calls"]:
            tool_name = tc["tool_name"]
            if tool_name not in AGENT_TOOLS[1]:
                feedback_parts.append(f"[ERROR] Tool '{tool_name}' is not available in Discovery stage.")
                continue

            args = tc["arguments"]

            # ── Domain format validation for explore_schema ──
            if tool_name == "explore_schema":
                pattern = args.get("pattern", "")
                if "." in pattern:
                    # Model used a relation path (e.g. "people.people.brother") instead of domain name
                    top_domain = pattern.split(".")[0]
                    feedback_parts.append(
                        f"[FORMAT ERROR] explore_schema expects a TOP-LEVEL domain name (no dots), "
                        f'not a relation path. Got "{pattern}". Did you mean "{top_domain}"? '
                        f'Use explore_schema(pattern="{top_domain}") instead.'
                    )
                    # Auto-fix: also call with the top-level domain
                    args = {**args, "pattern": top_domain}
                    tc = {**tc, "arguments": args}

            # ── Skip check_entities for temporal values ──
            if tool_name in ("check_entities", "find_entities"):
                entity_sub = args.get("entity_substring", "")
                if _is_temporal_value(entity_sub):
                    feedback_parts.append(
                        f"[SKIP] '{entity_sub}' is a temporal/numeric value, not a KG entity. "
                        f"Skipping check_entities for this value."
                    )
                    continue

            result = await _call_backend(session, tool_name, args, sample_id)
            state.all_tool_calls.append(tc)
            state.all_backend_results.append(result)

            if tool_name in ("check_entities", "find_entities"):
                if result["is_success"] and result["response_text"]:
                    for line in result["response_text"].split("\n"):
                        line = line.strip()
                        if line.startswith("- ") and not line.startswith("- relation:"):
                            name = line[2:].split("[")[0].strip()
                            ctx_match = re.search(r'\[Context:\s*([^\]]+)\]', line)
                            ctx = ctx_match.group(1) if ctx_match else ""
                            if name and len(name) > 1 and not _is_relation_name(name):
                                # Filter: skip entities that are clearly temporal/event noise
                                if not _is_temporal_entity(name):
                                    state.verified_entities.append(name)
                                    if ctx:
                                        state.entity_contexts[name] = ctx
                    feedback_parts.append(f"[{tool_name}] {result['response_text'][:800]}")

            elif tool_name == "explore_schema":
                pattern = args.get("pattern", "")
                state.explored_domains.add(pattern)
                if result["is_success"]:
                    rels = result.get("found_end_entities", [])
                    # Fallback: parse relations from response_text if found_end_entities empty
                    if not rels and result.get("response_text"):
                        # Try line-by-line first
                        for line in result["response_text"].split("\n"):
                            line = line.strip()
                            if line.startswith("- ") and "." in line:
                                rel_name = line[2:].strip()
                                if _is_relation_name(rel_name):
                                    rels.append(rel_name)
                        # Regex fallback: find all domain.type.relation patterns
                        if not rels:
                            rels = re.findall(r'([a-z_]+\.[a-z_]+\.[a-z_]+)', result["response_text"])
                    for r in rels:
                        if r not in state.discovered_relations:
                            state.discovered_relations.append(r)
                    print(f"  [explore_schema:{pattern}] backend_success={result['is_success']}, found_end_entities={len(result.get('found_end_entities', []))}, parsed_rels={len(rels)}, total_discovered={len(state.discovered_relations)}")
                    feedback_parts.append(f"[explore_schema:{pattern}] Found {len(rels)} relations")
                else:
                    print(f"  [explore_schema:{pattern}] FAILED: status={result.get('status')}, text={result.get('response_text', '')[:200]}")
                    feedback_parts.append(f"[explore_schema:{pattern}] No relations found. Try a different domain.")

        if state.verified_entities or state.discovered_relations:
            return True

        feedback = "\n".join(feedback_parts)
        messages.append({"role": "user", "content": f"[Results]\n{feedback}\n\nContinue exploring. Try different domains from the Available Domains list."})

    return len(state.verified_entities) > 0 or len(state.discovered_relations) > 0


def _format_entities(entities: List[str]) -> str:
    """Format entities one per line."""
    return "\n".join(f"- {e}" for e in entities[:20])


def _format_relations_by_domain(relations: List[str], limit: int = 80, recommended: Optional[set] = None) -> str:
    """Format relations grouped by domain, one per line. Mark GTE-recommended with ★."""
    from collections import OrderedDict
    groups: Dict[str, List[str]] = OrderedDict()
    for r in relations[:limit]:
        parts = r.split(".")
        domain = parts[0] if len(parts) >= 1 else "other"
        groups.setdefault(domain, []).append(r)
    lines = []
    for domain, rels in groups.items():
        lines.append(f"[{domain}]")
        for r in rels:
            marker = " ★" if recommended and r in recommended else ""
            lines.append(f"  - {r}{marker}")
    if recommended:
        lines.insert(0, "(★ = GTE-recommended by sub-question similarity)")
    return "\n".join(lines)


def _parse_nl_decomposition(text: str) -> Dict[str, Any]:
    """Parse NL preprocessing output into structured fields.
    Supports [MAIN]|[HOP]|[FILTER] (new), [MAIN]|[FOLLOW-UP] (legacy), and old 答案类型/锚点/推理链路 formats."""
    result = {
        "answer_type": "",
        "anchor": "",
        "reasoning_chain": "",
        "main_question": "",
        "attribute_questions": [],
        "constraint_entities": [],
        "constraint_meaning": "",
        "search_domains": [],
        "has_constraints": False,
        "constraint_relation_hint": "",
        "hops": [],          # legacy compatibility
        "filters": [],       # legacy compatibility
        "is_multi_hop": False,
        "entity_names": [],  # Raw entity names from [ENTITY CHECK]
        "raw_nl": text,
    }

    if "[MAIN]" in text:
        # ── New format: [MAIN] q | [ATTR] q (or legacy [HOP]/[FILTER]/[FOLLOW-UP]) ──

        # Extract named entities from [ENTITY CHECK] section
        entities = []
        for line in text.split("\n"):
            if "[ENTITY CHECK]" in line:
                for m in re.finditer(r'"([^"]+)"', line):
                    entities.append(m.group(1))
        result["entity_names"] = entities

        # Get text after </reasoning> (where [MAIN]/[HOP]/[FILTER] live)
        post_reasoning = text
        reasoning_end = text.find("</reasoning>")
        if reasoning_end >= 0:
            post_reasoning = text[reasoning_end + len("</reasoning>"):]

        # Parse [MAIN], [ATTR], [HOP], [FILTER], [FOLLOW-UP] segments
        segments = re.split(r'\s*\|\s*', post_reasoning)
        main_q = ""
        attrs = []
        hops = []
        filters = []
        follow_ups_legacy = []
        for seg in segments:
            seg = seg.strip()
            m_main = re.match(r'\[MAIN\]\s*(.*)', seg, re.S)
            m_attr = re.match(r'\[ATTR\]\s*(.*)', seg, re.S)
            m_hop = re.match(r'\[HOP\]\s*(.*)', seg, re.S)
            m_filt = re.match(r'\[FILTER\]\s*(.*)', seg, re.S)
            m_fu = re.match(r'\[FOLLOW-UP\]\s*(.*)', seg, re.S)
            if m_main:
                main_q = m_main.group(1).strip()
            elif m_attr:
                attrs.append(m_attr.group(1).strip())
            elif m_hop:
                hops.append(m_hop.group(1).strip())
            elif m_filt:
                filters.append(m_filt.group(1).strip())
            elif m_fu:
                # Legacy [FOLLOW-UP] — treat as filter for backward compat
                follow_ups_legacy.append(m_fu.group(1).strip())

        # Drop redundant meta-validation ATTR questions (e.g. "is X the same as Y?")
        def _is_redundant_attr(q: str) -> bool:
            ql = q.lower().strip()
            return (
                "same as" in ql
                or ql.startswith("is the ")
                or ql.startswith("does the ")
                and "[prev]" in ql
            )

        attrs = [q for q in attrs if not _is_redundant_attr(q)]
        filters = [q for q in filters if not _is_redundant_attr(q)]
        follow_ups_legacy = [q for q in follow_ups_legacy if not _is_redundant_attr(q)]

        attribute_questions = attrs + hops + filters + follow_ups_legacy
        if len(attribute_questions) > 1:
            attribute_questions = attribute_questions[:1]

        # Supplement named entities from quoted strings appearing in ATTR questions.
        for q in attribute_questions:
            for m in re.finditer(r'"([^"]+)"', q):
                entities.append(m.group(1))

        # Deduplicate while preserving order.
        dedup_entities = []
        seen_entities = set()
        for ent in entities:
            ent_norm = ent.strip()
            if ent_norm and ent_norm not in seen_entities:
                dedup_entities.append(ent_norm)
                seen_entities.add(ent_norm)
        entities = dedup_entities
        result["entity_names"] = entities

        result["reasoning_chain"] = main_q
        result["main_question"] = main_q
        result["attribute_questions"] = attribute_questions
        result["hops"] = hops
        result["filters"] = filters + attrs
        result["is_multi_hop"] = len(hops) > 0

        # Anchor: first entity from [ENTITY CHECK] found in MAIN
        if main_q and entities:
            for e in entities:
                if e.lower() in main_q.lower():
                    result["anchor"] = e
                    break

        # Combine all attribute questions for constraint info
        all_fu = attribute_questions
        if all_fu:
            result["has_constraints"] = True
            fu_text = "; ".join(all_fu)
            result["constraint_meaning"] = fu_text
            result["constraint_relation_hint"] = fu_text

            # Constraint entities: entities in non-MAIN sub-questions (deduplicated)
            seen = set()
            for e in entities:
                in_fu = any(e.lower() in fu.lower() for fu in all_fu)
                in_main = e.lower() in main_q.lower() if main_q else False
                if in_fu and not in_main and e.lower() not in seen:
                    result["constraint_entities"].append(e)
                    seen.add(e.lower())

        # Answer type heuristic: use the main direct-answer question
        type_source_q = main_q
        if type_source_q:
            type_lower = type_source_q.lower()
            answer_keywords = [
                ("who", "Person"), ("where", "Location"),
                ("what country", "Country"), ("what city", "City"),
                ("what language", "Language"), ("what religion", "Religion"),
                ("what currency", "Currency"), ("what team", "Organization"),
                ("what sport", "Sport"), ("what year", "Date"),
                ("when", "Date"), ("what time", "Timezone"),
                ("what county", "Location"), ("what state", "Location"),
                ("what capital", "City"), ("what continent", "Location"),
                ("what region", "Location"), ("what type", "Type"),
                ("what cause", "Cause"), ("what disease", "Disease"),
                ("what school", "Organization"), ("what university", "Organization"),
                ("what film", "Film"), ("what movie", "Film"),
            ]
            for kw, atype in answer_keywords:
                if kw in type_lower:
                    result["answer_type"] = atype
                    break

    else:
        # ── Old format: **答案类型**/锚点/推理链路 ──
        for line in text.split("\n"):
            line_stripped = line.strip()
            if "**答案类型**" in line_stripped:
                after = line_stripped.split(":", 1)
                if len(after) > 1:
                    result["answer_type"] = after[1].strip()
            elif "**锚点**" in line_stripped:
                after = line_stripped.split(":", 1)
                if len(after) > 1:
                    result["anchor"] = after[1].strip()
            elif "**推理链路**" in line_stripped:
                after = line_stripped.split(":", 1)
                if len(after) > 1:
                    result["reasoning_chain"] = after[1].strip()
            elif "**约束含义**" in line_stripped:
                after = line_stripped.split(":", 1)
                if len(after) > 1:
                    result["constraint_meaning"] = after[1].strip()

        # Detect [→CONSTRAINT] in reasoning chain
        chain = result["reasoning_chain"]
        if "[→CONSTRAINT]" in chain or "[→CONSTRAINT]" in text:
            result["has_constraints"] = True
            constraint_match = re.search(r'\[→ANSWER\].*?→\s*(\S+.*?)\s*\[→CONSTRAINT\]', chain)
            if constraint_match:
                result["constraint_relation_hint"] = constraint_match.group(1).strip()
            else:
                constraint_match2 = re.search(r'\[→CONSTRAINT\]\s*→?\s*(.*)', chain)
                if constraint_match2:
                    result["constraint_relation_hint"] = constraint_match2.group(1).strip()

        # Parse constraint entities (newline-separated list)
        in_constraints = False
        for line in text.split("\n"):
            stripped = line.strip()
            if "**约束实体**" in stripped:
                in_constraints = True
                continue
            if in_constraints:
                if stripped.startswith("- "):
                    ent = stripped[2:].strip()
                    if ent.startswith("**"):
                        in_constraints = False
                        continue
                    if ent and ent != "(none)":
                        result["constraint_entities"].append(ent)
                elif stripped.startswith("**") or (stripped and not stripped.startswith("-") and not stripped.startswith(" ")):
                    in_constraints = False

    # Extract domains from text keywords (works for both formats)
    chain_lower = text.lower()
    domain_keywords = {
        "government": ["governor", "government", "office", "jurisdiction", "politician"],
        "sports": ["sport", "team", "championship", "league", "football", "soccer"],
        "location": ["country", "city", "capital", "border", "location", "neighbouring"],
        "people": ["person", "people", "father", "brother", "gender", "profession"],
        "religion": ["religion", "sacred", "practiced"],
        "aviation": ["airport", "flight", "aviation"],
        "geography": ["river", "geography", "bisect", "body_of_water"],
        "film": ["film", "movie", "actor", "director"],
        "music": ["music", "album", "song", "band"],
        "education": ["college", "university", "school", "education", "newspaper"],
        "award": ["award", "championship", "prize", "tournament"],
        "organization": ["organization", "company", "museum"],
    }
    for domain, keywords in domain_keywords.items():
        if any(kw in chain_lower for kw in keywords):
            result["search_domains"].append(domain)

    return result


async def _run_preprocessing(
    *,
    state: StageState,
    question: str,
) -> Dict[str, Any]:
    """Run Agent 0: NL Question Preprocessing. Runs BEFORE discovery."""
    system_prompt = AGENT_SYSTEM_PROMPTS[0]
    user_content = AGENT_USER_TEMPLATES[0].format(
        question=question,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    raw = await _call_llm(messages, max_tokens=768)

    state.stage_logs.append({
        "stage": 0,
        "round": 1,
        "system_prompt": system_prompt,
        "user_message": user_content,
        "model_response": raw,
    })

    decomp = _parse_nl_decomposition(raw)
    state.decomposition = decomp
    return decomp


async def _run_planning(
    *,
    session: aiohttp.ClientSession,
    state: StageState,
    question: str,
    sample_id: str,
    max_retries: int = 2,
) -> bool:
    """Run Agent 2.5: Compact Planning with validation + retry."""
    decomp = state.decomposition
    main_question = decomp.get("main_question") or question
    relations = state.discovered_relations

    # Prioritize relations by search_domains from preprocessing
    search_domains = decomp.get("search_domains", [])
    if search_domains:
        domain_rels = [r for r in relations if r.split(".")[0] in search_domains]
        other_rels = [r for r in relations if r.split(".")[0] not in search_domains]
        relations = domain_rels + other_rels[:20]

    system_prompt = AGENT_SYSTEM_PROMPTS[2.5]
    decomp_nl = decomp.get("raw_nl", "")

    # Clear accumulators for multi-plan candidates
    state.plan_response = ""
    state.plan_action_hints = []

    user_content = AGENT_USER_TEMPLATES[2.5].format(
        question=question,
        decomposition=decomp_nl or question,
        analysis="",  # Agent 1 doesn't produce triple analysis
        entities=_format_entities(state.verified_entities),
        relations=_format_relations_by_domain(relations, recommended=state.gte_recommended),
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    for attempt in range(1 + max_retries):
        raw = await _call_llm(messages, max_tokens=1536)
        parsed = _parse_model_output(raw)

        state.stage_logs.append({
            "stage": 2,
            "round": attempt + 1,
            "system_prompt": system_prompt,
            "user_message": user_content if attempt == 0 else messages[-1]["content"],
            "model_response": raw,
        })

        if not parsed["tool_calls"]:
            if attempt < max_retries:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "You must call plan() with anchor, related, and constraint arguments. Try again."})
                continue
            return False

        # Find the plan() tool call
        plan_tc = None
        for tc in parsed["tool_calls"]:
            if tc["tool_name"] in AGENT_TOOLS.get(2.5, {"plan"}):
                plan_tc = tc
                break

        if not plan_tc:
            if attempt < max_retries:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "You must call plan() once with anchor, related (up to 3 relations), and constraint arguments. Try again."})
                continue
            return False

        args = plan_tc["arguments"]
        args["question"] = main_question

        # Frontend Validation
        validation_errors = _validate_plan_args(args, state)
        has_unverified_anchor = any("UNVERIFIED_ENTITY" in e or "INVALID_ANCHOR" in e for e in validation_errors)
        if validation_errors:
            print(f"  [Plan Validation] attempt {attempt+1}: {len(validation_errors)} warnings")

        # ── Anchor correction: replace fuzzy-matched anchor with exact verified name ──
        anchor = args.get("anchor", "")
        corrected_anchor = _fuzzy_match_anchor(anchor, state.verified_entities)
        if corrected_anchor and corrected_anchor != anchor:
            print(f"  [Anchor Correction] \"{anchor}\" → \"{corrected_anchor}\"")
            args = {**args, "anchor": corrected_anchor}

        # If anchor is temporal/numeric and couldn't be corrected, force retry
        if _is_temporal_value(anchor) and not corrected_anchor:
            if attempt < max_retries:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": f"Anchor \"{anchor}\" is a temporal/numeric value, not a KG entity. "
                    f"Use one of the verified entities as anchor: {state.verified_entities[:5]}. Try again."})
                continue
            return False

        # If anchor is completely unverified and no fuzzy match, retry with feedback
        if has_unverified_anchor and not corrected_anchor:
            if attempt < max_retries:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": f"Anchor \"{anchor}\" is not verified. "
                    f"Use EXACTLY one of: {state.verified_entities[:8]}. Try again."})
                continue

        # Filter constraint_entities: only keep verified ones
        verified_set = {e.lower() for e in state.verified_entities}
        raw_constraint_ents = args.get("constraint_entities", [])
        filtered_constraints = [
            c for c in raw_constraint_ents
            if any(c.lower() in ve or ve.lower() in c.lower() for ve in verified_set)
        ]
        if filtered_constraints != raw_constraint_ents:
            args = {**args, "constraint_entities": filtered_constraints}

        state.plan_anchor = args.get("anchor", "")
        state.plan_relations = args.get("related", []) + args.get("maybe_related", [])
        state.plan_constraint_entities = args.get("constraint_entities", [])
        state.plan_constraint_relations = args.get("constraint_relations", [])

        # Increase max_hops to 4 for deeper multi-hop paths
        args["max_hops"] = 4

        result = await _call_backend(session, plan_tc["tool_name"], args, sample_id)
        state.all_tool_calls.append(plan_tc)
        state.all_backend_results.append(result)

        if result["is_success"]:
            state.plan_response = result["response_text"]
            state.plan_action_hints = result.get("action_hints", [])
            return True
        else:
            # Backend failed — retry if possible
            if attempt < max_retries:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": f"[Backend Error] {result.get('response_text', 'Unknown error')}\n\nTry different parameters."})
                continue

    return False


# ── GTE-based simplified planning ──────────────────────────────────

_GTE_PLAN_SYSTEM = r"""You are a KGQA planning agent.

You receive:
- one [MAIN] question that directly asks for the final answer
- several [ATTR] questions that describe properties used to judge candidates
- GTE-ranked entities and relations for each question

Your output is ONE plan().

## Planning Contract

1. MAIN QUESTION decides:
   - anchor (chosen from explicit named entities in the ORIGINAL QUESTION)
   - related
   - maybe_related

2. ATTR questions decide:
   - bridge/support entities
   - constraint_entities
   - constraint_relations

3. Entity-first rule:
   - If an ATTR question has a concrete entity and that entity is not the chosen anchor, it may become constraint_entities.
   - If the ATTR question instead describes a bridge from the anchor to the answer, use it to justify anchor choice and relation choice.
   - Use constraint_relations when the attribute is relational and no stable entity is available.

4. Relation ranking:
   - related should contain the best answer-bearing relations in ranked order.
   - maybe_related can keep weaker but still relevant main-path relations.
   - The backend will cap excessive relations later.

## Rules

- anchor must be a verified entity that appears explicitly in the original question
- all relations must come from the displayed GTE-ranked relation lists
- constraint_entities must be verified and must not equal anchor
- constraint_relations should filter candidates, not duplicate the main path unless unavoidable
- Do not branch the problem here
- Do not answer here

## Output Format

<reasoning>
[MAIN]
- anchor: ...
- related: [...]
- maybe_related: [...]

[ATTR]
- constraint_entities: [...]
- constraint_relations: [...]
- entity_first: yes/no
</reasoning>
<act>
<query>plan(question="...", anchor="...", related=["..."], maybe_related=["..."], constraint_entities=["..."], constraint_relations=["..."])</query>
</act>

## Examples

Example 1 — explicit single anchor, bridge handled by backend
Question: "Which man is the leader of the country that uses Libya, Libya, Libya as its national anthem?"
[MAIN] Who is the leader of the country that uses "Libya, Libya, Libya" as its national anthem?
[ATTR] Which country uses "Libya, Libya, Libya" as its national anthem?

<reasoning>
[MAIN]
- anchor: Libya, Libya, Libya
- related: [government.government_office_or_title.office_holders, government.government_position_held.office_holder]
- maybe_related: [government.governmental_jurisdiction.governing_officials]

[ATTR]
- constraint_entities: []
- constraint_relations: [government.national_anthem_of_a_country.country]
- entity_first: no
</reasoning>
<act>
<query>plan(question="Who is the leader of the country that uses Libya, Libya, Libya as its national anthem?", anchor="Libya, Libya, Libya", related=["government.government_office_or_title.office_holders", "government.government_position_held.office_holder"], maybe_related=["government.governmental_jurisdiction.governing_officials"], constraint_relations=["government.national_anthem_of_a_country.country"])</query>
</act>

Example 2 — answer path + concrete attribute entity
Question: "Which nation has the Alta Verapaz Department and is in Central America?"
[MAIN] Which nation has the Alta Verapaz Department?
[ATTR] Is this nation located in Central America?

<reasoning>
[MAIN]
- anchor: Alta Verapaz Department
- related: [location.administrative_division.country, location.country.first_level_divisions]
- maybe_related: [base.locations.countries.states_provinces_within]

[ATTR]
- constraint_entities: [Central America]
- constraint_relations: [base.locations.countries.continent]
- entity_first: yes
</reasoning>
<act>
<query>plan(question="Which nation has the Alta Verapaz Department?", anchor="Alta Verapaz Department", related=["location.administrative_division.country", "location.country.first_level_divisions"], maybe_related=["base.locations.countries.states_provinces_within"], constraint_entities=["Central America"], constraint_relations=["base.locations.countries.continent"])</query>
</act>"""


def _format_gte_relations_by_subq(state: StageState) -> str:
    """Format GTE-recalled relations grouped by sub-question for the planning prompt."""
    lines = []
    for sq_info in state.gte_relations_by_subq:
        tag = sq_info["tag"]
        sq_text = sq_info["sub_question"]
        rels = sq_info["relations"]
        lines.append(f"[{tag}] {sq_text}")
        for r_info in rels:
            score = r_info["score"]
            rel = r_info["relation"]
            lines.append(f"  ★ {rel} (sim={score:.3f})")
        lines.append("")
    return "\n".join(lines)


def _format_gte_entities(entities: List[str], limit: int = 10) -> str:
    """Format GTE-ranked entities for the planning prompt."""
    lines = []
    for i, e in enumerate(entities[:limit], 1):
        lines.append(f"  {i}. {e}")
    return "\n".join(lines)


def _format_gte_entities_by_subq(state: StageState) -> str:
    lines = []
    for sq_info in state.gte_entities_by_subq:
        lines.append(f"[{sq_info['tag']}] {sq_info['sub_question']}")
        for i, ent in enumerate((sq_info.get("entities") or [])[:8], 1):
            lines.append(f"  {i}. {ent}")
        lines.append("")
    return "\n".join(lines).strip()


def _entities_matching_text(text: str, verified_entities: List[str]) -> List[str]:
    text_lower = str(text or "").lower()
    matches = []
    for ent in verified_entities:
        ent_lower = ent.lower()
        if ent_lower in text_lower:
            matches.append(ent)
    return matches


async def _run_planning_gte(
    *,
    session: aiohttp.ClientSession,
    state: StageState,
    question: str,
    sample_id: str,
    max_retries: int = 2,
) -> bool:
    """Simplified planning: LLM reads GTE-recalled relations (~10-15) grouped by
    sub-question, selects anchor + constraints + relations. Much simpler than
    selecting from 200+ relations."""
    decomp = state.decomposition
    decomp_nl = decomp.get("raw_nl", "")

    # Format relations grouped by question slice
    gte_rels_text = _format_gte_relations_by_subq(state)
    if not gte_rels_text:
        gte_rels_text = _format_relations_by_domain(state.discovered_relations[:30])

    # Format entities
    entities_text = _format_gte_entities_by_subq(state) if state.gte_entities_by_subq else (
        _format_gte_entities(state.gte_ranked_entities) if state.gte_ranked_entities else _format_entities(state.verified_entities)
    )

    main_question = decomp.get("main_question") or question
    attribute_questions = decomp.get("attribute_questions", []) or []
    attr_text = "\n".join(f"- {item}" for item in attribute_questions) if attribute_questions else "- none"
    explicit_entity_candidates = []
    for ent in (decomp.get("entity_names", []) or []):
        matched = _fuzzy_match_anchor(ent, state.verified_entities)
        if matched and matched not in explicit_entity_candidates:
            explicit_entity_candidates.append(matched)
    attr_entity_candidates = []
    for attr_q in attribute_questions:
        for ent in _entities_matching_text(attr_q, state.verified_entities):
            if ent not in attr_entity_candidates:
                attr_entity_candidates.append(ent)

    user_content = f"""## Question
{question}

## Decomposition
[MAIN]
{main_question}

[ATTR]
{attr_text}

## GTE-Ranked Entities
{entities_text}

## GTE-Recalled Relations
{gte_rels_text}

Build one plan: MAIN chooses anchor/related, ATTR chooses constraints."""

    messages = [
        {"role": "system", "content": _GTE_PLAN_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    state.plan_response = ""
    state.plan_action_hints = []

    for attempt in range(1 + max_retries):
        raw = await _call_llm(messages, max_tokens=1024)
        parsed = _parse_model_output(raw)

        state.stage_logs.append({
            "stage": 2.5,
            "round": attempt + 1,
            "system_prompt": _GTE_PLAN_SYSTEM,
            "user_message": user_content if attempt == 0 else messages[-1]["content"],
            "model_response": raw,
        })

        if not parsed["tool_calls"]:
            if attempt < max_retries:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "You must call plan() with anchor and related. Try again."})
                continue
            return False

        plan_tc = None
        for tc in parsed["tool_calls"]:
            if tc["tool_name"] == "plan":
                plan_tc = tc
                break

        if not plan_tc:
            if attempt < max_retries:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "You must call plan() with anchor and related. Try again."})
                continue
            return False

        args = plan_tc["arguments"]

        # Anchor correction: anchor must come from explicit question entities.
        anchor = args.get("anchor", "")
        corrected_anchor = _fuzzy_match_anchor(anchor, state.verified_entities)
        if corrected_anchor and corrected_anchor != anchor:
            print(f"  [Anchor Correction] \"{anchor}\" → \"{corrected_anchor}\"")
            args = {**args, "anchor": corrected_anchor}
        elif not corrected_anchor and state.gte_ranked_entities:
            gte_top = state.gte_ranked_entities[0]
            if _fuzzy_match_anchor(gte_top, state.verified_entities):
                corrected = _fuzzy_match_anchor(gte_top, state.verified_entities)
                print(f"  [Anchor Fallback] GTE top-1: \"{gte_top}\" → \"{corrected}\"")
                args = {**args, "anchor": corrected}
        anchor = args.get("anchor", "")
        if explicit_entity_candidates and anchor not in explicit_entity_candidates:
            args = {**args, "anchor": explicit_entity_candidates[0]}
            anchor = args["anchor"]
            print(f"  [Main-anchor enforcement] -> {anchor}")

        # Filter constraint_entities to verified ATTR entities only and distinct from anchor
        verified_set = {e.lower() for e in state.verified_entities}
        raw_constraints = args.get("constraint_entities", [])
        filtered = [
            c for c in raw_constraints
            if any(c.lower() in ve or ve.lower() in c.lower() for ve in verified_set)
        ]
        if attr_entity_candidates:
            filtered = [c for c in filtered if any(c.lower() == ent.lower() for ent in attr_entity_candidates)]
        filtered = [c for c in filtered if c.lower() != anchor.lower()]
        if filtered != raw_constraints:
            args = {**args, "constraint_entities": filtered}

        # Do NOT auto-inject attribute entities here.
        # The model must explicitly decide whether ATTR contributes
        # constraint_entities, constraint_relations, or only bridge context.

        state.plan_anchor = args.get("anchor", "")
        state.plan_relations = args.get("related", [])
        state.plan_constraint_entities = args.get("constraint_entities", [])
        state.plan_constraint_relations = args.get("constraint_relations", [])

        args["max_hops"] = 4

        result = await _call_backend(session, "plan", args, sample_id)
        state.all_tool_calls.append(plan_tc)
        state.all_backend_results.append(result)

        if result["is_success"]:
            state.plan_response = result["response_text"]
            state.plan_action_hints = result.get("action_hints", [])
            print(f"  [GTE plan] anchor={args.get('anchor')}, related={len(args.get('related', []))}, constraints={len(args.get('constraint_entities', []))}")
            return True
        else:
            if attempt < max_retries:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": f"[Backend Error] {result.get('response_text', 'Unknown error')[:200]}\nTry different parameters."})
                continue

    return False


async def _run_planning_legacy(
    *,
    session: aiohttp.ClientSession,
    state: StageState,
    question: str,
    sample_id: str,
    prioritized_relations: Optional[List[str]] = None,
) -> bool:
    """Legacy planning with optional prioritized relations from decomposition."""
    system_prompt = AGENT_SYSTEM_PROMPTS[2]
    relations = prioritized_relations if prioritized_relations is not None else state.discovered_relations
    user_content = AGENT_USER_TEMPLATES[2].format(
        question=question,
        entities=_format_entities(state.verified_entities),
        relations=_format_relations_by_domain(relations, recommended=state.gte_recommended),
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    raw = await _call_llm(messages, max_tokens=3072)
    parsed = _parse_model_output(raw)

    state.stage_logs.append({
        "stage": 2,
        "round": 1,
        "system_prompt": system_prompt,
        "user_message": user_content,
        "model_response": raw,
    })

    if not parsed["tool_calls"]:
        return False

    for tc in parsed["tool_calls"]:
        tool_name = tc["tool_name"]
        if tool_name not in AGENT_TOOLS[2]:
            continue

        args = tc["arguments"]
        state.plan_anchor = args.get("anchor", "")
        state.plan_relations = args.get("related", []) + args.get("maybe_related", [])

        # Filter constraint_entities: only keep verified ones
        verified_set = {e.lower() for e in state.verified_entities}
        raw_constraint_ents = args.get("constraint_entities", [])
        filtered_constraints = [
            c for c in raw_constraint_ents
            if any(c.lower() in ve or ve.lower() in c.lower() for ve in verified_set)
        ]
        if filtered_constraints != raw_constraint_ents:
            args = {**args, "constraint_entities": filtered_constraints}

        result = await _call_backend(session, tool_name, args, sample_id)
        state.all_tool_calls.append(tc)
        state.all_backend_results.append(result)

        if result["is_success"]:
            state.plan_response = result["response_text"]
            state.plan_action_hints = result.get("action_hints", [])
            return True

    return False


def _format_plan_output(plan_response: str, action_hints: List, state: StageState) -> str:
    """Format plan output with action IDs for selection (not copy)."""
    parts = []
    if plan_response:
        header = plan_response.split("[Action Space Generated]:", 1)[0].strip()
        parts.append(header[:1200])

    if action_hints:
        # Build action_id_map and show actions with IDs
        state.action_id_map = {}
        action_lines = []
        for i, hint in enumerate(action_hints[:6], 1):
            aid = f"A{i}"
            if isinstance(hint, dict):
                state.action_id_map[aid] = hint
                example = hint.get("example", "")
                rdf_steps = hint.get("rdf_steps", [])
                sig_rels = [s.get("relation", "") for s in hint.get("signature", []) if isinstance(s, dict)]
                action_lines.append(f"  [{aid}] {' → '.join(sig_rels)}")
                if rdf_steps:
                    for step in rdf_steps[:3]:
                        action_lines.append(f"      {step}")
                if example:
                    action_lines.append(f"      Example: {example}")
            else:
                action_lines.append(f"  [{aid}] {hint}")
                state.action_id_map[aid] = {"raw": hint}

        parts.append("\nAvailable Actions:")
        parts.extend(action_lines)
        parts.append(f"\nAction IDs: {', '.join(state.action_id_map.keys())}")

    return "\n".join(parts)


async def _run_action(
    *,
    session: aiohttp.ClientSession,
    state: StageState,
    question: str,
    sample_id: str,
    max_rounds: int = 5,
    max_actions: int = 5,
) -> bool:
    """Run Agent 3: Action expansion only (no filter). Select up to 5 actions."""
    system_prompt = AGENT_SYSTEM_PROMPTS[3]
    plan_output = _format_plan_output(state.plan_response, state.plan_action_hints, state)
    user_content = AGENT_USER_TEMPLATES[3].format(
        question=question,
        plan_output=plan_output,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    action_count = 0
    for round_idx in range(max_rounds):
        raw = await _call_llm(messages)
        messages.append({"role": "assistant", "content": raw})
        parsed = _parse_model_output(raw)

        state.stage_logs.append({
            "stage": 3,
            "round": round_idx + 1,
            "system_prompt": system_prompt,
            "user_message": messages[-2]["content"] if len(messages) >= 2 else user_content,
            "model_response": raw,
        })

        # Check for replan decision
        if "<decision>replan</decision>" in raw:
            return False

        feedback_parts = []
        for tc in parsed["tool_calls"]:
            tool_name = tc["tool_name"]
            args = tc["arguments"]

            # Materialize select_action → actual action call
            if tool_name == "select_action":
                action_id = args.get("action_id", "")
                hint = state.action_id_map.get(action_id)
                if not hint:
                    feedback_parts.append(f"[select_action] Unknown action_id: {action_id}")
                    continue
                args = {
                    "anchor": hint.get("start_entity", ""),
                    "path": hint.get("steps", []),
                }
                tool_name = "action"

            if tool_name not in AGENT_TOOLS[3]:
                continue

            if tool_name == "action":
                result = await _call_backend(session, tool_name, args, sample_id)
                state.all_tool_calls.append(tc)
                state.all_backend_results.append(result)
                action_count += 1

                if result.get("response_text"):
                    state.execution_results_text += result["response_text"] + "\n"
                if result.get("found_end_entities"):
                    for e in result["found_end_entities"]:
                        if e not in state.leaf_entities:
                            filtered_e = _filter_answer_entities([e])
                            if filtered_e:
                                state.leaf_entities.append(filtered_e[0])
                feedback_parts.append(f"[action] {result.get('response_text', '')[:800]}")

        # Stop after max actions or if model signals done
        if action_count >= max_actions or "<decision>" in raw:
            break

        if not parsed["tool_calls"] and not feedback_parts:
            messages.append({"role": "user", "content": "Select an action by ID from the Available Actions list."})
            continue

        if feedback_parts:
            feedback = "\n".join(feedback_parts)
            if action_count < max_actions and not state.leaf_entities:
                messages.append({"role": "user", "content": f"[Results]\n{feedback}\n\nResults are empty. Try a different action or output <decision>replan</decision>."})
            else:
                break

    return len(state.leaf_entities) > 0


async def _run_filter(
    *,
    session: aiohttp.ClientSession,
    state: StageState,
    question: str,
    sample_id: str,
) -> bool:
    """Run automatic entity-first constraint checking after action selection."""
    candidates = state.leaf_entities[:20]
    if not candidates:
        state.filter_mode = "skipped"
        return True

    constraint_entities = list(state.plan_constraint_entities or state.constraints.get("constraint_entities", []) or [])
    constraint_relations = list(state.plan_constraint_relations or [])

    if not constraint_entities and not constraint_relations:
        state.filter_mode = "skipped"
        return True

    state.stage_logs.append({
        "stage": 3.5,
        "round": 1,
        "system_prompt": "AUTO_FILTER",
        "user_message": f"candidates={candidates[:10]} | constraint_entities={constraint_entities} | constraint_relations={constraint_relations}",
        "model_response": "",
    })

    # 1) Entity-first filtering
    if constraint_entities:
        args = {
            "candidates": candidates,
            "constraint_entities": constraint_entities,
            "plan_relations": state.plan_relations,
        }
        print(f"  [Filter:auto entity-first] candidates={len(candidates)} entities={len(constraint_entities)}")
        tc = {"tool_name": "filter", "arguments": args}
        result = await _call_backend(session, "filter", args, sample_id)
        state.all_tool_calls.append(tc)
        state.all_backend_results.append(result)
        if result.get("response_text"):
            state.filter_results_text = result["response_text"]
        if result.get("found_end_entities"):
            filtered_ents = [_filter_answer_entities([e]) for e in result["found_end_entities"]]
            hits = [e[0] for e in filtered_ents if e]
            if hits:
                state.leaf_entities = hits
                state.filter_mode = "entity_first"
                return True

    # 2) Fallback to relation constraints
    if constraint_relations:
        args = {
            "candidates": candidates,
            "constraint_relations": constraint_relations,
            "plan_relations": state.plan_relations,
        }
        print(f"  [Filter:auto relation-fallback] candidates={len(candidates)} relations={len(constraint_relations)}")
        tc = {"tool_name": "filter", "arguments": args}
        result = await _call_backend(session, "filter", args, sample_id)
        state.all_tool_calls.append(tc)
        state.all_backend_results.append(result)
        if result.get("response_text"):
            state.filter_results_text = result["response_text"]
        if result.get("found_end_entities"):
            filtered_ents = [_filter_answer_entities([e]) for e in result["found_end_entities"]]
            hits = [e[0] for e in filtered_ents if e]
            if hits:
                state.leaf_entities = hits
        state.filter_mode = "relation_fallback"
        return True

    state.filter_mode = "skipped"
    return True


async def _run_answer(
    *,
    state: StageState,
    question: str,
) -> List[str]:
    """Run Agent 4: Collection & Final Answer."""
    system_prompt = AGENT_SYSTEM_PROMPTS[4]

    # Format execution results
    exec_text = state.execution_results_text if state.execution_results_text else "No execution results."
    filter_section = ""
    if state.filter_results_text:
        filter_section = f"Filter Results:\n{state.filter_results_text}"

    user_content = AGENT_USER_TEMPLATES[4].format(
        question=question,
        execution_results=exec_text[:3000],
        attr_values=filter_section[:1000],
        candidates="\n".join(f"- {e}" for e in state.leaf_entities[:20]),
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    raw = await _call_llm(messages)
    parsed = _parse_model_output(raw, candidates=state.leaf_entities)

    state.stage_logs.append({
        "stage": 4,
        "round": 1,
        "system_prompt": system_prompt,
        "user_message": user_content,
        "model_response": raw,
    })

    return parsed["answer"] if parsed["answer"] else []


# ── Main pipeline ──────────────────────────────────────────────────

async def _run_stage_pipeline_case(
    *,
    case: Dict[str, Any],
    max_turns: int = 10,
    semaphore: asyncio.Semaphore,
    case_index: int,
    total_cases: int,
) -> Dict[str, Any]:
    """Run a single test case through the stage-isolated pipeline."""
    async with semaphore:
        case_id = case.get("id", "unknown")
        user_message = next(
            (msg["content"] for msg in case.get("messages", []) if msg.get("role") == "user"),
            "",
        )
        gt = case.get("ground_truth", {})
        gt_answers = gt.get("global_truth_answers", []) or case.get("solution", [])

        question = _extract_clean_question(user_message)
        domains = _extract_domains(user_message)

        state = StageState()
        predicted: List[str] = []
        error_text = ""
        stages_completed = []

        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as backend_session:
            try:
                # Stage 0: NL Preprocessing (no backend needed)
                decomp = await _run_preprocessing(
                    state=state, question=question,
                )
                stages_completed.append(0)

                # Extract structured fields from NL decomposition
                core_question = question  # use original question
                state.core_question = core_question
                state.constraints = {
                    "answer_type": decomp.get("answer_type", ""),
                    "anchor": decomp.get("anchor", ""),
                    "constraint_entities": decomp.get("constraint_entities", []),
                    "constraint_meaning": decomp.get("constraint_meaning", ""),
                    "has_constraints": decomp.get("has_constraints", False),
                    "constraint_relation_hint": decomp.get("constraint_relation_hint", ""),
                }

                # Reorder domains: prioritize search_domains from preprocessing
                search_domains = decomp.get("search_domains", [])
                if search_domains:
                    prioritized = [d for d in search_domains if d in domains]
                    remaining = [d for d in domains if d not in prioritized]
                    domains = prioritized + remaining

                # Direct Discovery: API calls only (no LLM)
                entity_names = decomp.get("entity_names", [])
                disc_ok = await _run_direct_discovery(
                    session=backend_session, state=state,
                    entity_names=entity_names, domains=domains,
                    sample_id=case_id,
                )
                stages_completed.append(1)
                if not disc_ok:
                    error_text = "Discovery failed: no entities or relations found"

                # GTE Ranking: mark recommended relations + rank entities
                if disc_ok:
                    await _run_gte_ranking(
                        session=backend_session, state=state,
                        question=question,
                        sample_id=case_id,
                    )

                # Agent 2: Planning with GTE retrieval (MAIN + ATTR contract)
                if disc_ok:
                    plan_ok = await _run_planning_gte(
                        session=backend_session, state=state,
                        question=question, sample_id=case_id,
                    )
                    stages_completed.append(2)
                    if not plan_ok:
                        error_text = "Planning failed: no plan created"

                    # Agent 3: Action (expansion only, up to 2 actions)
                    if plan_ok:
                        exec_ok = await _run_action(
                            session=backend_session, state=state,
                            question=question, sample_id=case_id,
                            max_rounds=3,
                        )
                        stages_completed.append(3)

                        # Filter stage: model decides whether to filter candidates
                        if exec_ok:
                            await _run_filter(
                                session=backend_session, state=state,
                                question=question, sample_id=case_id,
                            )
                            stages_completed.append(3.5)

                        # Agent 4: Final Answer (no filter logic)
                        predicted = await _run_answer(
                            state=state, question=question,
                        )
                        stages_completed.append(4)

            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"

        # Calculate F1
        _, _, f1 = _entity_overlap(predicted, gt_answers)
        hit_at_1 = 1.0 if (predicted and _normalize_text(predicted[0]) in {_normalize_text(g) for g in gt_answers}) else 0.0

        print(f"[{case_index}/{total_cases}] {case_id}: F1={f1:.2f} | Stages={stages_completed} | Pred={predicted[:2]} | GT={gt_answers[:2]}")

        return {
            "case_id": case_id,
            "question": question,
            "f1": f1,
            "hit_at_1": hit_at_1,
            "predicted": predicted,
            "ground_truth": gt_answers,
            "stages_completed": stages_completed,
            "main_question": state.decomposition.get("main_question", ""),
            "attribute_questions": state.decomposition.get("attribute_questions", []),
            "verified_entities": state.verified_entities[:10],
            "discovered_relations": state.discovered_relations[:10],
            "plan_anchor": state.plan_anchor,
            "plan_relations": state.plan_relations,
            "plan_constraint_entities": state.plan_constraint_entities,
            "plan_constraint_relations": state.plan_constraint_relations,
            "filter_mode": state.filter_mode,
            "candidates": state.candidates[:10],
            "error": error_text,
            "tool_calls": [{"tool_name": tc["tool_name"], "arguments": tc["arguments"]} for tc in state.all_tool_calls],
            "backend_results": [
                {
                    "tool_name": r.get("tool_name", ""),
                    "is_success": r.get("is_success", False),
                    "status": r.get("status", ""),
                    "response_text": r.get("response_text", "")[:3000],
                    "found_end_entities": r.get("found_end_entities", []),
                    "action_hints": r.get("action_hints", []),
                }
                for r in state.all_backend_results
            ],
            "stage_logs": state.stage_logs,
        }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Per-stage independent pipeline test on CWQ")
    parser.add_argument("--data-path", default="data/cwq/cwq_test.jsonl")
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--case-id", action="append", default=[], help="Run specific case IDs")
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--label", default=None)
    parser.add_argument("--output-dir", default="reports/stage_pipeline_test")
    args = parser.parse_args()

    if args.label is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.label = f"{timestamp}_stage_pipeline"

    # Load cases
    data_path = Path(args.data_path)
    cases: List[Dict] = []
    wanted = set(args.case_id) if args.case_id else None
    with data_path.open() as f:
        for line in f:
            row = json.loads(line)
            if wanted and str(row.get("id")) not in wanted:
                continue
            if args.limit_cases and len(cases) >= args.limit_cases:
                break
            cases.append(row)
    print(f"Loaded {len(cases)} cases from {data_path}")

    # Run
    semaphore = asyncio.Semaphore(args.max_concurrency)
    start = asyncio.get_event_loop().time()

    tasks = [
        _run_stage_pipeline_case(
            case=case,
            semaphore=semaphore,
            case_index=i + 1,
            total_cases=len(cases),
        )
        for i, case in enumerate(cases)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    clean = []
    failed = 0
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            failed += 1
            case = cases[i]
            clean.append({
                "case_id": case.get("id", f"case_{i}"),
                "question": "",
                "f1": 0.0,
                "hit_at_1": 0.0,
                "predicted": [],
                "ground_truth": case.get("solution", []),
                "stages_completed": [],
                "error": str(r),
                "tool_calls": [],
            })
        else:
            clean.append(r)

    duration = asyncio.get_event_loop().time() - start

    # Report
    avg_f1 = mean(r["f1"] for r in clean) if clean else 0.0
    hit_at_1 = mean(r["hit_at_1"] for r in clean) if clean else 0.0
    exact = mean(1.0 if r["f1"] >= 0.95 else 0.0 for r in clean) if clean else 0.0

    # Stage completion stats
    stage_counts = {}
    for r in clean:
        for s in r.get("stages_completed", []):
            stage_counts[s] = stage_counts.get(s, 0) + 1

    output_dir = Path(args.output_dir) / args.label
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(clean, ensure_ascii=False, indent=2))

    # Generate report
    lines = [
        "# Per-Stage Independent Pipeline Report",
        "",
        f"- Label: `{args.label}`",
        f"- Data: `{args.data_path}`",
        f"- Timestamp: `{datetime.now().isoformat()}`",
        f"- Duration: `{duration:.1f}s`",
        "",
        "## Prompt Sizes",
    ]
    for agent, prompt in AGENT_SYSTEM_PROMPTS.items():
        tools = AGENT_TOOLS.get(agent, set())
        tool_str = ", ".join(sorted(tools)) if tools else "none"
        lines.append(f"- Agent {agent} ({tool_str}): `{len(prompt)}` chars")
    lines.extend([
        "",
        "## Overall Metrics",
        "",
        f"- Cases: `{len(clean)}` (failed: {failed})",
        f"- **Avg F1: `{avg_f1:.4f}`**",
        f"- **Hit@1: `{hit_at_1:.4f}`**",
        f"- **Exact Match: `{exact:.4f}`**",
        "",
        "## Agent Completion",
        "",
    ])
    for s in [0, 1, 2, 3, 4]:
        count = stage_counts.get(s, 0)
        label = "Decomposition" if s == 0 else f"Agent {s}"
        lines.append(f"- Stage {s} ({label}): `{count}/{len(clean)}` cases")

    lines.extend([
        "",
        "## Per-Case Results",
        "",
        "| Case ID | F1 | Hit@1 | Stages | Predicted | GT |",
        "|---|---:|---:|---:|---|---|",
    ])
    for item in sorted(clean, key=lambda x: x["f1"], reverse=True):
        pred = ", ".join(item["predicted"][:3]) if item["predicted"] else "-"
        gt = ", ".join(item["ground_truth"][:3]) if item["ground_truth"] else "-"
        stages = "".join(str(s) for s in item.get("stages_completed", []))
        lines.append(
            f"| {item['case_id']} | {item['f1']:.2f} | {item['hit_at_1']:.2f} | "
            f"{stages} | {pred} | {gt} |"
        )

    # Failed cases section with question text
    failed_cases = [r for r in clean if r["f1"] < 0.5]
    lines.extend([
        "",
        "## Failed Cases (F1 < 0.5)",
        "",
        "| Case ID | F1 | Stages | Question | Predicted | GT |",
        "|---|---:|---:|---|---|---|",
    ])
    for item in sorted(failed_cases, key=lambda x: x["f1"]):
        pred = ", ".join(item["predicted"][:3]) if item["predicted"] else "-"
        gt = ", ".join(item["ground_truth"][:3]) if item["ground_truth"] else "-"
        stages = "".join(str(s) for s in item.get("stages_completed", []))
        q = item["question"].replace("|", "\\|")[:80]
        lines.append(
            f"| {item['case_id']} | {item['f1']:.2f} | {stages} | {q} | {pred} | {gt} |"
        )

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines))

    print(f"\n=== Stage Pipeline Results ===")
    print(f"Avg F1: {avg_f1:.4f}")
    print(f"Hit@1: {hit_at_1:.4f}")
    print(f"Exact Match: {exact:.4f}")
    print(f"Report: {report_path}")
    print(f"JSON: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
