#!/usr/bin/env python3
"""Test NER-based entity resolution with token overlap + relation overlap scoring.

Pipeline:
1. LLM NER → extract named entities from question
2. GTE: question → top-K relations + question → top-K entity candidates
3. Token overlap filter (min_ratio=0.5) on entity candidates
4. Relation overlap scoring: entity graph relations vs question's GTE top-K relations
5. Sort by overlap desc, GTE score desc (tiebreaker)
"""
from __future__ import annotations

import asyncio
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
LLM_API_URL = "http://localhost:8000/v1/chat/completions"
LLM_MODEL = "qwen35-9b-local"
GTE_API_URL = "http://localhost:8003"
CWQ_PKL = Path("/zhaoshu/SubgraphRAG-main/retrieve/data_files/cwq/processed/test.pkl")
PILOT = ROOT / "reports" / "stage_pipeline_test" / "find_check_plan_pilot_10cases" / "results.json"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

NER_PROMPT = """Identify named entities in the question. Only literal entities explicitly written in the text.
Do not infer or guess. Numbers and percentages are NOT entities.
Output only entity names separated by comma, nothing else.
Entities:"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_cvt_like(name: str) -> bool:
    return bool(re.match(r"^[mg]\.[A-Za-z0-9_]+$", name))


def is_scalar_like(text: str) -> bool:
    t = text.strip().lower()
    if re.fullmatch(r"\d+(\.\d+)?%?", t):
        return True
    if re.fullmatch(r"\d{4}", t):
        return True
    return False


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9%.' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_overlap(question: str, entity: str) -> float:
    """Ratio of entity tokens found in question."""
    q_tokens = set(normalize(question).split())
    e_tokens = set(normalize(entity).split())
    if not e_tokens:
        return 0.0
    return len(e_tokens & q_tokens) / len(e_tokens)


def rel_to_text(rname: str) -> str:
    parts = rname.rsplit(".", 1)
    return parts[-1].replace("_", " ") if len(parts) == 2 else rname


# ---------------------------------------------------------------------------
# LLM / GTE calls
# ---------------------------------------------------------------------------

async def call_llm(session: aiohttp.ClientSession, system: str, user: str, max_tokens: int = 200) -> str:
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 0.8,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with session.post(LLM_API_URL, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
        data = await resp.json()
    return data["choices"][0]["message"]["content"]


async def gte_retrieve(
    session: aiohttp.ClientSession,
    query: str,
    candidates: List[str],
    candidate_texts: List[str] | None = None,
    top_k: int = 12,
) -> List[Dict[str, Any]]:
    payload = {"query": query, "candidates": candidates, "top_k": top_k}
    if candidate_texts:
        payload["candidate_texts"] = candidate_texts
    async with session.post(
        f"{GTE_API_URL}/retrieve", json=payload, timeout=aiohttp.ClientTimeout(total=60),
    ) as resp:
        data = await resp.json()
    return data.get("results", [])


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

async def run_case(
    session: aiohttp.ClientSession,
    sample: Dict[str, Any],
    pilot_row: Dict[str, Any],
) -> Dict[str, Any]:
    question = pilot_row["question"]

    # Build graph data
    entity_list = list(sample.get("text_entity_list", []) + sample.get("non_text_entity_list", []))
    rel_list = list(sample.get("relation_list", []))
    h_ids = list(sample.get("h_id_list", []))
    r_ids = list(sample.get("r_id_list", []))
    t_ids = list(sample.get("t_id_list", []))

    # Clean entity candidates (remove CVT, short names)
    clean_ents = []
    seen = set()
    for e in entity_list:
        if not e or is_cvt_like(e) or len(e) <= 1:
            continue
        if e not in seen:
            clean_ents.append(e)
            seen.add(e)

    # Relation display texts for GTE
    rel_texts = [rel_to_text(r) for r in rel_list]

    # ── Step 1: LLM NER ──────────────────────────────────────────────────
    ner_raw = await call_llm(session, NER_PROMPT, question, max_tokens=150)
    ner_entities = [e.strip() for e in ner_raw.split(",") if e.strip()]

    # ── Step 2: Parallel GTE queries ─────────────────────────────────────
    # 2a: question → top-K relations (for relation overlap scoring)
    # 2b: question → top-K entity candidates
    q_rel_rows, q_ent_rows = await asyncio.gather(
        gte_retrieve(session, question, rel_list, candidate_texts=rel_texts, top_k=5),
        gte_retrieve(session, question, clean_ents, top_k=12),
    )
    q_top_rel_indices = set()
    q_top_rel_names = []
    for r in q_rel_rows:
        cand = r.get("candidate", "")
        if cand in rel_list:
            idx = rel_list.index(cand)
            q_top_rel_indices.add(idx)
            q_top_rel_names.append(cand)

    # Build entity → index set mapping (for relation lookup)
    name_to_ids: Dict[str, List[int]] = {}
    for i, name in enumerate(entity_list):
        name_to_ids.setdefault(name, []).append(i)

    # GTE entity candidates with scores
    ent_candidates = []
    for r in q_ent_rows:
        cand = r.get("candidate", "")
        score = r.get("score", 0)
        if cand and cand in entity_list:
            ent_candidates.append((cand, score))

    # ── Step 3: Token overlap filter ─────────────────────────────────────
    MIN_TOKEN_RATIO = 0.5
    filtered = []
    for ent, gte_score in ent_candidates:
        to = token_overlap(question, ent)
        if to >= MIN_TOKEN_RATIO:
            filtered.append((ent, gte_score, to))

    # ── Step 4: Relation overlap scoring ──────────────────────────────────
    scored = []
    for ent, gte_score, to_ratio in filtered:
        # Get entity's graph relations
        ent_indices = set(name_to_ids.get(ent, []))
        ent_rel_indices = set()
        for h, r, t in zip(h_ids, r_ids, t_ids):
            if h in ent_indices or t in ent_indices:
                ent_rel_indices.add(r)
        overlap_count = len(ent_rel_indices & q_top_rel_indices)
        scored.append({
            "entity": ent,
            "gte_score": round(gte_score, 4),
            "token_overlap": round(to_ratio, 2),
            "relation_overlap": overlap_count,
            "entity_rel_count": len(ent_rel_indices),
        })

    # Sort: relation_overlap desc, then gte_score desc
    scored.sort(key=lambda x: (-x["relation_overlap"], -x["gte_score"]))

    # ── Step 5: Per-NER-entity resolution ─────────────────────────────────
    # For each NER entity, find best match in scored candidates
    ner_results = []
    for ner_ent in ner_entities:
        ner_norm = normalize(ner_ent)
        matches = []
        for s in scored:
            cand_norm = normalize(s["entity"])
            # Check if NER entity matches candidate (substring or containment)
            if ner_norm and cand_norm and (ner_norm in cand_norm or cand_norm in ner_norm):
                matches.append(s)
        if not matches:
            # Fallback: use GTE retrieval directly for this NER entity
            fb_rows = await gte_retrieve(session, ner_ent, clean_ents, top_k=5)
            for r in fb_rows:
                cand = r.get("candidate", "")
                score = r.get("score", 0)
                if cand and cand in entity_list:
                    to = token_overlap(question, cand)
                    if to >= MIN_TOKEN_RATIO:
                        ent_indices = set(name_to_ids.get(cand, []))
                        ent_rel_indices = set()
                        for h, r2, t in zip(h_ids, r_ids, t_ids):
                            if h in ent_indices or t in ent_indices:
                                ent_rel_indices.add(r2)
                        overlap_count = len(ent_rel_indices & q_top_rel_indices)
                        matches.append({
                            "entity": cand,
                            "gte_score": round(score, 4),
                            "token_overlap": round(to, 2),
                            "relation_overlap": overlap_count,
                            "entity_rel_count": len(ent_rel_indices),
                        })
            matches.sort(key=lambda x: (-x["relation_overlap"], -x["gte_score"]))
        ner_results.append({
            "ner_entity": ner_ent,
            "matches": matches[:5],
        })

    return {
        "case_id": pilot_row["case_id"],
        "question": question,
        "gt_answers": pilot_row.get("gt", []),
        "ner_raw": ner_raw,
        "ner_entities": ner_entities,
        "q_top_relations": q_top_rel_names,
        "all_scored": scored[:10],
        "ner_results": ner_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def amain():
    pilot_rows = json.loads(Path(PILOT).read_text())
    with Path(CWQ_PKL).open("rb") as f:
        samples = pickle.load(f)
    sample_map = {s["id"]: s for s in samples if "id" in s}

    results = []
    async with aiohttp.ClientSession() as session:
        for pilot_row in pilot_rows:
            sample = sample_map[pilot_row["case_id"]]
            result = await run_case(session, sample, pilot_row)
            results.append(result)

            # Print per-case summary
            print(f"=== {result['case_id']} ===")
            print(f"Q: {result['question']}")
            print(f"NER raw: {result['ner_raw']}")
            print(f"Q top-5 relations: {result['q_top_relations']}")
            print()

            for nr in result["ner_results"]:
                print(f"  NER: {nr['ner_entity']}")
                for m in nr["matches"][:3]:
                    marker = " ★" if m["relation_overlap"] > 0 else ""
                    print(f"    → {m['entity']}  (gte={m['gte_score']:.3f}, "
                          f"tok={m['token_overlap']:.2f}, rel_overlap={m['relation_overlap']}{marker})")
                if not nr["matches"]:
                    print("    → (no match)")
                print()
            print("---\n")

    # Save results
    out_dir = ROOT / "reports" / "stage_pipeline_test" / "ner_entity_resolution_10cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Saved to {out_dir / 'results.json'}")


if __name__ == "__main__":
    asyncio.run(amain())
