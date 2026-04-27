#!/usr/bin/env python3
"""Inject NER entity resolution into main pipeline.

Replace: decomposition (①) + anchor resolution (②)
Reuse: dual GTE (④), unified prune (⑤), relation_prior_expand (⑥),
       CVT expand (⑦), GT check (⑧), endpoint resolve (③)
"""
import asyncio
import json
import pickle
import re
import sys
from pathlib import Path
from collections import defaultdict

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_chain_decompose import (
    relation_prior_expand, is_cvt_like, expand_through_cvt,
    llm_prune_all_relations, llm_resolve_entity,
    get_entity_contexts, rel_to_text, normalize as main_normalize,
    parse_decomposition, call_llm,
)

LLM_API_URL = "http://localhost:8000/v1/chat/completions"
LLM_MODEL = "qwen35-9b-local"
GTE_API_URL = "http://localhost:8003"
CWQ_PKL = Path("/zhaoshu/SubgraphRAG-main/retrieve/data_files/cwq/processed/test.pkl")
PILOT = ROOT / "reports" / "stage_pipeline_test" / "find_check_plan_pilot_10cases" / "results.json"

# Our NER-aware decomposition prompt
DECOMP_PROMPT = """Decompose the question into sub-questions that trace a path through the knowledge graph.
Do not answer the question. Only decompose.

Rules:
- Each step must be ONE single relation lookup. Do NOT merge multiple hops into one step.
- Use as many steps as needed (1-4 steps). Do NOT force into exactly 2 steps.
- Start and End must be different entities.
- Entities must be selected from the provided retrieval results.
- Keep analysis under 4 short lines. Do not deliberate.

<analysis>
- Entities from question: ...
- Best anchor: ... (fewest branches)
- Shortest path: [if anchor already satisfies a constraint, count 0 hops for it]
- Direction: one sentence
</analysis>

Start: [entity]
1. [sub-question] (relation: [compact phrase for relation retrieval])
...
End: [entity or none]

Example (1 step):
Question: Which man is the leader of the country that uses Libya, Libya, Libya as its national anthem?
Retrieved entities: [("Libya", 1.0)]
<analysis>
- Entities: Libya
- Best anchor: Libya (it IS the country mentioned)
- Shortest path: Libya -> leader (1 hop)
- Direction: Libya -> leader directly
</analysis>
Start: Libya
1. Who is the leader of Libya? (relation: country has leader)
End: none

Example (3 steps):
Question: What country bordering France contains an airport that serves Nijmegen?
Retrieved entities: [("Nijmegen", 1.0), ("France", 1.0)]
<analysis>
- Entities: Nijmegen, France
- Best anchor: Nijmegen (specific, few airports)
- Shortest path: Nijmegen -> airport -> area -> country (3 hops)
- Direction: Nijmegen -> airport -> city -> country bordering France
</analysis>
Start: Nijmegen
1. What airport serves Nijmegen? (relation: airport serves city)
2. What city or area contains that airport? (relation: city contains airport)
3. What country bordering France is that city in? (relation: city located in country)
End: France
"""


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9%.' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


async def gte_retrieve(session, query, candidates, candidate_texts=None, top_k=5, retries=3):
    payload = {"query": query, "candidates": candidates, "top_k": top_k}
    if candidate_texts:
        payload["candidate_texts"] = candidate_texts
    for attempt in range(retries):
        try:
            async with session.post(f"{GTE_API_URL}/retrieve", json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                data = await resp.json()
            return data.get("results", [])
        except (aiohttp.client_exceptions.ServerDisconnectedError, asyncio.TimeoutError, OSError):
            if attempt < retries - 1:
                await asyncio.sleep(2)
            else:
                raise


def token_overlap(question, entity):
    q_tokens = set(normalize(question).split())
    e_tokens = set(normalize(entity).split())
    if not e_tokens:
        return 0.0
    return len(e_tokens & q_tokens) / len(e_tokens)


def parse_our_steps(raw):
    """Parse steps from our decomposition format."""
    steps = []
    for line in raw.split("\n"):
        m = re.match(r"^\d+\.\s+(.+?)\s*\(relation:\s*(.+?)\)\s*$", line.strip())
        if m:
            steps.append({
                "step": len(steps) + 1,
                "question": m.group(1).strip(),
                "relation_query": m.group(2).strip(),
                "endpoint": None,
                "endpoint_query": None,
            })
    return steps


def parse_our_start(raw):
    m = re.search(r"^Start:\s*(.+)$", raw, re.M)
    return m.group(1).strip() if m else None


def parse_our_end(raw):
    m = re.search(r"^End:\s*(.+)$", raw, re.M)
    end = m.group(1).strip() if m else "none"
    if end.lower() == "none":
        return None, None
    return end, end


async def resolve_anchor_ner(session, question, entity_list, rel_list, h_ids, r_ids, t_ids):
    """Our NER + GTE + token overlap + relation overlap pipeline."""
    # Clean entities
    clean_ents, seen = [], set()
    for e in entity_list:
        if not e or is_cvt_like(e) or len(e) <= 1:
            continue
        if e not in seen:
            clean_ents.append(e)
            seen.add(e)

    rel_texts = [rel_to_text(r) for r in rel_list]

    # GTE entity retrieval
    q_ent_rows = await gte_retrieve(session, question, clean_ents, top_k=12)
    gte_ents = [(r["candidate"], r.get("score", 0)) for r in q_ent_rows if r.get("candidate") and r["candidate"] in entity_list]

    # Token overlap filter
    filtered = [(e, s) for e, s in gte_ents if token_overlap(question, e) >= 0.5]

    # Relation overlap scoring
    q_rel_rows = await gte_retrieve(session, question, rel_list, candidate_texts=rel_texts, top_k=5)
    q_top_rel_idx = {rel_list.index(r["candidate"]) for r in q_rel_rows if r.get("candidate") and r["candidate"] in rel_list}

    name_to_ids = defaultdict(list)
    for i, name in enumerate(entity_list):
        name_to_ids[name].append(i)

    scored = []
    for ent, gte_score in filtered:
        ent_idx = set(name_to_ids.get(ent, []))
        ent_rels = set()
        for h, r, t in zip(h_ids, r_ids, t_ids):
            if h in ent_idx or t in ent_idx:
                ent_rels.add(r)
        overlap = len(ent_rels & q_top_rel_idx)
        scored.append({"entity": ent, "gte": round(gte_score, 4), "overlap": overlap})
    scored.sort(key=lambda x: (-x["overlap"], -x["gte"]))

    return scored, name_to_ids


async def run_case(session, sample, pilot_row):
    question = pilot_row["question"]
    gt_answers = pilot_row.get("gt", [])
    a_entity = sample.get("a_entity", [])

    entity_list = list(sample.get("text_entity_list", []) + sample.get("non_text_entity_list", []))
    ents = entity_list  # alias for main pipeline compat
    rel_list = list(sample.get("relation_list", []))
    rels = rel_list
    h_ids = list(sample.get("h_id_list", []))
    r_ids = list(sample.get("r_id_list", []))
    t_ids = list(sample.get("t_id_list", []))
    rel_texts = [rel_to_text(r) for r in rel_list]

    # Clean entity candidates for GTE
    ent_candidates = []
    seen = set()
    for e in entity_list:
        if not e or is_cvt_like(e) or len(e) <= 1:
            continue
        if e not in seen:
            ent_candidates.append(e)
            seen.add(e)

    # ── Step ①②: Our NER entity resolution ──────────────────────────
    scored, name_to_ids = await resolve_anchor_ner(session, question, entity_list, rel_list, h_ids, r_ids, t_ids)

    # Top entities for decomposition
    top_ents = []
    seen_e = set()
    for s in scored[:6]:
        if s["entity"] not in seen_e:
            top_ents.append((s["entity"], s["gte"]))
            seen_e.add(s["entity"])

    if not top_ents:
        return {"case_id": pilot_row["case_id"], "question": question, "error": "no entities"}

    # Decompose with our prompt
    ent_str = ", ".join(f'("{e}", {s:.1f})' for e, s in top_ents)
    decomp_raw = await call_llm(session, [
        {"role": "system", "content": DECOMP_PROMPT},
        {"role": "user", "content": f"Question: {question}\nRetrieved entities: [{ent_str}]"},
    ], max_tokens=600)

    start_name = parse_our_start(decomp_raw)
    steps = parse_our_steps(decomp_raw)
    end_name, end_query = parse_our_end(decomp_raw)

    if not steps:
        return {"case_id": pilot_row["case_id"], "question": question, "error": "no steps", "raw": decomp_raw}

    # Resolve anchor idx
    anchor_idx = None
    if start_name:
        sn = normalize(start_name)
        for s in scored:
            en = normalize(s["entity"])
            if sn == en or sn in en or en in sn:
                anchor_idx = name_to_ids.get(s["entity"], [None])[0]
                start_name = s["entity"]
                break
    if anchor_idx is None and scored:
        start_name = scored[0]["entity"]
        anchor_idx = name_to_ids.get(start_name, [None])[0]

    # ── Step ③: Endpoint resolve (reuse main pipeline) ───────────────
    breakpoints = {}
    if end_name and end_query:
        ep_rows = await gte_retrieve(session, end_query, ent_candidates, top_k=3)
        ep_cands = [r.get("candidate", "") for r in ep_rows if r.get("candidate")]
        ep_ctx = get_entity_contexts(ep_cands, h_ids, r_ids, t_ids, ents, rels)
        ep_cands_with_ctx = [(n, ep_ctx.get(n, "")) for n in ep_cands]
        best = await llm_resolve_entity(session, question, end_query, ep_cands_with_ctx)
        idx = ents.index(best) if best and best in ents else None
        if idx is not None:
            breakpoints[steps[-1]["step"]] = idx

    # ── Step ④: Dual GTE relation retrieval (reuse main pipeline) ────
    step_candidates = {}
    for step in steps:
        rq = step.get("relation_query", step["question"])
        gte_all = {}
        for query in [rq, step["question"]]:
            rows = await gte_retrieve(session, query, rels, candidate_texts=rel_texts, top_k=10)
            for r in rows:
                cand = r.get("candidate", "")
                score = round(r.get("score", 0), 4)
                if cand in rels:
                    idx = rels.index(cand)
                    if idx not in gte_all or score > gte_all[idx][1]:
                        gte_all[idx] = (rels[idx], score)
        gte_candidates = sorted(gte_all.items(), key=lambda x: -x[1][1])
        step_candidates[step["step"]] = [(idx, name, score) for idx, (name, score) in gte_candidates]

    # ── Step ⑤: Unified LLM prune (reuse main pipeline) ─────────────
    prune_result, prune_debug = await llm_prune_all_relations(session, question, steps, step_candidates)

    step_relations = []
    for step in steps:
        pruned = prune_result.get(step["step"], set())
        step_relations.append(pruned)

    # ── Step ⑥: relation_prior_expand + subgraph from path nodes ────
    paths, max_depth, max_cov = [], 0, 0
    all_triples = []  # (h_idx, r_idx, t_idx) for subgraph
    all_subgraph_nodes = set()
    if anchor_idx is not None:
        bp_set = set(breakpoints.values()) - {anchor_idx, None}

        # Full multi-layer expansion for path ranking
        paths, max_depth, max_cov = relation_prior_expand(
            anchor_idx, step_relations, h_ids, r_ids, t_ids, ents,
            explicit_targets=bp_set if bp_set else None,
            max_hops=len(steps),
        )

        # Collect ALL path nodes (includes bridge nodes from relation_prior_expand)
        all_subgraph_nodes = {anchor_idx}
        for p in paths:
            all_subgraph_nodes.update(p["nodes"])

        # Per-step HR collection: also add frontier nodes from each layer
        # (catches nodes like Mozambique that get dropped in later layers)
        frontier = {anchor_idx}
        for step_rels in step_relations:
            if not step_rels:
                continue
            next_frontier = set()
            for hi, ri, ti in zip(h_ids, r_ids, t_ids):
                if ri not in step_rels:
                    continue
                if hi in frontier:
                    all_triples.append((hi, ri, ti))
                    next_frontier.add(ti)
                if ti in frontier:
                    all_triples.append((hi, ri, ti))
                    next_frontier.add(hi)
            all_subgraph_nodes |= next_frontier
            frontier = next_frontier

    # ── Step ⑦: Build answer candidates from subgraph ────────────────
    answer_candidates = []
    gt_answers_idx = set()

    # Resolve GT entity indices
    for gt in gt_answers:
        gt_norm = normalize(gt)
        for i, e in enumerate(ents):
            e_norm = normalize(e)
            if gt_norm in e_norm or e_norm in gt_norm:
                gt_answers_idx.add(i)

    # Collect named entities from subgraph (skip CVT)
    for idx in sorted(all_subgraph_nodes):
        if idx == anchor_idx:
            continue
        name = ents[idx] if 0 <= idx < len(ents) else ""
        if is_cvt_like(name):
            for cvt_idx, _ in expand_through_cvt(idx, h_ids, r_ids, t_ids, ents):
                if cvt_idx != anchor_idx and 0 <= cvt_idx < len(ents) and not is_cvt_like(ents[cvt_idx]):
                    answer_candidates.append(ents[cvt_idx])
        elif name:
            answer_candidates.append(name)

    seen_c = set()
    unique = []
    for c in answer_candidates:
        nc = normalize(c)
        if nc not in seen_c:
            seen_c.add(nc)
            unique.append(c)
    answer_candidates = unique

    # ── Step ⑧: GT check — GT in subgraph nodes ─────────────────────
    gt_hit = bool(gt_answers_idx & all_subgraph_nodes)

    # Fallback: string match on candidates (handles normalization edge cases)
    if not gt_hit:
        for gt in gt_answers:
            gt_norm = normalize(gt)
            for c in answer_candidates:
                c_norm = normalize(c)
                if gt_norm and c_norm and (gt_norm in c_norm or c_norm in gt_norm):
                    gt_hit = True
                    break
            if gt_hit:
                break

    # Fallback: string match on candidates (handles normalization edge cases)
    if not gt_hit:
        for gt in gt_answers:
            gt_norm = normalize(gt)
            for c in answer_candidates:
                c_norm = normalize(c)
                if gt_norm and c_norm and (gt_norm in c_norm or c_norm in gt_norm):
                    gt_hit = True
                    break
            if gt_hit:
                break

    return {
        "case_id": pilot_row["case_id"],
        "question": question,
        "gt_answers": gt_answers,
        "anchor": start_name,
        "anchor_idx": anchor_idx,
        "steps": [{"step": s["step"], "rq": s["relation_query"], "q": s["question"]} for s in steps],
        "selected_rels": {s["step"]: [rels[r] for r in step_relations[i]] for i, s in enumerate(steps)},
        "breakpoints": breakpoints,
        "n_paths": len(paths),
        "n_triples": len(all_triples),
        "n_subgraph_nodes": len(all_subgraph_nodes),
        "max_cov": max_cov,
        "gt_hit": gt_hit,
        "answer_candidates": answer_candidates[:20],
    }


async def amain():
    pilot_rows = json.loads(PILOT.read_text())
    with open(CWQ_PKL, "rb") as f:
        samples = pickle.load(f)
    sample_map = {s["id"]: s for s in samples if "id" in s}

    results = []
    async with aiohttp.ClientSession() as session:
        for pilot_row in pilot_rows:
            sample = sample_map[pilot_row["case_id"]]
            r = await run_case(session, sample, pilot_row)
            results.append(r)

            mark = "✓" if r["gt_hit"] else "✗"
            print(f"=== {r['case_id']} [{mark}] ===")
            print(f"Q: {r['question']}")
            print(f"Anchor: {r['anchor']} (idx={r['anchor_idx']})")
            print(f"GT: {r['gt_answers']}")
            for s in r["steps"]:
                rels_sel = r["selected_rels"].get(s["step"], [])
                print(f"  Step {s['step']}: {s['q'][:60]}... | rels={rels_sel[:3]}")
            print(f"Paths: {r['n_paths']}, triples: {r['n_triples']}, subgraph nodes: {r['n_subgraph_nodes']}, cov: {r['max_cov']}")
            print(f"GT hit: {r['gt_hit']} | Candidates: {r['answer_candidates'][:8]}")
            print()

    hits = sum(1 for r in results if r["gt_hit"])
    print(f"=== GT RECALL: {hits}/10 ===")

    out_dir = ROOT / "reports" / "stage_pipeline_test" / "full_pipeline_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved to {out_dir / 'results.json'}")


if __name__ == "__main__":
    asyncio.run(amain())
