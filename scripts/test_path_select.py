#!/usr/bin/env python3
"""Test model-based path selection from expanded subgraph.

Pipeline: NER resolve → decompose → dual GTE → unified prune → relation_prior_expand
→ format paths as readable text → LLM selects top 3 → GT check on selected paths.
"""
import asyncio
import json
import pickle
import re
import sys
import time
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

PATH_SELECT_PROMPT = """You are analyzing knowledge graph paths for a multi-hop question.

Given the question and several candidate paths through the knowledge graph, select the top 3 paths that best answer the question.

Rules:
1. A good path connects entities mentioned in the question through meaningful relations.
2. Prefer paths whose endpoint could directly answer the question.
3. Prefer shorter, more direct paths over longer ones.
4. Prefer paths that pass through bridge entities that make sense (e.g., an airport for a city-country question).
5. Ignore paths that loop back to the start entity.
6. If fewer than 3 paths are meaningful, select fewer.

Output format:
<analysis>
- Question asks for: [answer type]
- Key constraints: [list constraints]
- Path evaluation: [brief evaluation of top paths]
</analysis>

Selected paths (ranked):
1. Path [N]: [one-line reason]
2. Path [N]: [one-line reason]
3. Path [N]: [one-line reason]
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
    clean_ents, seen = [], set()
    for e in entity_list:
        if not e or is_cvt_like(e) or len(e) <= 1:
            continue
        if e not in seen:
            clean_ents.append(e)
            seen.add(e)

    rel_texts = [rel_to_text(r) for r in rel_list]

    q_ent_rows = await gte_retrieve(session, question, clean_ents, top_k=12)
    gte_ents = [(r["candidate"], r.get("score", 0)) for r in q_ent_rows if r.get("candidate") and r["candidate"] in entity_list]

    filtered = [(e, s) for e, s in gte_ents if token_overlap(question, e) >= 0.5]

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


def format_path_text(path, ents, rels):
    """Format a single path as readable text."""
    nodes = path["nodes"]
    relations = path["relations"]
    parts = [ents[nodes[0]] if 0 <= nodes[0] < len(ents) else f"#{nodes[0]}"]
    for i, r_idx in enumerate(relations):
        r_name = rels[r_idx] if 0 <= r_idx < len(rels) else f"r#{r_idx}"
        next_ent = ents[nodes[i + 1]] if i + 1 < len(nodes) and 0 <= nodes[i + 1] < len(ents) else f"#{nodes[i+1]}"
        parts.append(f"--[{r_name}]--> {next_ent}")
    return " ".join(parts)


def parse_selected_paths(raw):
    """Parse model's path selection output, return list of path numbers."""
    selected = []
    for m in re.finditer(r"(?:^|\n)\s*\d+\.\s+Path\s+(\d+)", raw):
        try:
            selected.append(int(m.group(1)))
        except ValueError:
            pass
    return selected


BLIND_REASON_PROMPT = """Answer the following multi-hop question using your own knowledge.

Rules:
- Think step by step.
- Identify key entities and their relationships.
- Give a final concise answer.

<reasoning>
[Your step-by-step analysis]
</reasoning>

Answer: [your answer]
"""


KG_REASON_PROMPT = """You previously analyzed a question. Now you are given a knowledge graph subgraph to verify or correct your answer.

Use the subgraph facts to reason about the correct answer. If the subgraph contradicts your initial analysis, correct it.

Subgraph triples:
{subgraph_text}

Previous analysis:
{blind_reasoning}

Question: {question}

<verification>
- Compare your initial analysis with the subgraph facts.
- Identify any contradictions or confirmations.
- Trace the path through the subgraph that answers the question.
</verification>

Final Answer: [your answer based on the verified reasoning]
"""


def build_selected_subgraph(selected_indices, paths, step_relations, h_ids, r_ids, t_ids, ents, rels, anchor_idx):
    """Build subgraph triples from selected paths + HR frontier."""
    # Collect all selected path nodes
    selected_path_nodes = set()
    for idx in selected_indices:
        selected_path_nodes.update(paths[idx]["nodes"])

    # All selected step relations (flat set)
    all_step_rels = set()
    for sr in step_relations:
        all_step_rels.update(sr)

    # Collect triples along selected paths (exact path edges)
    path_triples = set()
    for idx in selected_indices:
        p = paths[idx]
        for i, r_idx in enumerate(p["relations"]):
            h, t = p["nodes"][i], p["nodes"][i + 1]
            path_triples.add((h, r_idx, t))

    # HR frontier: from selected path nodes, collect all triples matching step relations
    hr_triples = set()
    for hi, ri, ti in zip(h_ids, r_ids, t_ids):
        if ri not in all_step_rels:
            continue
        if hi in selected_path_nodes or ti in selected_path_nodes:
            hr_triples.add((hi, ri, ti))

    # CVT expansion: expand CVT nodes in selected path nodes
    cvt_expanded_nodes = set()
    for nid in selected_path_nodes:
        name = ents[nid] if 0 <= nid < len(ents) else ""
        if is_cvt_like(name):
            for cvt_idx, _ in expand_through_cvt(nid, h_ids, r_ids, t_ids, ents):
                cvt_expanded_nodes.add(cvt_idx)
                # Also add triples connecting to CVT-expanded nodes
                for hi, ri, ti in zip(h_ids, r_ids, t_ids):
                    if hi == nid or ti == nid:
                        hr_triples.add((hi, ri, ti))

    all_triples = path_triples | hr_triples
    # Include HR frontier target nodes (not just path nodes)
    all_nodes = set(selected_path_nodes) | cvt_expanded_nodes
    for h, r, t in all_triples:
        all_nodes.add(h)
        all_nodes.add(t)

    return all_triples, all_nodes


def format_subgraph_text(triples, ents, rels):
    """Format subgraph triples as readable text for LLM."""
    lines = []
    seen = set()
    for h, r, t in sorted(triples, key=lambda x: (x[0], x[2])):
        key = (h, r, t)
        if key in seen:
            continue
        seen.add(key)
        h_name = ents[h] if 0 <= h < len(ents) else f"#{h}"
        r_name = rels[r] if 0 <= r < len(rels) else f"r#{r}"
        t_name = ents[t] if 0 <= t < len(ents) else f"#{t}"
        # Shorten relation for readability
        r_short = r_name.rsplit(".", 1)[-1].replace("_", " ") if "." in r_name else r_name
        lines.append(f"- {h_name} --[{r_short}]--> {t_name}")
    return "\n".join(lines)


def parse_answer(raw):
    """Extract final answer from model output."""
    # Try "Final Answer:" first, then "Answer:"
    for prefix in ["Final Answer:", "Answer:"]:
        m = re.search(rf"{re.escape(prefix)}\s*(.+?)(?:\n|$)", raw)
        if m:
            return m.group(1).strip()
    # Fallback: last non-empty line
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


async def run_case(session, sample, pilot_row):
    t0 = time.time()
    timings = {}

    question = pilot_row["question"]
    gt_answers = pilot_row.get("gt", [])

    entity_list = list(sample.get("text_entity_list", []) + sample.get("non_text_entity_list", []))
    ents = entity_list
    rel_list = list(sample.get("relation_list", []))
    rels = rel_list
    h_ids = list(sample.get("h_id_list", []))
    r_ids = list(sample.get("r_id_list", []))
    t_ids = list(sample.get("t_id_list", []))
    rel_texts = [rel_to_text(r) for r in rel_list]

    # Clean entity candidates
    ent_candidates = []
    seen = set()
    for e in entity_list:
        if not e or is_cvt_like(e) or len(e) <= 1:
            continue
        if e not in seen:
            ent_candidates.append(e)
            seen.add(e)

    # ── Step ①②: NER entity resolution ─────────────────────────────────
    ta = time.time()
    scored, name_to_ids = await resolve_anchor_ner(session, question, entity_list, rel_list, h_ids, r_ids, t_ids)
    timings["ner_resolve"] = round(time.time() - ta, 2)

    top_ents = []
    seen_e = set()
    for s in scored[:6]:
        if s["entity"] not in seen_e:
            top_ents.append((s["entity"], s["gte"]))
            seen_e.add(s["entity"])

    if not top_ents:
        return {"case_id": pilot_row["case_id"], "question": question, "error": "no entities", "timings": timings}

    # Decompose
    ta = time.time()
    ent_str = ", ".join(f'("{e}", {s:.1f})' for e, s in top_ents)
    decomp_raw = await call_llm(session, [
        {"role": "system", "content": DECOMP_PROMPT},
        {"role": "user", "content": f"Question: {question}\nRetrieved entities: [{ent_str}]"},
    ], max_tokens=600)
    timings["decompose"] = round(time.time() - ta, 2)

    start_name = parse_our_start(decomp_raw)
    steps = parse_our_steps(decomp_raw)
    end_name, end_query = parse_our_end(decomp_raw)

    if not steps:
        return {"case_id": pilot_row["case_id"], "question": question, "error": "no steps", "raw": decomp_raw, "timings": timings}

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

    # ── Step ③: Endpoint resolve ────────────────────────────────────────
    ta = time.time()
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
    timings["endpoint_resolve"] = round(time.time() - ta, 2)

    # ── Step ④: Dual GTE relation retrieval ─────────────────────────────
    ta = time.time()
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
    timings["dual_gte"] = round(time.time() - ta, 2)

    # ── Step ⑤: Unified LLM prune ──────────────────────────────────────
    ta = time.time()
    prune_result, prune_debug = await llm_prune_all_relations(session, question, steps, step_candidates)
    timings["llm_prune"] = round(time.time() - ta, 2)

    step_relations = []
    for step in steps:
        pruned = prune_result.get(step["step"], set())
        step_relations.append(pruned)

    # ── Step ⑥: relation_prior_expand ───────────────────────────────────
    ta = time.time()
    paths, max_depth, max_cov = [], 0, 0
    all_subgraph_nodes = set()
    all_triples = []
    if anchor_idx is not None:
        bp_set = set(breakpoints.values()) - {anchor_idx, None}
        paths, max_depth, max_cov = relation_prior_expand(
            anchor_idx, step_relations, h_ids, r_ids, t_ids, ents,
            explicit_targets=bp_set if bp_set else None,
            max_hops=len(steps),
        )
        # Path nodes
        all_subgraph_nodes = {anchor_idx}
        for p in paths:
            all_subgraph_nodes.update(p["nodes"])
        # HR frontier nodes (from test_full_pipeline.py)
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
    timings["expand"] = round(time.time() - ta, 2)

    # ── Step ⑦: Format paths for model selection ────────────────────────
    path_texts = []
    for i, p in enumerate(paths[:30]):  # cap at 30 paths
        text = format_path_text(p, ents, rels)
        cov = p.get("covered_steps", frozenset())
        cov_str = ",".join(str(s) for s in sorted(cov))
        path_texts.append(f"Path {i + 1}: {text}  [covers steps: {cov_str}]")

    # ── Step ⑧: Model path selection ────────────────────────────────────
    ta = time.time()
    selected_indices = []
    selection_raw = ""
    if path_texts:
        paths_str = "\n".join(path_texts)
        user_msg = f"Question: {question}\n\nCandidate paths:\n{paths_str}"
        selection_raw = await call_llm(session, [
            {"role": "system", "content": PATH_SELECT_PROMPT},
            {"role": "user", "content": user_msg},
        ], max_tokens=2000)

        selected_nums = parse_selected_paths(selection_raw)
        for n in selected_nums[:3]:
            idx = n - 1  # 0-indexed
            if 0 <= idx < len(paths):
                selected_indices.append(idx)
    timings["path_select_llm"] = round(time.time() - ta, 2)

    # ── Step ⑨: Build selected subgraph ────────────────────────────────
    ta = time.time()
    selected_triples, selected_nodes = set(), set()
    subgraph_text = ""
    if selected_indices:
        selected_triples, selected_nodes = build_selected_subgraph(
            selected_indices, paths, step_relations, h_ids, r_ids, t_ids, ents, rels, anchor_idx
        )
        subgraph_text = format_subgraph_text(selected_triples, ents, rels)
    timings["build_subgraph"] = round(time.time() - ta, 2)

    # ── Step ⑩a: Blind reasoning (no KG) ───────────────────────────────
    ta = time.time()
    blind_raw = await call_llm(session, [
        {"role": "system", "content": BLIND_REASON_PROMPT},
        {"role": "user", "content": f"Question: {question}"},
    ], max_tokens=800)
    blind_answer = parse_answer(blind_raw)
    timings["blind_reason"] = round(time.time() - ta, 2)

    # ── Step ⑩b: KG-verified reasoning ─────────────────────────────────
    ta = time.time()
    kg_raw = ""
    kg_answer = ""
    if subgraph_text:
        kg_raw = await call_llm(session, [
            {"role": "system", "content": KG_REASON_PROMPT.format(
                subgraph_text=subgraph_text,
                blind_reasoning=blind_raw,
                question=question,
            )},
            {"role": "user", "content": question},
        ], max_tokens=1200)
        kg_answer = parse_answer(kg_raw)
    timings["kg_reason"] = round(time.time() - ta, 2)

    # ── GT check ────────────────────────────────────────────────────────
    gt_answers_idx = set()
    for gt in gt_answers:
        gt_norm = normalize(gt)
        for i, e in enumerate(ents):
            e_norm = normalize(e)
            if gt_norm in e_norm or e_norm in gt_norm:
                gt_answers_idx.add(i)

    # CVT-expanded node set for all subgraph nodes
    all_expanded_nodes = set(all_subgraph_nodes)
    for idx in list(all_subgraph_nodes):
        name = ents[idx] if 0 <= idx < len(ents) else ""
        if is_cvt_like(name):
            for cvt_idx, _ in expand_through_cvt(idx, h_ids, r_ids, t_ids, ents):
                all_expanded_nodes.add(cvt_idx)

    # GT check on ALL (paths + HR frontier + CVT expansion)
    gt_hit_all = bool(gt_answers_idx & all_expanded_nodes)
    if not gt_hit_all:
        for gt in gt_answers:
            gt_norm = normalize(gt)
            for idx in all_expanded_nodes:
                if 0 <= idx < len(ents):
                    e_norm = normalize(ents[idx])
                    if gt_norm and e_norm and (gt_norm in e_norm or e_norm in gt_norm):
                        gt_hit_all = True
                        break
            if gt_hit_all:
                break

    # GT check on SELECTED paths (+ HR frontier of ALL paths)
    selected_nodes = set()
    for idx in selected_indices:
        selected_nodes.update(paths[idx]["nodes"])
    for idx in selected_indices:
        for nid in paths[idx]["nodes"]:
            if is_cvt_like(ents[nid] if 0 <= nid < len(ents) else ""):
                for cvt_idx, _ in expand_through_cvt(nid, h_ids, r_ids, t_ids, ents):
                    selected_nodes.add(cvt_idx)

    gt_hit_selected = bool(gt_answers_idx & selected_nodes)
    if not gt_hit_selected:
        for gt in gt_answers:
            gt_norm = normalize(gt)
            for idx in selected_indices:
                for nid in paths[idx]["nodes"]:
                    if 0 <= nid < len(ents):
                        e_norm = normalize(ents[nid])
                        if gt_norm and e_norm and (gt_norm in e_norm or e_norm in gt_norm):
                            gt_hit_selected = True
                            break
                if gt_hit_selected:
                    break

    timings["total"] = round(time.time() - t0, 2)

    return {
        "case_id": pilot_row["case_id"],
        "question": question,
        "gt_answers": gt_answers,
        "anchor": start_name,
        "anchor_idx": anchor_idx,
        "steps": [{"step": s["step"], "rq": s["relation_query"], "q": s["question"]} for s in steps],
        "n_paths": len(paths),
        "n_triples": len(all_triples),
        "n_subgraph_nodes": len(all_subgraph_nodes),
        "selected_indices": selected_indices,
        "selection_raw": selection_raw,
        "selected_triples_count": len(selected_triples),
        "selected_nodes_count": len(selected_nodes),
        "subgraph_text": subgraph_text,
        "blind_raw": blind_raw,
        "blind_answer": blind_answer,
        "kg_raw": kg_raw,
        "kg_answer": kg_answer,
        "gt_hit_all": gt_hit_all,
        "gt_hit_selected": gt_hit_selected,
        "gt_answers_idx": list(gt_answers_idx),
        "timings": timings,
    }


async def amain():
    pilot_rows = json.loads(PILOT.read_text())
    with open(CWQ_PKL, "rb") as f:
        samples = pickle.load(f)
    sample_map = {s["id"]: s for s in samples if "id" in s}

    results = []
    async with aiohttp.ClientSession() as session:
        for pilot_row in pilot_rows:
            r = await run_case(session, sample_map[pilot_row["case_id"]], pilot_row)
            results.append(r)

            mark_all = "Y" if r.get("gt_hit_all") else "N"
            mark_sel = "Y" if r.get("gt_hit_selected") else "N"
            t = r.get("timings", {})
            print(f"=== {r['case_id']} [all={mark_all} sel={mark_sel}] {t.get('total',0)}s ===")
            print(f"Q: {r['question']}")
            print(f"Anchor: {r['anchor']} | Paths: {r['n_paths']} | Nodes: {r.get('n_subgraph_nodes',0)} | Selected: {r['selected_indices']}")
            print(f"Timing: ner={t.get('ner_resolve',0)} decomp={t.get('decompose',0)} ep={t.get('endpoint_resolve',0)} "
                  f"gte={t.get('dual_gte',0)} prune={t.get('llm_prune',0)} expand={t.get('expand',0)} "
                  f"select={t.get('path_select_llm',0)} blind={t.get('blind_reason',0)} kg={t.get('kg_reason',0)}")
            # Selected subgraph
            sg = r.get("subgraph_text", "")
            if sg:
                print(f"\nSubgraph ({r.get('selected_triples_count',0)} triples, {r.get('selected_nodes_count',0)} nodes):")
                for line in sg.split("\n")[:15]:
                    print(f"  {line}")
                if sg.count("\n") > 15:
                    print(f"  ... ({sg.count(chr(10)) - 15} more triples)")
            # Reasoning
            print(f"\nBlind answer: {r.get('blind_answer', '')}")
            print(f"KG answer:    {r.get('kg_answer', '')}")
            print(f"GT answers:   {r['gt_answers']}")
            # Check answer correctness
            gt_norms = [normalize(a) for a in r['gt_answers'] if normalize(a)]
            for label, ans in [("blind", r.get("blind_answer", "")), ("kg", r.get("kg_answer", ""))]:
                if not ans:
                    continue
                ans_norm = normalize(ans)
                hit = any(g in ans_norm or ans_norm in g for g in gt_norms if g)
                print(f"  {label}_hit={hit}")
            print()

    hits_all = sum(1 for r in results if r.get("gt_hit_all"))
    hits_sel = sum(1 for r in results if r.get("gt_hit_selected"))
    print(f"=== GT RECALL: all_paths={hits_all}/10 | selected_paths={hits_sel}/10 ===")

    out_dir = ROOT / "reports" / "stage_pipeline_test" / "path_select_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved to {out_dir / 'results.json'}")


if __name__ == "__main__":
    asyncio.run(amain())
