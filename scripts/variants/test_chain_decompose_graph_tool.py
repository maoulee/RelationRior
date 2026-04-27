#!/usr/bin/env python3
"""Quick integration test: chain decomposition → GTE retrieval → graph traversal."""
from __future__ import annotations
import asyncio, json, pickle, re
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import aiohttp

try:
    import graph_tool.all as gt
    _HAS_GT = True
except ImportError:
    _HAS_GT = False

ROOT = Path(__file__).resolve().parents[1]
LLM_API_URL = "http://localhost:8000/v1/chat/completions"
LLM_MODEL = "qwen35-9b-local"
GTE_API_URL = "http://localhost:8003"
DEFAULT_PILOT = ROOT / "reports/stage_pipeline_test/find_check_plan_pilot_10cases/results.json"
DEFAULT_CWQ = Path("/zhaoshu/SubgraphRAG-main/retrieve/data_files/cwq/processed/test.pkl")
DEFAULT_OUTPUT = ROOT / "reports/stage_pipeline_test/chain_decompose_test"

DECOMP_PROMPT = '''Decompose the question into an ordered retrieval plan.

This is a decomposition task, not answer generation.
Derive the plan only from the current question. Do not reuse, quote, or imitate benchmark cases, memorized examples, or prior dataset items.

Think silently before writing the final decomposition:
1. Identify the best visible anchor entity in the question.
2. Identify the final answer type.
3. Determine the main retrieval chain needed to reach the answer.
4. Separate "find" steps from later "check/filter" constraints.
5. Prefer a short ordered chain where each step adds one new semantic relation.
6. Decide whether the LAST step contains a concrete fixed entity mentioned in the question.

Decomposition rules:
1. START from one concrete anchor entity explicitly mentioned in the question.
2. Use 1 to 4 ordered steps.
3. Each step should express ONE retrieval relation or ONE check/filter relation.
4. Earlier steps should build the answer path; later steps may verify or filter candidates.
5. If a numeric, temporal, geographic, or descriptive condition does not define the main path, place it as a later check/filter step.
6. For anchor and endpoints, provide the best semantic entity search term, not a full sentence.
7. For each step, provide a compact relation_query that names the relation family needed for retrieval, not a vague phrase. Prefer content words like "airport serves city", "country borders", "film cast child role", "language spoken in country", "country predominant religion". Avoid generic phrases like "located in", "role", "country of origin", "related to", "associated with".
8. The default endpoint rule is: only the LAST step may carry an endpoint, and only if that last step contains a fixed entity explicitly given in the question. Otherwise use `none`.
9. Earlier steps should normally use `endpoint: none`, unless a fixed entity from the question must explicitly appear to make the step meaningful.
10. Never output placeholder endpoints such as "[Country Name]", "team name", "movie name", "actor name", "year", or any bracketed template.
11. Do not output chain-of-thought, hidden reasoning, explanations, examples, or alternative plans.

Output format:
Anchor: [entity name] (entity_query: [search term for entity retrieval])
1. [sub-question] (relation_query: [search term for relation retrieval]; endpoint: [entity name (entity_query: [search term]) or none])
2. ...

Good decomposition properties:
- The anchor is a visible entity from the question text.
- The relation_query is broad enough for semantic retrieval but specific to the step.
- The ordered steps reflect dependency: step N should make sense after step N-1.
- The final steps may express checks/filters rather than direct answer retrieval.
- Endpoints are sparse and usually only appear in the last step.
- Endpoints are only for fixed entities already present in the question, not unknown answer slots.
- relation_query should use a compact semantic phrase, not a vague generic verb.

Return only the decomposition in the exact format above.
'''


def is_cvt_like(name: str) -> bool:
    """Detect CVT (Compound Value Type) nodes.
    Matches m.xxx / g.xxx pattern OR entities in non_text with no readable name."""
    if not name or len(name) < 2:
        return False
    if re.match(r"^[mg]\.[A-Za-z0-9_]+$", name):
        return True
    return False


def expand_cvt_leaves(ents, rels, h_ids, r_ids, t_ids):
    """Auto-expand CVT leaf nodes (degree ≤ 1) by finding additional edges
    from other triples in the subgraph that share the same CVT node.
    Returns potentially augmented (ents, rels, h_ids, r_ids, t_ids)."""
    from collections import Counter
    node_degree = Counter()
    for i in range(len(h_ids)):
        node_degree[h_ids[i]] += 1
        node_degree[t_ids[i]] += 1

    # Find CVT nodes with degree ≤ 1 (leaf/dead-end)
    cvt_leaves = []
    for idx, name in enumerate(ents):
        if is_cvt_like(name) and node_degree.get(idx, 0) <= 1:
            cvt_leaves.append(idx)

    if not cvt_leaves:
        return ents, rels, h_ids, r_ids, t_ids

    # For each CVT leaf, search the subgraph for any triples where it appears
    # that weren't included (e.g., via shared intermediate nodes)
    # Since we only have the subgraph, we can only find edges already present
    # but potentially missed due to indexing
    # No-op for now: full expansion requires KG API access
    # Flag count for debugging
    return ents, rels, h_ids, r_ids, t_ids


def rel_to_text(rel: str) -> str:
    """Return original dot-notation format for LLM display: 'people.person.religion'"""
    return rel


def rel_to_text_short(rel: str) -> str:
    """Short format (last 2 segments) for GTE retrieval: 'person religion'"""
    parts = rel.split(".")
    return " ".join(p.replace("_", " ") for p in parts[-2:]) if len(parts) >= 2 else rel.replace("_", " ")


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9%.' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def get_entity_contexts(entity_names, h_ids, r_ids, t_ids, ents, rels):
    """For each candidate entity, extract surrounding relation context from subgraph.
    Similar to check_entities' [Context: ...] annotation."""
    # Build name→idx lookup
    name_to_indices = {}
    for i, e in enumerate(ents):
        name_to_indices.setdefault(e, []).append(i)

    contexts = {}
    for cand in entity_names:
        indices = name_to_indices.get(cand, [])
        if not indices:
            continue
        idx_set = set(indices)
        outgoing = []
        incoming = []
        for i in range(len(h_ids)):
            if h_ids[i] in idx_set:
                rel_text = rel_to_text(rels[r_ids[i]]) if 0 <= r_ids[i] < len(rels) else "?"
                t_name = ents[t_ids[i]] if 0 <= t_ids[i] < len(ents) else "?"
                if not is_cvt_like(t_name) and t_name != cand:
                    outgoing.append(f"{rel_text}→{t_name}")
            if t_ids[i] in idx_set:
                rel_text = rel_to_text(rels[r_ids[i]]) if 0 <= r_ids[i] < len(rels) else "?"
                h_name = ents[h_ids[i]] if 0 <= h_ids[i] < len(ents) else "?"
                if not is_cvt_like(h_name) and h_name != cand:
                    incoming.append(f"{h_name}→{rel_text}")
        parts = outgoing[:2] + incoming[:1]
        if parts:
            contexts[cand] = "; ".join(parts)
    return contexts


async def llm_resolve_entity(session, question, query, candidates_with_ctx):
    """LLM selects the correct entity from GTE top-k candidates using surrounding relation context.
    candidates_with_ctx: list of (name, context_str) tuples.
    Returns selected entity name or None."""
    if not candidates_with_ctx:
        return None
    if len(candidates_with_ctx) == 1:
        return candidates_with_ctx[0][0]

    cand_lines = []
    for i, (name, ctx) in enumerate(candidates_with_ctx, 1):
        cand_lines.append(f"  {i}. {name} [{ctx}]" if ctx else f"  {i}. {name}")

    prompt = f"""Search query: {query}

Candidate entities (with relation context from knowledge graph):
{chr(10).join(cand_lines)}

Which candidate best matches the search query? Use the relation context to identify what each entity actually IS (a person, a location, a schema type, etc). Pick the specific entity, not generic types or schema entries.

<analysis>Brief reasoning about which candidate matches the query</analysis>
<selected>entity name</selected>"""

    for _ in range(2):
        raw = await call_llm(session, [
            {"role": "system", "content": "Select the correct entity from candidates. Output <analysis> and <selected> XML tags."},
            {"role": "user", "content": prompt},
        ], max_tokens=300)
        sel = extract_xml_tag(raw, "selected")
        if sel:
            sel = sel.strip().strip('"').strip("'")
            # Match to candidate names
            for name, _ in candidates_with_ctx:
                if normalize(sel) == normalize(name):
                    return name
            # Fuzzy match
            for name, _ in candidates_with_ctx:
                if normalize(sel) in normalize(name) or normalize(name) in normalize(sel):
                    return name
    # Fallback to GTE top-1
    return candidates_with_ctx[0][0]


def candidate_hit(cands: List[str], targets: List[str]) -> bool:
    norm_cands = [normalize(c) for c in cands]
    for t in targets:
        nt = normalize(t)
        for c in norm_cands:
            if c == nt or nt in c or c in nt:
                return True
    return False


async def call_llm(session: aiohttp.ClientSession, messages: list, max_tokens: int = 500, retries: int = 3) -> str:
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 0.8,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for attempt in range(retries):
        try:
            async with session.post(LLM_API_URL, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                data = await resp.json()
            return data["choices"][0]["message"]["content"]
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt < retries - 1:
                await asyncio.sleep(2)
            else:
                raise


async def gte_retrieve(session, query, candidates, candidate_texts=None, top_k=10):
    payload = {"query": query, "candidates": candidates, "candidate_texts": candidate_texts, "top_k": top_k}
    async with session.post(f"{GTE_API_URL}/retrieve", json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
        data = await resp.json()
    return data.get("results", [])


def score_causal_tier(hit_set, bridge_length=0):
    """Score path by decomposition layer coverage using lexicographic tuple.

    Returns (hit_count, max_layer_hit, -bridge_length) for comparison.
    Higher = better. Layer dedup: each layer counted once regardless of
    how many of its relations match.

    3-step: {R1,R2,R3} > {R1,R3}={R2,R3} > {R1,R2} > {R3} > {R2} > {R1}
      (3,2,*)       (2,2,*)  (2,2,*)    (2,1,*)   (1,2,*)  (1,1,*)  (1,0,*)
    """
    if not hit_set:
        return (0, -1, 0)
    return (len(hit_set), max(hit_set), -bridge_length)


def extract_xml_tag(text, tag):
    """Extract content between <tag>...</tag>, returns None if not found."""
    match = re.search(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    return match.group(1).strip() if match else None


async def llm_prune_all_relations(session, question, all_steps, step_candidates):
    """Single LLM call to prune relations for ALL steps at once.

    Args:
        all_steps: list of parsed step dicts
        step_candidates: dict mapping step_num -> list of (idx, rel_name, score)

    Returns:
        dict mapping step_num -> set of selected relation indices,
        and debug dict with full prompt/response
    """
    # Build the prompt with all steps' candidates
    chain_lines = []
    step_blocks = []
    for s in all_steps:
        sn = s["step"]
        ep_str = f" -> endpoint: {s['endpoint']}" if s.get("endpoint") else ""
        chain_lines.append(f"  Step {sn}: {s['question']}{ep_str}")

        cands = step_candidates.get(sn, [])
        if not cands:
            step_blocks.append(f"Step {sn}: {s['question']}\n  Purpose: {s.get('relation_query', '')}\n  Candidates: (none)")
            continue

        cand_lines = [f"    {i}. {name}"
                      for i, (idx, name, score) in enumerate(cands, 1)]
        step_blocks.append(
            f"Step {sn}: {s['question']}\n"
            f"  Purpose: {s.get('relation_query', '')}\n"
            f"  Candidates:\n" + "\n".join(cand_lines)
        )

    chain_text = "\n".join(chain_lines)
    blocks_text = "\n\n".join(step_blocks)

    prompt = f"""Analyze and select knowledge graph relations for each step of this reasoning chain.

Question: {question}

Reasoning chain:
{chain_text}

Step-by-step candidates:
{blocks_text}

Rules:
1. Each step connects FROM previous output TO next — select bridge relations
2. Prefer 2-3 precise relations over many noisy ones
3. Ignore unrelated attributes (currency, codes when asking about geography)
4. If no relations fit a step, output empty list

Output format:
<analysis>
One sentence per step: what it needs and which relations fit.
</analysis>
<selected>
step_1: [1, 3]
step_2: [2, 5]
</selected>"""

    system = "You are a knowledge graph relation selector for multi-step QA. Analyze the full chain, then select relevant relations per step. Output <analysis> and <selected> XML tags."

    # Try up to 3 times (1 initial + 2 retries)
    for attempt in range(3):
        raw = await call_llm(session, [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ], max_tokens=1500)

        selected_yaml = extract_xml_tag(raw, "selected")
        if selected_yaml:
            break
    else:
        # All attempts failed to produce <selected> tag
        selected_yaml = None

    # Parse YAML-style step_N: [1, 3, 5]
    result = {}
    if selected_yaml:
        for line in selected_yaml.split('\n'):
            line = line.strip()
            m = re.match(r'step_(\d+)\s*:\s*\[(.*?)\]', line)
            if m:
                sn = int(m.group(1))
                nums = [int(x.strip()) for x in m.group(2).split(',') if x.strip().isdigit()]
                cands = step_candidates.get(sn, [])
                selected_indices = set()
                for n in nums:
                    if 1 <= n <= len(cands):
                        selected_indices.add(cands[n - 1][0])
                result[sn] = selected_indices

    # Fallback: for any step missing from result, use top-3 GTE
    for s in all_steps:
        sn = s["step"]
        if sn not in result or not result[sn]:
            cands = step_candidates.get(sn, [])
            result[sn] = set(idx for idx, _, _ in cands[:3])

    debug = {
        "prompt": prompt,
        "response": raw,
        "parsed_yaml": selected_yaml,
    }
    return result, debug


async def llm_reselect_single_step_relation(session, question, step, step_candidates, current_indices):
    """Reselect relations for one failed step, avoiding the current failed choice set."""
    cands = step_candidates.get(step["step"], [])
    if not cands:
        return set()

    cand_lines = []
    current_pos = set()
    for i, (idx, name, score) in enumerate(cands, 1):
        marker = " [CURRENT]" if idx in current_indices else ""
        if idx in current_indices:
            current_pos.add(i)
        cand_lines.append(f"  {i}. {name}{marker}")

    prompt = f"""The current relation choice for one reasoning step appears to be wrong or too noisy.

Question: {question}
Failed step: {step['question']}
Step purpose: {step.get('relation_query', '')}

Candidate relations:
{chr(10).join(cand_lines)}

Select 1 to 3 BETTER alternative relations for this step.

Rules:
- Prefer relations that directly express the step semantics.
- Avoid the currently marked failed choices if better alternatives exist.
- Do not select generic or weakly related relations just because they are broad.

Output format:
<analysis>One short sentence.</analysis>
<selected>comma-separated candidate numbers only</selected>"""

    raw = await call_llm(session, [
        {"role": "system", "content": "You reselect better knowledge graph relations for a single failed reasoning step. Output <analysis> and <selected>."},
        {"role": "user", "content": prompt},
    ], max_tokens=400)

    selected = set()
    sel_text = extract_xml_tag(raw, "selected") or ""
    for m in re.finditer(r"\d+", sel_text):
        n = int(m.group())
        if 1 <= n <= len(cands):
            idx = cands[n - 1][0]
            selected.add(idx)

    if not selected:
        for idx, _, _ in cands:
            if idx not in current_indices:
                selected.add(idx)
            if len(selected) >= 3:
                break
    if not selected:
        selected = set(idx for idx, _, _ in cands[:3])
    return selected


def parse_decomposition(raw: str) -> Tuple[Optional[str], Optional[str], List[Dict[str, Any]]]:
    """Parse LLM decomposition: extract anchor name, entity_query, and structured steps."""
    anchor_name = None
    anchor_entity_query = None
    steps = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        # Extract anchor line: Anchor: name (entity_query: ...)
        m_anchor = re.match(r"^Anchor:\s*(.+?)\s*\(entity_query:\s*(.+?)\)\s*$", line, re.IGNORECASE)
        if m_anchor:
            anchor_name = m_anchor.group(1).strip()
            anchor_entity_query = m_anchor.group(2).strip()
            continue
        # Fallback: anchor without entity_query
        m_anchor2 = re.match(r"^Anchor:\s*(.+)$", line, re.IGNORECASE)
        if m_anchor2 and not anchor_name:
            anchor_name = m_anchor2.group(1).strip()
            anchor_entity_query = anchor_name
            continue
        # Extract step lines: 1. question (relation_query: ...; endpoint: ... or none)
        m = re.match(r"^(\d+)\.\s+(.+?)\s*\(relation_query:\s*(.+?);\s*endpoint:\s*(.+?)\)\s*$", line)
        if not m:
            continue
        endpoint_raw = m.group(4).strip()
        # Parse endpoint: "entity name (entity_query: ...)" or "none"
        endpoint = None
        endpoint_query = None
        if endpoint_raw.lower() != "none":
            m_ep = re.match(r"(.+?)\s*\(entity_query:\s*(.+?)\)", endpoint_raw)
            if m_ep:
                endpoint = m_ep.group(1).strip()
                endpoint_query = m_ep.group(2).strip()
            else:
                endpoint = endpoint_raw
                endpoint_query = endpoint_raw
        steps.append({
            "step": int(m.group(1)),
            "question": m.group(2).strip(),
            "relation_query": m.group(3).strip(),
            "endpoint": endpoint,
            "endpoint_query": endpoint_query,
        })
    return anchor_name, anchor_entity_query, steps


def expand_node(node_idx, rel_indices, h_ids, r_ids, t_ids, reverse=False):
    children = []
    for i in range(len(h_ids)):
        if reverse:
            if t_ids[i] == node_idx and r_ids[i] in rel_indices:
                children.append((h_ids[i], r_ids[i]))
        else:
            if h_ids[i] == node_idx and r_ids[i] in rel_indices:
                children.append((t_ids[i], r_ids[i]))
    return children


def expand_through_cvt(node_idx, h_ids, r_ids, t_ids, entity_list):
    name = entity_list[node_idx] if 0 <= node_idx < len(entity_list) else ""
    if not is_cvt_like(name):
        return []
    children, seen = [], set()
    for i in range(len(h_ids)):
        if h_ids[i] == node_idx and t_ids[i] not in seen:
            children.append((t_ids[i], r_ids[i])); seen.add(t_ids[i])
        if t_ids[i] == node_idx and h_ids[i] not in seen:
            children.append((h_ids[i], r_ids[i])); seen.add(h_ids[i])
    return children


def chain_expand(anchor_idx, step_relations, h_ids, r_ids, t_ids, entity_list):
    """Ordered chain expansion with forward validation: step 1 → step 2 → ...
    After expanding step K, validate that resulting nodes have step K+1 edges.
    Prunes structurally incomplete paths — if a tail node can't continue the
    chain, the triple is removed. Last step has no constraint (all results kept).
    Returns ALL paths, not just deepest.
    """
    paths = [{"nodes": [anchor_idx], "relations": [], "depth": 0}]

    for step_idx, rel_indices in enumerate(step_relations):
        if not rel_indices:
            continue
        # Look ahead: find next non-empty step for forward validation
        next_rel_indices = set()
        for nxt in range(step_idx + 1, len(step_relations)):
            if step_relations[nxt]:
                next_rel_indices = step_relations[nxt]
                break

        new_paths = []
        for path in paths:
            current = path["nodes"][-1]
            fwd = expand_node(current, rel_indices, h_ids, r_ids, t_ids)
            rev = expand_node(current, rel_indices, h_ids, r_ids, t_ids, reverse=True)
            all_children = fwd + rev
            if all_children:
                seen = set(path["nodes"])
                for child_idx, rel_idx in all_children:
                    if child_idx in seen:
                        continue
                    # Forward validation: skip child if it can't continue the chain
                    if next_rel_indices and not _has_step_edges(
                            child_idx, next_rel_indices, h_ids, r_ids, t_ids, entity_list):
                        continue
                    new_path = {"nodes": path["nodes"] + [child_idx], "relations": path["relations"] + [rel_idx], "depth": path["depth"] + 1}
                    child_name = entity_list[child_idx] if 0 <= child_idx < len(entity_list) else ""
                    if is_cvt_like(child_name):
                        cvt_children = expand_through_cvt(child_idx, h_ids, r_ids, t_ids, entity_list)
                        new_seen = set(new_path["nodes"])
                        for cvt_idx, cvt_rel in cvt_children:
                            if cvt_idx in new_seen:
                                continue
                            new_paths.append({"nodes": new_path["nodes"] + [cvt_idx], "relations": new_path["relations"] + [cvt_rel], "depth": new_path["depth"] + 1})
                    else:
                        new_paths.append(new_path)
            else:
                # Dead end: keep path as-is for this step
                new_paths.append(path)
        paths = new_paths if new_paths else paths

    if not paths:
        return [], 0
    max_depth = max(p["depth"] for p in paths)
    return paths, max_depth


def _has_step_edges(node_idx, rel_indices, h_ids, r_ids, t_ids, entity_list):
    """Check if node has edges matching rel_indices within 2 hops.
    Direct edges first, then through CVT intermediary:
      node → [any rel] → CVT → [target rel]
    This handles cases like child_labor_percent where the relation
    bridges through a CVT node rather than connecting directly.
    """
    # Direct check
    fwd = expand_node(node_idx, rel_indices, h_ids, r_ids, t_ids)
    rev = expand_node(node_idx, rel_indices, h_ids, r_ids, t_ids, reverse=True)
    if fwd or rev:
        return True
    # 2-hop: node → CVT → [target rel]
    seen = {node_idx}
    for i in range(len(h_ids)):
        if h_ids[i] == node_idx:
            neighbor = t_ids[i]
        elif t_ids[i] == node_idx:
            neighbor = h_ids[i]
        else:
            continue
        if neighbor in seen:
            continue
        neighbor_name = entity_list[neighbor] if 0 <= neighbor < len(entity_list) else ""
        if not is_cvt_like(neighbor_name):
            continue
        seen.add(neighbor)
        fwd2 = expand_node(neighbor, rel_indices, h_ids, r_ids, t_ids)
        rev2 = expand_node(neighbor, rel_indices, h_ids, r_ids, t_ids, reverse=True)
        if fwd2 or rev2:
            return True
    return False




def chain_expand_v2(anchor_idx, step_relations, h_ids, r_ids, t_ids, entity_list,
                    max_hops=4, beam_width=80, per_branch_width=5):
    """Relation-anchored multi-hop expansion: anchor on relations, not hop count.

    For step_relations = [R1_set, R2_set, ..., Rn_set]:
    1. MAIN: Search all paths within max_hops where the LAST hop's relation is in Rn_set.
    2. BACKTRACK: Check if earlier relations (R1..Rn-1) appear in order in the path.
    3. SCORE: Full match (all relations in order) > partial match > single relation.
    4. FALLBACK: If no Rn paths found, try Rn-1, then Rn-2, ..., down to R1.

    CVT nodes pass through without counting toward hop limit.

    Returns (paths, max_depth, max_coverage_tier) where paths are dicts with keys:
      nodes, relations, depth, covered_steps, matched_relations, coverage_tier
    """
    n_steps = len(step_relations)
    if n_steps == 0:
        return [], 0, 0

    # Build adjacency list (undirected)
    adj = {}
    for i in range(len(h_ids)):
        h, r, t = h_ids[i], r_ids[i], t_ids[i]
        adj.setdefault(h, []).append((t, r))
        adj.setdefault(t, []).append((h, r))

    # Build rel_to_step mapping
    rel_to_step = {}
    all_target_rels = set()
    for step_idx, rel_set in enumerate(step_relations):
        for rel in rel_set:
            rel_to_step.setdefault(rel, set()).add(step_idx)
            all_target_rels.add(rel)

    # Coverage tier computation:
    # Given a path's matched steps, compute tier score.
    # For [R1, R2]: R1+R2=3, R2_only=2, R1_only=1
    # For [R1, R2, R3]: R1+R2+R3=7, R1+R3=5, R2+R3=6, R3_only=4, R1+R2=3, R2_only=2, R1_only=1
    # General: priority by last-step presence, then by earlier steps, ordered correctly.
    def compute_tier(matched_steps_set):
        """Higher is better. Must include the target step for main strategy."""
        if not matched_steps_set:
            return 0
        # Score: each matched step contributes 2^(position). Order matters implicitly
        # because we check subset ordering separately.
        score = 0
        for s in matched_steps_set:
            score += (1 << s)
        return score

    def check_order(path_rels, target_rels_ordered):
        """Check which target relation sets appear in the path in correct order.

        Returns set of step indices that are matched in order.
        """
        if not path_rels:
            return set()

        matched = set()
        search_start = 0
        for step_idx, rel_set in enumerate(target_rels_ordered):
            if not rel_set:
                continue
            # Find first occurrence of any rel in rel_set at or after search_start
            for pos in range(search_start, len(path_rels)):
                if path_rels[pos] in rel_set:
                    matched.add(step_idx)
                    search_start = pos + 1
                    break
        return matched

    def _beam_prune(paths, limit):
        if len(paths) <= limit:
            return paths
        # Group by branch signature (last node + covered steps pattern) to maintain diversity
        groups = {}
        for p in paths:
            sig = (p["nodes"][-1], tuple(sorted(p.get("covered_steps", set()))))
            groups.setdefault(sig, []).append(p)
        result = []
        overflow = []
        for sig, group in groups.items():
            group.sort(key=lambda x: (x.get("coverage_tier", 0), -x.get("depth", 0)), reverse=True)
            take = min(len(group), per_branch_width)
            result.extend(group[:take])
            overflow.extend(group[take:])
        if len(result) < limit and overflow:
            overflow.sort(key=lambda x: (x.get("coverage_tier", 0), -x.get("depth", 0)), reverse=True)
            result.extend(overflow[:limit - len(result)])
        if len(result) > limit:
            result.sort(key=lambda x: (x.get("coverage_tier", 0), -x.get("depth", 0)), reverse=True)
            result = result[:limit]
        return result

    def _search_from_anchor(target_step_indices):
        """BFS/DFS hybrid from anchor, looking for paths ending with target step relations.

        target_step_indices: list of step indices whose relations are acceptable as the last hop.
        Returns list of path dicts.
        """
        # Collect target relations for this search
        target_rels = set()
        for si in target_step_indices:
            target_rels |= step_relations[si]

        if not target_rels:
            return []

        # BFS with beam
        # State: list of path dicts
        initial_path = {
            "nodes": [anchor_idx],
            "relations": [],
            "depth": 0,
            "real_hops": 0,  # hops excluding CVT passthrough
            "covered_steps": set(),
            "matched_relations": set(),
            "coverage_tier": 0,
            "last_hop_is_target": False,
        }
        active = [initial_path]
        completed = []  # paths that ended with a target relation

        for _ in range(max_hops * 3):  # enough iterations for CVT passthroughs
            if not active:
                break

            new_active = []
            for path in active:
                current = path["nodes"][-1]
                current_name = entity_list[current] if 0 <= current < len(entity_list) else ""
                is_at_cvt = is_cvt_like(current_name)

                all_neighbors = adj.get(current, [])
                if not all_neighbors:
                    continue

                seen = set(path["nodes"])

                # 3-tier edge filtering: pool relations first, non-pool strictly limited
                target_edges = []   # matches current search target (e.g. R_n)
                pool_edges = []     # in any step's relation set
                other_edges = []    # not in any step — noise
                for neighbor, rel in all_neighbors:
                    if neighbor in seen:
                        continue
                    if rel in target_rels:
                        target_edges.append((neighbor, rel))
                    elif rel in all_target_rels:
                        pool_edges.append((neighbor, rel))
                    else:
                        other_edges.append((neighbor, rel))

                # Priority: target > pool > limited fallback
                if target_edges or pool_edges:
                    edges = target_edges + pool_edges
                else:
                    edges = other_edges[:5]

                for neighbor, rel in edges:
                    new_nodes = path["nodes"] + [neighbor]
                    new_rels = path["relations"] + [rel]
                    new_real_hops = path["real_hops"]

                    # CVT passthrough: don't count toward hop limit
                    neighbor_name = entity_list[neighbor] if 0 <= neighbor < len(entity_list) else ""
                    if not is_cvt_like(current_name) or not is_at_cvt:
                        new_real_hops += 1

                    # Check hop limit
                    if new_real_hops > max_hops:
                        continue

                    # Check if this hop uses a target relation
                    rel_step_matches = rel_to_step.get(rel, set())
                    is_target_hop = rel in target_rels

                    new_covered = set(path["covered_steps"])
                    new_matched = set(path["matched_relations"])
                    if rel_step_matches:
                        new_covered |= rel_step_matches
                        new_matched.add(rel)

                    new_path = {
                        "nodes": new_nodes,
                        "relations": new_rels,
                        "depth": path["depth"] + 1,
                        "real_hops": new_real_hops,
                        "covered_steps": new_covered,
                        "matched_relations": new_matched,
                        "coverage_tier": 0,
                        "last_hop_is_target": is_target_hop,
                    }

                    if is_target_hop:
                        # This path ends with a target relation — compute coverage
                        ordered_match = check_order(new_rels, step_relations)
                        new_path["covered_steps"] = ordered_match
                        new_path["coverage_tier"] = compute_tier(ordered_match)
                        completed.append(new_path)
                        # Continue expanding from here too (might find longer matches)
                        if new_real_hops < max_hops:
                            new_active.append(new_path)
                    else:
                        # Not a target hop, keep searching
                        new_active.append(new_path)

            # Beam prune active paths
            active = _beam_prune(new_active, beam_width)

        # Cap completed paths: sort by coverage tier, keep top beam_width
        if len(completed) > beam_width:
            completed.sort(key=lambda p: (p.get("coverage_tier", 0), -p.get("depth", 0)), reverse=True)
            completed = completed[:beam_width]
        return completed

    # ---- Main strategy: search for paths ending with R_n ----
    result_paths = []

    # Try from last step backwards (fallback cascade)
    for target_depth in range(n_steps - 1, -1, -1):
        target_indices = [target_depth]
        found = _search_from_anchor(target_indices)

        if found:
            # Deduplicate by (nodes tuple, relations tuple)
            seen_keys = set()
            for p in result_paths:
                seen_keys.add((tuple(p["nodes"]), tuple(p["relations"])))

            for p in found:
                key = (tuple(p["nodes"]), tuple(p["relations"]))
                if key not in seen_keys:
                    seen_keys.add(key)
                    result_paths.append(p)

            # If we found paths for the deepest target step, we can still look for earlier ones
            # but only if the main target was the last step (not a fallback)
            if target_depth == n_steps - 1:
                # Also search for earlier steps as supplementary (these are lower tier)
                for supplementary_depth in range(n_steps - 2, -1, -1):
                    supp_found = _search_from_anchor([supplementary_depth])
                    for p in supp_found:
                        key = (tuple(p["nodes"]), tuple(p["relations"]))
                        if key not in seen_keys:
                            seen_keys.add(key)
                            result_paths.append(p)
                break  # Main strategy succeeded, don't fallback further
        # If target_depth < n_steps-1, this is a fallback — accept and stop

    if not result_paths:
        return [], 0, 0

    # Sort by coverage tier (desc), then depth (desc for deeper=more info), then fewer nodes
    result_paths.sort(key=lambda p: (p.get("coverage_tier", 0), -p.get("depth", 0)), reverse=True)

    # Final beam prune on total results
    if len(result_paths) > beam_width * 2:
        result_paths = _beam_prune(result_paths, beam_width * 2)

    max_depth = max(p.get("depth", 0) for p in result_paths)
    max_tier = max(p.get("coverage_tier", 0) for p in result_paths)

    return result_paths, max_depth, max_tier


def _coverage_rank(path):
    """Causal tier ranking: layer coverage count > max layer hit > shorter bridge."""
    covered = path.get("covered_steps", frozenset())
    depth = path.get("depth", 0)
    return score_causal_tier(covered, bridge_length=depth)


def _merge_paths(fwd_path, bwd_path, entity_list):
    fwd_nodes = fwd_path["nodes"]
    bwd_nodes = bwd_path["nodes"]
    meeting_node = fwd_nodes[-1]
    if meeting_node != bwd_nodes[-1]:
        return None
    if set(fwd_nodes[:-1]) & set(bwd_nodes[:-1]):
        return None
    return {
        "nodes": fwd_nodes + list(reversed(bwd_nodes[:-1])),
        "relations": fwd_path["relations"] + list(reversed(bwd_path["relations"])),
        "depth": fwd_path["depth"] + bwd_path["depth"],
        "covered_steps": frozenset(set(fwd_path.get("covered_steps", frozenset())) | set(bwd_path.get("covered_steps", frozenset()))),
        "matched_relations": frozenset(set(fwd_path.get("matched_relations", frozenset())) | set(bwd_path.get("matched_relations", frozenset()))),
    }



def bidirectional_expand(anchor_idx, target_idx, step_relations, h_ids, r_ids, t_ids, entity_list,
                         max_hops=5, beam_width=50, per_branch_width=6):
    """Relation-prior-guided bidirectional BFS with step-aware scoring.

    Forward from anchor: preferentially expands via step-aligned relations (step 0 at hop 0, etc.)
    Backward from endpoint: preferentially expands via reversed step relations (step N-1 at hop 0, etc.)
    Three-tier edge priority: guided (step-aligned) > in-pool (other steps) > fallback (all edges).
    Post-hoc ranking by step coverage count.
    """
    n_steps = len(step_relations)
    # Build rel_to_steps mapping and pooled relation set
    rel_to_steps: Dict[int, set] = {}
    relation_pool: set = set()
    for step_idx, rel_set in enumerate(step_relations):
        for rel in rel_set:
            rel_to_steps.setdefault(rel, set()).add(step_idx)
            relation_pool.add(rel)

    # Build undirected adjacency
    adj: Dict[int, List[tuple]] = {}
    for i in range(len(h_ids)):
        h, r, t = h_ids[i], r_ids[i], t_ids[i]
        adj.setdefault(h, []).append((t, r))
        adj.setdefault(t, []).append((h, r))

    def _make_path(nodes, relations, depth, covered=frozenset(), matched=frozenset()):
        return {"nodes": nodes, "relations": relations, "depth": depth,
                "covered_steps": covered, "matched_relations": matched}

    def _extend_path(path, neighbor, rel):
        r_steps = rel_to_steps.get(rel, set())
        new_covered = frozenset(set(path["covered_steps"]) | r_steps)
        new_matched = frozenset(set(path["matched_relations"]) | ({rel} if r_steps else set()))
        return _make_path(
            path["nodes"] + [neighbor], path["relations"] + [rel], path["depth"] + 1,
            new_covered, new_matched)

    def _branch_sig(path):
        """Branch signature: (step_idx, first_matched_rel) per covered step + last node."""
        covered = path.get("covered_steps", frozenset())
        sig = []
        for step_idx in sorted(covered):
            for r in sorted(path.get("matched_relations", frozenset())):
                if step_idx in rel_to_steps.get(r, set()):
                    sig.append((step_idx, r))
                    break
        sig.append(path["nodes"][-1])
        return tuple(sig)

    def _beam_prune(paths, limit):
        if len(paths) <= limit:
            return paths
        groups: Dict[tuple, list] = {}
        for p in paths:
            sig = _branch_sig(p)
            groups.setdefault(sig, []).append(p)
        result = []
        remaining = []
        for sig, group in groups.items():
            group.sort(key=_coverage_rank, reverse=True)
            result.extend(group[:per_branch_width])
            remaining.extend(group[per_branch_width:])
        if len(result) < limit and remaining:
            remaining.sort(key=_coverage_rank, reverse=True)
            result.extend(remaining[:limit - len(result)])
        if len(result) > limit:
            result.sort(key=_coverage_rank, reverse=True)
            result = result[:limit]
        return result

    def _get_edges(node_idx, hop, is_forward):
        """Get edges for expansion with 3-tier priority: guided > in-pool > fallback."""
        all_edges = adj.get(node_idx, [])
        if not all_edges:
            return []
        # Determine expected step for this hop
        if is_forward:
            expected_step = hop % n_steps if n_steps > 0 else -1
        else:
            expected_step = (n_steps - 1 - hop % n_steps) if n_steps > 0 else -1

        expected_rels = step_relations[expected_step] if 0 <= expected_step < n_steps else set()

        guided = []    # Matches expected step relation
        in_pool = []   # In some step but not expected
        fallback = []  # Not in any step

        for neighbor, rel in all_edges:
            if rel in expected_rels:
                guided.append((neighbor, rel))
            elif rel in relation_pool:
                in_pool.append((neighbor, rel))
            else:
                fallback.append((neighbor, rel))

        # Priority: guided first, then in-pool (limited), then fallback (very limited)
        if guided:
            return guided + in_pool[:5]
        elif in_pool:
            return in_pool + fallback[:5]
        else:
            return fallback[:10]

    # Initialize frontiers
    init_fwd = _make_path([anchor_idx], [], 0)
    init_bwd = _make_path([target_idx], [], 0)
    fwd_frontier: Dict[int, List[Dict]] = {anchor_idx: [init_fwd]}
    bwd_frontier: Dict[int, List[Dict]] = {target_idx: [init_bwd]}
    fwd_path_map: Dict[int, List[Dict]] = {anchor_idx: [init_fwd]}
    bwd_path_map: Dict[int, List[Dict]] = {target_idx: [init_bwd]}
    fwd_visited: Dict[int, int] = {anchor_idx: 0}
    bwd_visited: Dict[int, int] = {target_idx: 0}
    meeting_paths = []

    for hop in range(max_hops):
        if len(fwd_frontier) <= len(bwd_frontier):
            # Forward expansion with guided edges
            new_frontier: Dict[int, List[Dict]] = {}
            for node_idx, paths in fwd_frontier.items():
                edges = _get_edges(node_idx, hop, is_forward=True)
                for neighbor, rel in edges:
                    if neighbor in fwd_visited and fwd_visited[neighbor] < hop + 1:
                        continue
                    fwd_visited.setdefault(neighbor, hop + 1)
                    for path in paths:
                        if neighbor in set(path["nodes"]):
                            continue
                        new_path = _extend_path(path, neighbor, rel)
                        if neighbor in bwd_visited:
                            for bwd_path in bwd_path_map.get(neighbor, []):
                                merged = _merge_paths(new_path, bwd_path, entity_list)
                                if merged:
                                    meeting_paths.append(merged)
                        new_frontier.setdefault(neighbor, []).append(new_path)
            for node in new_frontier:
                new_frontier[node] = _beam_prune(new_frontier[node], per_branch_width * 3)
            fwd_frontier = new_frontier
            for node, node_paths in new_frontier.items():
                fwd_path_map.setdefault(node, []).extend(node_paths)
                fwd_path_map[node] = _beam_prune(fwd_path_map[node], beam_width)
        else:
            # Backward expansion with guided edges
            new_frontier: Dict[int, List[Dict]] = {}
            for node_idx, paths in bwd_frontier.items():
                edges = _get_edges(node_idx, hop, is_forward=False)
                for neighbor, rel in edges:
                    if neighbor in bwd_visited and bwd_visited[neighbor] < hop + 1:
                        continue
                    bwd_visited.setdefault(neighbor, hop + 1)
                    for path in paths:
                        if neighbor in set(path["nodes"]):
                            continue
                        new_path = _extend_path(path, neighbor, rel)
                        if neighbor in fwd_visited:
                            for fwd_path in fwd_path_map.get(neighbor, []):
                                merged = _merge_paths(fwd_path, new_path, entity_list)
                                if merged:
                                    meeting_paths.append(merged)
                        new_frontier.setdefault(neighbor, []).append(new_path)
            for node in new_frontier:
                new_frontier[node] = _beam_prune(new_frontier[node], per_branch_width * 3)
            bwd_frontier = new_frontier
            for node, node_paths in new_frontier.items():
                bwd_path_map.setdefault(node, []).extend(node_paths)
                bwd_path_map[node] = _beam_prune(bwd_path_map[node], beam_width)

    if not meeting_paths:
        return [], 0, 0

    # Deduplicate and rank
    seen = set()
    unique = []
    for p in meeting_paths:
        key = (tuple(p["nodes"]), tuple(p["relations"]))
        if key not in seen:
            seen.add(key)
            unique.append(p)

    unique.sort(key=_coverage_rank, reverse=True)
    max_cov = len(unique[0].get("covered_steps", frozenset())) if unique else 0
    max_depth = max(p.get("depth", 0) for p in unique)

    return unique, max_depth, max_cov



class GraphBackend:
    """graph_tool-backed adjacency structure for fast neighbor lookups.

    Wraps a graph_tool.Graph with:
    - Undirected adjacency (both h->t and t->h traversals)
    - relation_id stored as an edge property
    - Pre-computed CVT vertex property
    - Cached neighbor lookups returning (neighbor_idx, relation_idx) tuples
    """

    def __init__(self, h_ids, r_ids, t_ids, entity_list):
        self.n_ents = len(entity_list)
        self._is_cvt = [is_cvt_like(name) for name in entity_list]

        # Build undirected graph_tool graph
        self._g = gt.Graph(directed=False)
        self._g.add_vertex(self.n_ents)

        # Edge property for relation id
        ep_rel = self._g.new_edge_property("int")
        self._g.edge_properties["relation_id"] = ep_rel

        # Add edges
        for i in range(len(h_ids)):
            e = self._g.add_edge(h_ids[i], t_ids[i])
            ep_rel[e] = r_ids[i]

        self._ep_rel = ep_rel
        self._neighbor_cache: Dict[int, tuple] = {}

    def get_neighbors(self, node_idx: int) -> tuple:
        """Return cached tuple of (neighbor_idx, relation_idx) pairs for a node."""
        cached = self._neighbor_cache.get(node_idx)
        if cached is not None:
            return cached
        v = self._g.vertex(node_idx)
        neighbors = []
        for e in v.out_edges():
            s = int(e.source())
            t = int(e.target())
            neighbor = t if s == node_idx else s
            neighbors.append((neighbor, int(self._ep_rel[e])))
        result = tuple(neighbors)
        self._neighbor_cache[node_idx] = result
        return result

    def is_cvt_node(self, idx: int) -> bool:
        return self._is_cvt[idx]

    def real_hop_inc(self, curr_idx: int, next_idx: int) -> int:
        """0 if either endpoint is CVT, else 1."""
        if curr_idx < self.n_ents and next_idx < self.n_ents:
            if self._is_cvt[curr_idx] or self._is_cvt[next_idx]:
                return 0
        return 1


def relation_prior_expand(anchor_idx, step_relations, h_ids, r_ids, t_ids, entity_list,
                          explicit_targets=None, max_hops=3, beam_width=80, per_branch_width=5):
    """Forward layer-by-layer relation-prior expansion.

    New behavior:
    1. Start from current entity frontier (initially the anchor).
    2. For layer i, search all paths within max_hops whose LAST hop relation is in R_i.
    3. Use the endpoints of those matched paths as the start frontier for the next layer.
    4. If a layer has no hit, skip it and continue from the current frontier.
    5. If explicit endpoint targets exist, connect the final frontier to those targets
       via a shortest path search within max_hops.

    This removes the backward-target template and avoids the repeated-relation
    penetration issue seen in bidirectional matching such as r1 -> r1 collapse.

    Performance optimizations (v2):
    - Paths stored as tuples (nodes, rels, depth, real_hops, covered, matched)
      instead of dicts, avoiding dict creation overhead in the hot inner loop.
    - CVT status pre-computed once as a boolean list.
    - Adjacency neighbor lists stored as tuples for faster iteration.
    - BFS in _connect_to_targets uses collections.deque.
    - Reduced frozenset churn: only create new frozensets when coverage changes.
    - _prune_paths uses frozenset directly as hash key instead of sorted tuple.
    """
    n_steps = len(step_relations)
    if n_steps == 0:
        return [], 0, 0

    # ── Build adjacency ──────────────────────────────────────
    # ── Graph backend or pure-Python adjacency ───────────────
    use_gt = _HAS_GT
    if use_gt:
        backend = GraphBackend(h_ids, r_ids, t_ids, entity_list)
        adj = None
        adj_empty = ()
        is_cvt = backend._is_cvt
        n_ents = backend.n_ents
    else:
        adj: Dict[int, tuple] = {}
        for i in range(len(h_ids)):
            h, r, t = h_ids[i], r_ids[i], t_ids[i]
            if h in adj:
                adj[h] = adj[h] + ((t, r),)
            else:
                adj[h] = ((t, r),)
            if t in adj:
                adj[t] = adj[t] + ((h, r),)
            else:
                adj[t] = ((h, r),)
        adj_empty = ()
        is_cvt = [is_cvt_like(name) for name in entity_list]
        n_ents = len(entity_list)

    # ── Build reverse mapping ────────────────────────────────
    rel_to_step: Dict[int, set] = {}
    for si, rs in enumerate(step_relations):
        for r in rs:
            rel_to_step.setdefault(r, set()).add(si)
    all_layer_rels = set(rel_to_step.keys())

    # ── Helpers ───────────────────────────────────────────────
    def _get_neighbors(node_idx):
        """Dispatch to graph_tool backend or pure-Python adjacency."""
        if use_gt:
            return backend.get_neighbors(node_idx)
        return adj.get(node_idx, adj_empty)

    def _real_hop_inc(curr_idx, next_idx):
        """0 if either endpoint is CVT, else 1."""
        if use_gt:
            return backend.real_hop_inc(curr_idx, next_idx)
        if curr_idx < n_ents and next_idx < n_ents:
            if is_cvt[curr_idx] or is_cvt[next_idx]:
                return 0
        return 1

    def _coverage_rank_fast(path):
        """Path is tuple: (nodes, rels, depth, real_hops, covered_steps, matched_rels)."""
        covered = path[4]
        depth = path[2]
        if not covered:
            return (0, -1, 0)
        return (len(covered), max(covered), -depth)

    def _prune_paths(paths, limit):
        if len(paths) <= limit:
            return paths
        # Group by (endpoint, covered_steps) — frozenset is directly hashable
        grouped: Dict[tuple, list] = {}
        for p in paths:
            sig = (p[0][-1], p[4])  # (nodes[-1], covered_steps)
            if sig in grouped:
                grouped[sig].append(p)
            else:
                grouped[sig] = [p]
        result = []
        overflow = []
        for group in grouped.values():
            group.sort(key=_coverage_rank_fast, reverse=True)
            result.extend(group[:per_branch_width])
            overflow.extend(group[per_branch_width:])
        if len(result) < limit and overflow:
            overflow.sort(key=_coverage_rank_fast, reverse=True)
            result.extend(overflow[: limit - len(result)])
        if len(result) > limit:
            result.sort(key=_coverage_rank_fast, reverse=True)
            result = result[:limit]
        return result

    def _search_terminal_relation_paths(start_paths, target_rels, layer_idx):
        """From start_paths, search local segments that terminate at the FIRST hit of target_rels.

        Rules:
        - bridge hops may not use any selected layer relation
        - once a target relation is hit, the segment ends immediately
        - same-layer relations cannot chain within one segment
        """
        if not target_rels:
            return []
        active = list(start_paths)
        matched = []
        seen_matched = set()
        layer_idx_frozen = frozenset({layer_idx})

        for _ in range(max_hops * 3):
            if not active:
                break
            new_active = []
            for path in active:
                nodes, rels, depth, real_hops, covered, matched_rels = path
                current = nodes[-1]
                neighbors = _get_neighbors(current)
                for neighbor, rel in neighbors:
                    if neighbor in nodes:
                        continue
                    inc = _real_hop_inc(current, neighbor)
                    new_real_hops = real_hops + inc
                    if new_real_hops > max_hops:
                        continue

                    is_target_rel = rel in target_rels
                    is_any_layer_rel = rel in all_layer_rels

                    if not is_target_rel and is_any_layer_rel:
                        continue

                    new_nodes = nodes + (neighbor,)
                    new_rels = rels + (rel,)
                    new_depth = depth + 1

                    rel_steps = rel_to_step.get(rel)
                    if rel_steps:
                        new_covered = covered | rel_steps
                        new_matched = matched_rels | frozenset({rel})
                    else:
                        new_covered = covered
                        new_matched = matched_rels

                    new_path = (new_nodes, new_rels, new_depth, new_real_hops, new_covered, new_matched)

                    if is_target_rel:
                        final_covered = new_covered | layer_idx_frozen
                        final_path = (new_nodes, new_rels, new_depth, new_real_hops, final_covered, new_matched)
                        key = (new_nodes, new_rels)
                        if key not in seen_matched:
                            seen_matched.add(key)
                            matched.append(final_path)
                    else:
                        new_active.append(new_path)
            active = _prune_paths(new_active, beam_width)
        return _prune_paths(matched, beam_width)

    def _connect_to_targets(paths, targets):
        """Attach final frontier endpoints to explicit targets by shortest unconstrained path."""
        if not targets or not paths:
            return paths
        targets_set = set(targets) - {anchor_idx, None}
        if not targets_set:
            return paths

        connected = []
        seen = set()
        for base in paths:
            start = base[0][-1]  # nodes[-1]
            queue = deque([(start, (start,), (), 0)])
            local_seen = {(start, 0)}
            best = []
            while queue:
                node, nodes_seq, rel_seq, rhops = queue.popleft()
                if node in targets_set and node != start:
                    merged = (
                        base[0] + nodes_seq[1:],
                        base[1] + rel_seq,
                        base[2] + len(rel_seq),
                        base[3] + rhops,
                        base[4],
                        base[5],
                    )
                    best.append(merged)
                    continue
                for neighbor, rel in _get_neighbors(node):
                    if neighbor in nodes_seq:
                        continue
                    inc = _real_hop_inc(node, neighbor)
                    new_hops = rhops + inc
                    if new_hops > max_hops:
                        continue
                    state_key = (neighbor, new_hops)
                    if state_key in local_seen:
                        continue
                    local_seen.add(state_key)
                    queue.append((neighbor, nodes_seq + (neighbor,), rel_seq + (rel,), new_hops))
            best.sort(key=_coverage_rank_fast, reverse=True)
            for p in best[:per_branch_width]:
                key = (p[0], p[1])
                if key not in seen:
                    seen.add(key)
                    connected.append(p)
        return _prune_paths(connected, beam_width) if connected else paths

    # ── Main expansion logic ─────────────────────────────────
    # Internal path format: (nodes_tuple, rels_tuple, depth, real_hops, covered_steps, matched_rels)
    frontier_paths = [((anchor_idx,), (), 0, 0, frozenset(), frozenset())]
    all_result_paths = []
    matched_layer_indices = []
    nonempty_layers = [i for i, rs in enumerate(step_relations) if rs]

    for layer_idx, target_rels in enumerate(step_relations):
        if not target_rels:
            continue
        matched = _search_terminal_relation_paths(frontier_paths, target_rels, layer_idx)
        if not matched:
            continue
        frontier_paths = matched
        all_result_paths = matched
        matched_layer_indices.append(layer_idx)

    # Minimal repair: only repair the final non-empty layer if it was missed.
    if nonempty_layers and matched_layer_indices:
        last_nonempty_idx = nonempty_layers[-1]
        if last_nonempty_idx not in matched_layer_indices and frontier_paths:
            repaired = _search_terminal_relation_paths(frontier_paths, step_relations[last_nonempty_idx], last_nonempty_idx)
            if repaired:
                merged = list(all_result_paths or []) + repaired
                merged.sort(key=_coverage_rank_fast, reverse=True)
                all_result_paths = _prune_paths(merged, beam_width)

    if explicit_targets:
        all_result_paths = _connect_to_targets(all_result_paths or frontier_paths, explicit_targets)
    elif not all_result_paths:
        all_result_paths = frontier_paths

    if not all_result_paths:
        return [], 0, 0

    # Dedup
    dedup = []
    seen = set()
    for p in all_result_paths:
        key = (p[0], p[1])
        if key not in seen:
            seen.add(key)
            dedup.append(p)

    dedup.sort(key=_coverage_rank_fast, reverse=True)
    dedup = _prune_paths(dedup, beam_width)

    # Convert internal tuple format back to dict format for API compatibility
    result_dicts = []
    for p in dedup:
        result_dicts.append({
            "nodes": list(p[0]),
            "relations": list(p[1]),
            "depth": p[2],
            "real_hops": p[3],
            "covered_steps": p[4],
            "matched_relations": p[5],
        })

    max_cov = max((len(p[4]) for p in dedup), default=0)
    max_depth = max((p[2] for p in dedup), default=0)
    return result_dicts, max_depth, max_cov

def diagnose_layers(anchor_idx, step_relations, h_ids, r_ids, t_ids, entity_list,
                    max_hops=3, beam_width=80, per_branch_width=5):
    """Diagnose each layer without changing the main traversal implementation.

    For each layer Li:
    - frontier_hit: can Li be reached from the current sequential frontier?
    - anchor_hit: can Li be reached directly from the anchor?
    """
    if anchor_idx is None:
        return []

    adj: Dict[int, List[tuple]] = {}
    for i in range(len(h_ids)):
        h, r, t = h_ids[i], r_ids[i], t_ids[i]
        adj.setdefault(h, []).append((t, r))
        adj.setdefault(t, []).append((h, r))

    rel_to_step: Dict[int, set] = {}
    for si, rs in enumerate(step_relations):
        for r in rs:
            rel_to_step.setdefault(r, set()).add(si)

    def _make_path(nodes, relations, depth, real_hops=0, covered=frozenset(), matched=frozenset()):
        return {
            "nodes": nodes,
            "relations": relations,
            "depth": depth,
            "real_hops": real_hops,
            "covered_steps": covered,
            "matched_relations": matched,
        }

    def _real_hop_increment(curr_idx, next_idx):
        curr_name = entity_list[curr_idx] if 0 <= curr_idx < len(entity_list) else ""
        next_name = entity_list[next_idx] if 0 <= next_idx < len(entity_list) else ""
        if is_cvt_like(curr_name) or is_cvt_like(next_name):
            return 0
        return 1

    def _prune_paths(paths, limit):
        if len(paths) <= limit:
            return paths
        grouped = {}
        for p in paths:
            end = p["nodes"][-1]
            sig = (end, tuple(sorted(p.get("covered_steps", frozenset()))))
            grouped.setdefault(sig, []).append(p)
        result = []
        overflow = []
        for group in grouped.values():
            group.sort(key=_coverage_rank, reverse=True)
            result.extend(group[:per_branch_width])
            overflow.extend(group[per_branch_width:])
        if len(result) < limit and overflow:
            overflow.sort(key=_coverage_rank, reverse=True)
            result.extend(overflow[: limit - len(result)])
        if len(result) > limit:
            result.sort(key=_coverage_rank, reverse=True)
            result = result[:limit]
        return result

    def _search_terminal_relation_paths(start_paths, target_rels, layer_idx):
        if not target_rels:
            return []
        active = list(start_paths)
        matched = []
        seen = set()
        for _ in range(max_hops * 3):
            if not active:
                break
            new_active = []
            for path in active:
                current = path["nodes"][-1]
                for neighbor, rel in adj.get(current, []):
                    if neighbor in path["nodes"]:
                        continue
                    new_real_hops = path.get("real_hops", 0) + _real_hop_increment(current, neighbor)
                    if new_real_hops > max_hops:
                        continue
                    rel_steps = rel_to_step.get(rel, set())
                    new_path = _make_path(
                        path["nodes"] + [neighbor],
                        path["relations"] + [rel],
                        path["depth"] + 1,
                        real_hops=new_real_hops,
                        covered=frozenset(set(path.get("covered_steps", frozenset())) | rel_steps),
                        matched=frozenset(set(path.get("matched_relations", frozenset())) | ({rel} if rel_steps else set())),
                    )
                    if rel in target_rels:
                        final_path = dict(new_path)
                        final_path["covered_steps"] = frozenset(set(final_path["covered_steps"]) | {layer_idx})
                        key = (tuple(final_path["nodes"]), tuple(final_path["relations"]))
                        if key not in seen:
                            seen.add(key)
                            matched.append(final_path)
                    else:
                        new_active.append(new_path)
            active = _prune_paths(new_active, beam_width)
        return _prune_paths(matched, beam_width)

    frontier_paths = [_make_path([anchor_idx], [], 0)]
    anchor_paths = [_make_path([anchor_idx], [], 0)]
    diagnostics = []
    for layer_idx, target_rels in enumerate(step_relations):
        frontier_hit_paths = _search_terminal_relation_paths(frontier_paths, target_rels, layer_idx) if target_rels else []
        anchor_hit_paths = _search_terminal_relation_paths(anchor_paths, target_rels, layer_idx) if target_rels else []
        diagnostics.append({
            "layer_idx": layer_idx,
            "frontier_hit": bool(frontier_hit_paths),
            "anchor_hit": bool(anchor_hit_paths),
            "frontier_count": len(frontier_hit_paths),
            "anchor_count": len(anchor_hit_paths),
        })
        if frontier_hit_paths:
            frontier_paths = frontier_hit_paths
    return diagnostics


def compress_paths(paths, ents, rels_list, anchor_idx, breakpoint_indices):
    """Compress raw paths into logical patterns with role-aware node identification.

    Pattern format: Anchor --[rel_chain]--> <node> ... --> [Candidate] --> Endpoint
    - Anchor: starting entity name
    - <node>: abstracted bridge entities (not shown by name)
    - [Candidate]: potential answer entities
    - Endpoint: constraint entity (if exists)
    Merges paths with same relation chain pattern, collecting unique candidates.
    """
    if not paths or anchor_idx is None:
        return []

    anchor_name = ents[anchor_idx] if 0 <= anchor_idx < len(ents) else "?"
    patterns: Dict[tuple, dict] = {}

    def _raw_path_to_readable(path):
        """Render one witness path with abstract node labels.

        Keep relation order and path shape, but hide concrete intermediate entities.
        This lets the model judge semantic fit of the path rather than overfitting
        to visible candidate names.
        """
        nodes = path["nodes"]
        rels = path["relations"]
        if not nodes:
            return anchor_name
        # If the path already reaches a non-CVT entity and then only trails into
        # CVT/value nodes, hide that trailing CVT tail from the logical path.
        # We only keep a CVT tail visible when the core relation itself lands on CVT
        # (i.e. no intermediate non-CVT endpoint has already been reached).
        cutoff_len = len(nodes)
        non_cvt_positions = []
        for i, node_idx in enumerate(nodes):
            name = ents[node_idx] if 0 <= node_idx < len(ents) else ""
            if not is_cvt_like(name):
                non_cvt_positions.append(i)
        if len(non_cvt_positions) >= 2:
            last_non_cvt_pos = non_cvt_positions[-1]
            if last_non_cvt_pos < len(nodes) - 1:
                cutoff_len = last_non_cvt_pos + 1
        nodes = nodes[:cutoff_len]
        rels = rels[: max(0, cutoff_len - 1)]
        parts = []
        node_labels = {}
        next_label_id = 1
        for i, node_idx in enumerate(nodes):
            name = ents[node_idx] if 0 <= node_idx < len(ents) else "?"
            if i == 0:
                if node_idx == anchor_idx:
                    parts.append(anchor_name)
                else:
                    parts.append("node0")
                continue
            prev_rel_idx = rels[i - 1] if i - 1 < len(rels) else None
            rel_text = rel_to_text(rels_list[prev_rel_idx]) if prev_rel_idx is not None and prev_rel_idx < len(rels_list) else "?"
            if node_idx in node_labels:
                node_text = node_labels[node_idx]
            elif node_idx in breakpoint_indices and node_idx != anchor_idx:
                node_text = f"node{next_label_id}[endpoint]"
                node_labels[node_idx] = node_text
                next_label_id += 1
            elif is_cvt_like(name):
                node_text = f"node{next_label_id}[cvt]"
                node_labels[node_idx] = node_text
                next_label_id += 1
            else:
                node_text = f"node{next_label_id}"
                node_labels[node_idx] = node_text
                next_label_id += 1
            parts.append(f"--[{rel_text}]--> {node_text}")
        return " ".join(parts)

    for path in paths:
        # Extract non-CVT nodes with positions
        sig_nodes = []
        for i, node_idx in enumerate(path["nodes"]):
            name = ents[node_idx] if 0 <= node_idx < len(ents) else ""
            if not is_cvt_like(name):
                sig_nodes.append((i, node_idx, name))

        if len(sig_nodes) < 2:
            continue

        # Build relation chain between consecutive non-CVT nodes
        rel_chain = []
        for j in range(1, len(sig_nodes)):
            prev_pos = sig_nodes[j - 1][0]
            curr_pos = sig_nodes[j][0]
            rel_indices_between = path["relations"][prev_pos:curr_pos]
            rel_texts = [rel_to_text(rels_list[ri]) for ri in rel_indices_between if ri < len(rels_list)]
            rel_chain.append(" -> ".join(rel_texts) if rel_texts else "?")

        # Identify endpoint
        endpoint_name = None
        for _, node_idx, name in sig_nodes:
            if node_idx in breakpoint_indices and node_idx != anchor_idx:
                endpoint_name = name
                break

        # Identify candidates: ALL non-anchor, non-endpoint, non-CVT nodes
        # The answer is often a bridge entity being verified by the last relation,
        # not the last node before the endpoint.
        candidates = set()
        bp_set = breakpoint_indices if breakpoint_indices else set()
        for _, node_idx, name in sig_nodes:
            if node_idx != anchor_idx and node_idx not in bp_set and not is_cvt_like(name):
                candidates.add(name)

        # Pattern key: relation chain + endpoint
        key = (tuple(rel_chain), endpoint_name)

        # Compute causal tier for this raw path
        covered = path.get("covered_steps", frozenset())
        tier = score_causal_tier(covered, bridge_length=path.get("depth", 0))

        if key not in patterns:
            patterns[key] = {
                "rel_chain": rel_chain,
                "endpoint": endpoint_name,
                "candidates": set(),
                "raw_paths": [],
                "best_tier": (0, -1, 0),
                "best_raw_path": path,
                "best_depth": path.get("depth", 0),
            }
        patterns[key]["candidates"].update(candidates)
        if tier > patterns[key]["best_tier"]:
            patterns[key]["best_tier"] = tier
            patterns[key]["best_raw_path"] = path
            patterns[key]["best_depth"] = path.get("depth", 0)
        elif tier == patterns[key]["best_tier"] and path.get("depth", 0) < patterns[key].get("best_depth", 10**9):
            patterns[key]["best_raw_path"] = path
            patterns[key]["best_depth"] = path.get("depth", 0)
        if len(patterns[key]["raw_paths"]) < 50:
            patterns[key]["raw_paths"].append(path)

    # Convert to sorted list — rank only by the best causal tier of the pattern
    result = []
    for key, group in sorted(patterns.items(), key=lambda x: x[1]["best_tier"], reverse=True):
        cands = sorted(group["candidates"])[:20]
        # Use one witness path to preserve bridge structure for display.
        readable = _raw_path_to_readable(group["best_raw_path"])

        result.append({
            "rel_chain": group["rel_chain"],
            "endpoint": group["endpoint"],
            "candidates": cands,
            "best_tier": group["best_tier"],
            "readable": readable,
            "raw_paths": group["raw_paths"],
            "best_raw_path": group["best_raw_path"],
        })
    return result


def expand_to_triples(paths, ents, rels_list):
    """Expand raw paths to triples, bridging through CVT intermediate nodes.

    When a triple endpoint is a CVT node (m.xxx / g.xxx), we bridge forward
    through consecutive CVT nodes to the next real entity, merging relation
    names with " > ". This preserves connectivity while hiding gibberish.
    """
    triples = []
    seen = set()
    for path in paths:
        nodes = path["nodes"]
        rels = path["relations"]
        # Build raw (h_name, r_text, t_name) triples
        raw = []
        for i in range(min(len(rels), len(nodes) - 1)):
            h_idx, r_idx, t_idx = nodes[i], rels[i], nodes[i + 1]
            h_name = ents[h_idx] if 0 <= h_idx < len(ents) else "?"
            t_name = ents[t_idx] if 0 <= t_idx < len(ents) else "?"
            r_text = rel_to_text(rels_list[r_idx]) if 0 <= r_idx < len(rels_list) else "?"
            raw.append((h_name, r_text, t_name))
        # Bridge through CVT nodes
        i = 0
        while i < len(raw):
            h, r, t = raw[i]
            if is_cvt_like(h):
                i += 1
                continue
            if is_cvt_like(t):
                merged_rels = [r]
                j = i
                while j < len(raw) - 1 and is_cvt_like(raw[j][2]):
                    j += 1
                    merged_rels.append(raw[j][1])
                final_t = raw[j][2]
                if not is_cvt_like(final_t):
                    merged_r = " > ".join(merged_rels)
                    sig = (normalize(h), normalize(merged_r), normalize(final_t))
                    if sig not in seen:
                        seen.add(sig)
                        triples.append((h, merged_r, final_t))
                    i = j + 1
                else:
                    # Preserve tail relation even if it ends at a CVT/value node.
                    sig = (normalize(h), normalize(r), normalize(final_t))
                    if sig not in seen:
                        seen.add(sig)
                        triples.append((h, r, final_t))
                    i += 1
            else:
                sig = (normalize(h), normalize(r), normalize(t))
                if sig not in seen:
                    seen.add(sig)
                    triples.append((h, r, t))
                i += 1
    return triples


def _extract_relation_segments_from_path(path, ents):
    """Collapse a raw path into relation segments.

    Keep:
    - non-CVT -> non-CVT segments
    - direct non-CVT -> ... -> CVT tail segment only when no later non-CVT entity
      has already been reached

    This preserves direct value constraints such as:
    country -> statistical_region.child_labor_percent -> g.xxx
    but removes tails such as:
    person -> religion -> Judaism -> membership -> CVT
    where the core relation has already reached a real entity.
    """
    sig_positions = []
    for i, node_idx in enumerate(path.get("nodes", [])):
        name = ents[node_idx] if 0 <= node_idx < len(ents) else ""
        if not is_cvt_like(name):
            sig_positions.append(i)
    segments = []
    for j in range(1, len(sig_positions)):
        prev_pos = sig_positions[j - 1]
        curr_pos = sig_positions[j]
        rel_seq = path.get("relations", [])[prev_pos:curr_pos]
        if rel_seq:
            segments.append(list(rel_seq))
    # Preserve the tail segment only if the path never reached a second non-CVT
    # node. Once a real entity has already been reached, later CVT tails are not
    # part of the core relation pattern.
    nodes = path.get("nodes", [])
    rels = path.get("relations", [])
    if len(sig_positions) == 1:
        last_sig = sig_positions[-1]
        if last_sig < len(nodes) - 1:
            tail_rel_seq = rels[last_sig:]
            if tail_rel_seq:
                segments.append(list(tail_rel_seq))
    return segments


def build_pattern_evidence_triples(selected_patterns, ents, rels_list, h_ids, r_ids, t_ids,
                                   anchor_idx, max_grouped_lines=120):
    """Rebuild evidence subgraph from selected patterns instead of replaying a single path.

    For each selected pattern:
    - take its witness raw path
    - extract relation segments between non-CVT nodes
    - starting from the anchor, expand segment by segment
    - keep all triples matched at the current segment, even if a branch cannot continue later

    Important:
    - do NOT add generic 1-hop local neighborhoods
    - only keep pattern-driven evidence triples
    This avoids drowning later-step evidence in anchor-side noise.
    """
    triples = []
    seen = set()

    def _add_triple(h_idx, r_idx, t_idx):
        h_name = ents[h_idx] if 0 <= h_idx < len(ents) else "?"
        t_name = ents[t_idx] if 0 <= t_idx < len(ents) else "?"
        r_text = rel_to_text(rels_list[r_idx]) if 0 <= r_idx < len(rels_list) else "?"
        sig = (normalize(h_name), normalize(r_text), normalize(t_name))
        if sig not in seen:
            seen.add(sig)
            triples.append((h_name, r_text, t_name))

    # Pre-index triples by node and relation for fast local expansion
    node_rel_edges = {}
    for i in range(len(h_ids)):
        h, r, t = h_ids[i], r_ids[i], t_ids[i]
        node_rel_edges.setdefault((h, r), []).append((h, r, t))
        node_rel_edges.setdefault((t, r), []).append((h, r, t))

    for lp in selected_patterns:
        witness = lp.get("best_raw_path")
        if not witness:
            continue
        segments = _extract_relation_segments_from_path(witness, ents)
        frontier = {anchor_idx}
        for rel_seq in segments:
            if not frontier:
                break
            current_frontier = set(frontier)
            next_frontier = set()
            for rel_idx in rel_seq:
                step_next = set()
                for node_idx in current_frontier:
                    for h_idx, r_idx, t_idx in node_rel_edges.get((node_idx, rel_idx), []):
                        _add_triple(h_idx, r_idx, t_idx)
                        if h_idx == node_idx:
                            step_next.add(t_idx)
                        if t_idx == node_idx:
                            step_next.add(h_idx)
                current_frontier = step_next
                next_frontier = step_next
            if next_frontier:
                frontier = next_frontier

    return triples


def format_grouped_triples(triples, max_lines=80, max_tails=12):
    """Group triples by (head, relation) to reduce token usage."""
    groups = {}
    for h, r, t in triples:
        groups.setdefault((h, r), []).append(t)
    lines = []
    for (h, r), tails in groups.items():
        uniq = []
        seen = set()
        for t in tails:
            nt = normalize(t)
            if nt not in seen:
                seen.add(nt)
                uniq.append(t)
        if len(uniq) == 1:
            lines.append(f"({h}, {r}, {uniq[0]})")
        else:
            shown = uniq[:max_tails]
            suffix = f", ...(+{len(uniq) - len(shown)})" if len(uniq) > len(shown) else ""
            lines.append(f"({h}, {r}, [{', '.join(shown)}{suffix}])")
        if len(lines) >= max_lines:
            break
    return lines


def collect_local_subgraph_triples(seed_node_indices, ents, rels_list, h_ids, r_ids, t_ids,
                                   max_per_seed=6, global_limit=120):
    """Collect a small 1-hop local subgraph around selected witness/candidate nodes.

    This is for final reasoning only. It exposes local evidence around bridge nodes
    and candidate entities that is not visible from the witness path alone.
    """
    triples = []
    seen = set()

    def _add_triple(h_idx, r_idx, t_idx):
        h_name = ents[h_idx] if 0 <= h_idx < len(ents) else "?"
        t_name = ents[t_idx] if 0 <= t_idx < len(ents) else "?"
        r_text = rel_to_text(rels_list[r_idx]) if 0 <= r_idx < len(rels_list) else "?"
        sig = (normalize(h_name), normalize(r_text), normalize(t_name))
        if sig not in seen:
            seen.add(sig)
            triples.append((h_name, r_text, t_name))

    for node_idx in seed_node_indices:
        count = 0
        if node_idx is None or not (0 <= node_idx < len(ents)):
            continue
        for i in range(len(h_ids)):
            if len(triples) >= global_limit or count >= max_per_seed:
                break
            if h_ids[i] == node_idx or t_ids[i] == node_idx:
                _add_triple(h_ids[i], r_ids[i], t_ids[i])
                count += 1
        if len(triples) >= global_limit:
            break
    return triples


# ---------------------------------------------------------------------------
# NER-based entity resolution (validated approach from test_full_pipeline.py)
# ---------------------------------------------------------------------------

DECOMP_PROMPT_NER = """Decompose the question into sub-questions that trace a path through the knowledge graph.
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


def _token_overlap(question: str, entity: str) -> float:
    """Ratio of entity tokens found in question."""
    q_tokens = set(normalize(question).split())
    e_tokens = set(normalize(entity).split())
    if not e_tokens:
        return 0.0
    return len(e_tokens & q_tokens) / len(e_tokens)


async def resolve_anchor_ner(session, question, entity_list, rel_list, h_ids, r_ids, t_ids):
    """NER + GTE + token overlap + relation overlap scoring for entity resolution."""
    from collections import defaultdict

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
    gte_ents = [(r["candidate"], r.get("score", 0)) for r in q_ent_rows
                if r.get("candidate") and r["candidate"] in entity_list]

    # Token overlap filter (min 0.5)
    filtered = [(e, s) for e, s in gte_ents if _token_overlap(question, e) >= 0.5]

    # Relation overlap scoring
    q_rel_rows = await gte_retrieve(session, question, rel_list,
                                    candidate_texts=rel_texts, top_k=5)
    q_top_rel_idx = {rel_list.index(r["candidate"]) for r in q_rel_rows
                     if r.get("candidate") and r["candidate"] in rel_list}

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


def _parse_ner_steps(raw):
    """Parse steps from NER decomposition format."""
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


def _parse_ner_start(raw):
    m = re.search(r"^Start:\s*(.+)$", raw, re.M)
    return m.group(1).strip() if m else None


def _parse_ner_end(raw):
    m = re.search(r"^End:\s*(.+)$", raw, re.M)
    end = m.group(1).strip() if m else "none"
    if end.lower() == "none":
        return None, None
    return end, end


def _collect_hr_frontier(anchor_idx, step_relations, h_ids, r_ids, t_ids):
    """Collect HR frontier: from anchor, expand layer-by-layer using step relations."""
    all_triples = []
    all_nodes = {anchor_idx}
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
        all_nodes |= next_frontier
        frontier = next_frontier
    return all_triples, all_nodes


async def run_case(session, sample, pilot_row):
    question = pilot_row["question"]
    gt_answers = pilot_row.get("gt", pilot_row.get("ground_truth", pilot_row.get("gt_answers", [])))
    ents = sample.get("text_entity_list", []) + sample.get("non_text_entity_list", [])
    rels = list(sample.get("relation_list", []))
    h_ids, r_ids, t_ids = sample.get("h_id_list", []), sample.get("r_id_list", []), sample.get("t_id_list", [])

    # ── NER entity resolution (BEFORE expand_cvt_leaves, using original data) ──
    ner_scored, ner_name_to_ids = await resolve_anchor_ner(
        session, question, ents, rels, h_ids, r_ids, t_ids)
    ner_top_ents = []
    _seen_e = set()
    for s in ner_scored[:6]:
        if s["entity"] not in _seen_e:
            ner_top_ents.append((s["entity"], s["gte"]))
            _seen_e.add(s["entity"])

    # Auto-expand CVT leaf nodes (degree ≤ 1)
    ents, rels, h_ids, r_ids, t_ids = expand_cvt_leaves(ents, rels, h_ids, r_ids, t_ids)
    rel_texts = [f"{r} ; {rel_to_text_short(r)}" for r in rels]

    # Build entity candidates (non-CVT)
    ent_candidates = [e for e in ents if e and len(e) > 1 and not is_cvt_like(e)]

    # Re-map NER entity names to expanded entity list indices
    ner_name_to_ids_expanded = {}
    for i, name in enumerate(ents):
        ner_name_to_ids_expanded.setdefault(name, []).append(i)

    async def execute_planning(anchor_forbidden=None, step_relations_override=None,
                               retry_note=None, use_ner=True):
        entity_retrieval_details = []
        anchor_idx = None
        anchor_name = None

        # ── NER-based anchor resolution (with fallback) ────────────────
        ner_ok = False
        if use_ner and ner_top_ents:
            decomp_question = f"Question: {question} [NER mode]"
            ent_str = ", ".join(f'("{e}", {s:.1f})' for e, s in ner_top_ents)
            raw = await call_llm(session, [
                {"role": "system", "content": DECOMP_PROMPT_NER},
                {"role": "user", "content": f"Question: {question}\nRetrieved entities: [{ent_str}]"},
            ], max_tokens=600)

            start_name = _parse_ner_start(raw)
            steps = _parse_ner_steps(raw)
            end_name, end_query = _parse_ner_end(raw)

            if steps:
                # Resolve anchor from NER scored entities
                if start_name:
                    sn = normalize(start_name)
                    for s in ner_scored:
                        en = normalize(s["entity"])
                        if sn == en or sn in en or en in sn:
                            anchor_idx = ner_name_to_ids_expanded.get(s["entity"], [None])[0]
                            anchor_name = s["entity"]
                            break
                if anchor_idx is None and ner_scored:
                    anchor_name = ner_scored[0]["entity"]
                    anchor_idx = ner_name_to_ids_expanded.get(anchor_name, [None])[0]

                entity_retrieval_details.append({
                    "role": "anchor_ner",
                    "ner_top_ents": ner_top_ents[:6],
                    "selected": anchor_name,
                    "selected_idx": anchor_idx,
                })

                # Convert NER steps to standard format
                for step in steps:
                    step["entity_query"] = None
                # Endpoint resolve
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
                    entity_retrieval_details.append({
                        "role": f"endpoint_step{steps[-1]['step']}",
                        "query": end_query,
                        "selected": best,
                        "selected_idx": idx,
                        "llm_resolved": True,
                    })
                ner_ok = True

        # ── Original decomposition-based anchor resolution (fallback) ──
        if not ner_ok:
            decomp_question = f"Question: {question}"
            if retry_note:
                decomp_question += f"\n\nRetry instruction: {retry_note}"
            if anchor_forbidden:
                decomp_question += f"\nDo not use this previous anchor again: {anchor_forbidden}"

            raw = await call_llm(session, [
                {"role": "system", "content": DECOMP_PROMPT},
                {"role": "user", "content": decomp_question},
            ])
            anchor_eq_name, anchor_eq, steps = parse_decomposition(raw)
            if not steps:
                return {"error": "decomposition failed", "raw": raw, "decomp_question": decomp_question}

            if anchor_eq:
                rows = await gte_retrieve(session, anchor_eq, ent_candidates, top_k=5)
                topk = [{"rank": i+1, "candidate": r.get("candidate", ""), "score": round(r.get("score", 0), 4)} for i, r in enumerate(rows)]
                anchor_cands = [r["candidate"] for r in topk if r["candidate"]]
                anchor_ctx = get_entity_contexts(anchor_cands, h_ids, r_ids, t_ids, ents, rels)
                anchor_cands_with_ctx = [(n, anchor_ctx.get(n, "")) for n in anchor_cands]
                selected = await llm_resolve_entity(session, question, anchor_eq, anchor_cands_with_ctx)
                anchor_idx = ents.index(selected) if selected and selected in ents else None
                anchor_name = selected or anchor_eq_name
                entity_retrieval_details.append({
                    "role": "anchor",
                    "query": anchor_eq,
                    "top_k": topk,
                    "selected": selected,
                    "selected_idx": anchor_idx,
                    "llm_resolved": True,
                })

            # Endpoint resolve for original format
            breakpoints = {}
            for step in steps:
                if step["endpoint"] and step.get("endpoint_query"):
                    ep_rows = await gte_retrieve(session, step["endpoint_query"], ent_candidates, top_k=3)
                    ep_topk = [{"rank": i+1, "candidate": r.get("candidate", ""), "score": round(r.get("score", 0), 4)} for i, r in enumerate(ep_rows)]
                    ep_cands = [r["candidate"] for r in ep_topk if r["candidate"]]
                    ep_ctx = get_entity_contexts(ep_cands, h_ids, r_ids, t_ids, ents, rels)
                    ep_cands_with_ctx = [(n, ep_ctx.get(n, "")) for n in ep_cands]
                    best = await llm_resolve_entity(session, question, step["endpoint_query"], ep_cands_with_ctx)
                    idx = ents.index(best) if best and best in ents else None
                    if idx is not None:
                        breakpoints[step["step"]] = idx
                    entity_retrieval_details.append({
                        "role": f"endpoint_step{step['step']}",
                        "query": step["endpoint_query"],
                        "top_k": ep_topk,
                        "selected": best,
                        "selected_idx": idx,
                        "llm_resolved": True,
                    })

        # ── Shared: dual GTE + prune + expand ──────────────────────────
        step_candidates = {}
        gte_per_step = {}
        relation_retrieval_details = []
        for step in steps:
            rq = step.get("relation_query", step["question"])
            gte_all = {}
            queries_detail = []
            for query in [rq, step["question"]]:
                rows = await gte_retrieve(session, query, rels, candidate_texts=rel_texts, top_k=10)
                topk = []
                for i, r in enumerate(rows):
                    cand = r.get("candidate", "")
                    score = round(r.get("score", 0), 4)
                    idx_in_rels = rels.index(cand) if cand in rels else None
                    topk.append({"rank": i+1, "candidate": cand, "score": score, "rel_idx": idx_in_rels,
                                 "rel_text": rel_to_text(cand) if idx_in_rels is not None else ""})
                    if idx_in_rels is not None:
                        if idx_in_rels not in gte_all or score > gte_all[idx_in_rels][1]:
                            gte_all[idx_in_rels] = (rels[idx_in_rels], score)
                queries_detail.append({"query": query, "top_k": topk})

            gte_candidates = sorted(gte_all.items(), key=lambda x: -x[1][1])
            candidate_list = [(idx, name, score) for idx, (name, score) in gte_candidates]
            step_candidates[step["step"]] = candidate_list
            gte_per_step[step["step"]] = gte_all
            relation_retrieval_details.append({
                "step": step["step"],
                "queries": queries_detail,
                "gte_candidates_count": len(gte_all),
                "gte_indices": sorted(gte_all.keys()),
            })

        prune_result, prune_debug = await llm_prune_all_relations(session, question, steps, step_candidates)

        step_relations = []
        for step in steps:
            sn = step["step"]
            pruned = prune_result.get(sn, set())
            # Safety net: always include top-2 GTE results alongside LLM selection
            gte_top2 = {idx for idx, _, _ in step_candidates.get(sn, [])[:2]}
            merged = pruned | gte_top2
            step_relations.append(merged)
            gte_all = gte_per_step.get(sn, {})
            resolved_names = [{"idx": ri, "name": gte_all[ri][0]} for ri in sorted(pruned)] if pruned else []
            for rd in relation_retrieval_details:
                if rd["step"] == sn:
                    rd["resolved_indices"] = sorted(pruned)
                    rd["resolved_names"] = resolved_names
                    break

        if step_relations_override:
            for layer_idx, override_set in step_relations_override.items():
                if 0 <= layer_idx < len(step_relations):
                    step_relations[layer_idx] = set(override_set)
                    for rd in relation_retrieval_details:
                        if rd["step"] == steps[layer_idx]["step"]:
                            rd["override_indices"] = sorted(step_relations[layer_idx])
                            rd["override_names"] = [{"idx": ri, "name": rels[ri]} for ri in sorted(step_relations[layer_idx]) if 0 <= ri < len(rels)]
                            break

        prune_debug_field = {
            "prompt": prune_debug["prompt"],
            "response": prune_debug["response"],
            "parsed_yaml": prune_debug.get("parsed_yaml"),
        }

        paths, max_depth, max_cov = [], 0, 0
        answer_candidates = []
        all_subgraph_nodes = set()
        if anchor_idx is not None:
            bp_set = set(breakpoints.values()) - {anchor_idx, None}
            paths, max_depth, max_cov = relation_prior_expand(
                anchor_idx, step_relations, h_ids, r_ids, t_ids, ents,
                explicit_targets=bp_set if bp_set else None)

            # Collect path nodes
            all_subgraph_nodes = {anchor_idx}
            for path in paths:
                all_subgraph_nodes.update(path["nodes"])

            # HR frontier: expand layer-by-layer using step relations
            hr_triples, hr_nodes = _collect_hr_frontier(
                anchor_idx, step_relations, h_ids, r_ids, t_ids)
            all_subgraph_nodes |= hr_nodes

            # Extract answer candidates from subgraph nodes (+ CVT expansion)
            for node_idx in sorted(all_subgraph_nodes):
                if node_idx == anchor_idx:
                    continue
                name = ents[node_idx] if 0 <= node_idx < len(ents) else ""
                if is_cvt_like(name):
                    for cvt_idx, _ in expand_through_cvt(node_idx, h_ids, r_ids, t_ids, ents):
                        if cvt_idx != anchor_idx and 0 <= cvt_idx < len(ents) and not is_cvt_like(ents[cvt_idx]):
                            answer_candidates.append(ents[cvt_idx])
                else:
                    answer_candidates.append(name)

            seen = set()
            unique = []
            for c in answer_candidates:
                nc = normalize(c)
                if nc not in seen:
                    seen.add(nc)
                    unique.append(c)
            answer_candidates = unique

        gt_hit = candidate_hit(answer_candidates, gt_answers) if answer_candidates else False
        return {
            "error": None,
            "raw": raw,
            "decomp_question": decomp_question,
            "steps": steps,
            "anchor_idx": anchor_idx,
            "anchor_name": ents[anchor_idx] if anchor_idx is not None else anchor_name,
            "breakpoints": breakpoints,
            "step_relations_sets": step_relations,
            "entity_retrieval_details": entity_retrieval_details,
            "relation_retrieval_details": relation_retrieval_details,
            "prune_debug": prune_debug_field,
            "step_candidates": step_candidates,
            "paths": paths,
            "max_depth": max_depth,
            "max_cov": max_cov,
            "answer_candidates": answer_candidates,
            "gt_hit": gt_hit,
        }

    def _attempt_score(state):
        if state.get("error"):
            return (-1, -1, -1)
        return (
            state.get("max_cov", 0),
            len(state.get("paths", [])),
            len(state.get("answer_candidates", [])),
        )

    planning_attempts = []
    active = await execute_planning()
    planning_attempts.append({"label": "primary", "score": _attempt_score(active), "anchor": active.get("anchor_name")})

    if active.get("error"):
        return {
            "case_id": pilot_row["case_id"],
            "question": question,
            "error": active["error"],
            "gt_answers": gt_answers,
            "answer_candidates": [],
            "gt_hit": False,
            "raw": active.get("raw"),
        }

    nonempty_layers = [i for i, rs in enumerate(active["step_relations_sets"]) if rs]
    layer_diagnostics = diagnose_layers(
        active["anchor_idx"], active["step_relations_sets"], h_ids, r_ids, t_ids, ents,
        max_hops=3,
    ) if active.get("anchor_idx") is not None else []

    first_miss = None
    for li in nonempty_layers:
        if li < len(layer_diagnostics) and not layer_diagnostics[li]["frontier_hit"]:
            first_miss = li
            break

    # Retry A: if current layer can be reached from anchor but not current frontier,
    # treat previous layer as noise and skip it.
    if first_miss is not None and first_miss > 0:
        d = layer_diagnostics[first_miss]
        if d["anchor_hit"] and not d["frontier_hit"]:
            overrides = {first_miss - 1: set()}
            alt = await execute_planning(
                step_relations_override=overrides,
                retry_note=f"Previous layer appears noisy. Keep the same decomposition but skip step {first_miss} if needed.",
            )
            planning_attempts.append({"label": "skip_previous_layer", "score": _attempt_score(alt), "anchor": alt.get("anchor_name")})
            if _attempt_score(alt) > _attempt_score(active):
                active = alt
                layer_diagnostics = diagnose_layers(
                    active["anchor_idx"], active["step_relations_sets"], h_ids, r_ids, t_ids, ents,
                    max_hops=3,
                ) if active.get("anchor_idx") is not None else []
                first_miss = None
                for li in [i for i, rs in enumerate(active["step_relations_sets"]) if rs]:
                    if li < len(layer_diagnostics) and not layer_diagnostics[li]["frontier_hit"]:
                        first_miss = li
                        break

    # Retry B: if a layer misses both from frontier and anchor, reselect that layer's relations.
    if first_miss is not None:
        d = layer_diagnostics[first_miss]
        if not d["frontier_hit"] and not d["anchor_hit"]:
            reselection = await llm_reselect_single_step_relation(
                session, question, active["steps"][first_miss],
                active["step_candidates"], active["step_relations_sets"][first_miss],
            )
            overrides = {first_miss: reselection}
            alt = await execute_planning(
                step_relations_override=overrides,
                retry_note=f"Retry relation selection for step {first_miss + 1}. The previous relation choice was too weak or too noisy.",
            )
            planning_attempts.append({"label": "reselect_failed_layer", "score": _attempt_score(alt), "anchor": alt.get("anchor_name")})
            if _attempt_score(alt) > _attempt_score(active):
                active = alt

    # Retry C: if still weak and there are multiple explicit entities, redecompose with a different anchor.
    active_nonempty = [i for i, rs in enumerate(active["step_relations_sets"]) if rs]
    if active_nonempty and active.get("max_cov", 0) < len(active_nonempty):
        explicit_entities = {active.get("anchor_name")} | set(ents[idx] for idx in active.get("breakpoints", {}).values() if idx is not None and 0 <= idx < len(ents))
        explicit_entities = {e for e in explicit_entities if e}
        if len(explicit_entities) >= 2 and active.get("anchor_name"):
            alt = await execute_planning(
                anchor_forbidden=active["anchor_name"],
                retry_note="Choose a different explicit anchor entity from the question and redecompose the plan from that anchor.",
            )
            planning_attempts.append({"label": "anchor_swap_redecompose", "score": _attempt_score(alt), "anchor": alt.get("anchor_name")})
            if _attempt_score(alt) > _attempt_score(active):
                active = alt

    final_layer_diagnostics = diagnose_layers(
        active["anchor_idx"], active["step_relations_sets"], h_ids, r_ids, t_ids, ents,
        max_hops=3,
    ) if active.get("anchor_idx") is not None else []

    raw = active["raw"]
    decomp_question = active["decomp_question"]
    steps = active["steps"]
    anchor_idx = active["anchor_idx"]
    breakpoints = active["breakpoints"]
    step_relations = active["step_relations_sets"]
    entity_retrieval_details = active["entity_retrieval_details"]
    relation_retrieval_details = active["relation_retrieval_details"]
    prune_debug_field = active["prune_debug"]
    paths = active["paths"]
    max_depth = active["max_depth"]
    answer_candidates = active["answer_candidates"]
    gt_hit = active["gt_hit"]

    # Step 5: Compress paths into logical patterns
    breakpoint_indices = set(breakpoints.values())
    logical_paths = compress_paths(paths, ents, rels, anchor_idx, breakpoint_indices) if paths else []

    # Step 6: Multi-attempt LLM reasoning with rollback
    # Model selects paths, can trigger rollback to get more paths (up to 3 attempts)
    # After all attempts, model reasons over all selected paths' triples
    llm_answer = None
    llm_hit = False
    selected_paths = []
    num_triples = 0
    attempt_log = []
    llm_reasoning_prompt = None
    llm_reasoning_full = None
    if logical_paths:
        try:
            remaining = list(range(len(logical_paths)))

            for attempt in range(3):
                if not remaining:
                    break
                # Present remaining patterns with sequential numbering
                path_lines = []
                idx_map = {}  # display_number -> actual index in logical_paths
                for display_num, actual_idx in enumerate(remaining[:15], 1):
                    path_lines.append(f"{display_num}. {logical_paths[actual_idx]['readable']}")
                    idx_map[display_num] = actual_idx
                paths_text = "\n".join(path_lines)

                select_prompt = f"""Analyze and select reasoning paths for this question.

Question: {question}

Paths:
{paths_text}

Instructions:
1. Ignore hidden entity identities. Judge each path only by its relation sequence and node structure.
2. Select paths by semantic relevance, not by shortest length.
3. Keep a path if it preserves the intended multi-step meaning of the question, even if it is longer, includes bridge nodes, or is not the most direct-looking path.
4. Do not eliminate a path only because it is longer, slightly noisy, or contains extra intermediate structure.
5. When uncertain, prefer recall over precision: keep all paths that are semantically plausible.
6. Remove only paths that clearly contradict the question semantics.

Output format:
<analysis>
Two or three short sentences about which paths preserve the question semantics and which clearly do not.
</analysis>
<selected>comma-separated path indices only</selected>
<need_more>yes or no</need_more>

Rules:
- Do not copy example numbers.
- Do not leave out a semantically plausible path just because it is longer.
- If several paths are plausible, select all of them."""

                # Try up to 3 times for valid XML output
                sel_raw = ""
                for sel_attempt in range(3):
                    sel_raw = await call_llm(session, [
                        {"role": "system", "content": "You analyze and select reasoning paths for multi-step QA. Your goal is high-recall semantic path selection. Judge semantic fit of relation chains to the question. Keep all semantically plausible paths. Output exactly three XML tags: <analysis>, <selected>, and <need_more>."},
                        {"role": "user", "content": select_prompt},
                    ], max_tokens=1500)
                    if extract_xml_tag(sel_raw, "selected"):
                        break

                # Parse selected indices from <selected> tag
                sel_indices = []
                sel_text = extract_xml_tag(sel_raw, "selected") or ""
                for m in re.finditer(r'\d+', sel_text):
                    display_num = int(m.group())
                    if display_num in idx_map:
                        sel_indices.append(idx_map[display_num])

                # Parse rollback flag from <need_more> tag
                need_more_text = extract_xml_tag(sel_raw, "need_more") or "no"
                rollback = "yes" in need_more_text.lower()

                selected_paths.extend(sel_indices)
                attempt_log.append({
                    "attempt": attempt + 1,
                    "selected": sel_indices,
                    "rollback": rollback,
                    "raw": sel_raw.strip(),
                    "prompt": select_prompt,
                    "full_response": sel_raw,
                })

                # Remove selected from remaining pool
                selected_set = set(sel_indices)
                remaining = [i for i in remaining if i not in selected_set]

                # Stop if no rollback or max attempts reached
                if not rollback or attempt >= 2:
                    break

            # Deduplicate selected paths
            selected_paths = list(dict.fromkeys(selected_paths))
            # Fallback: if nothing selected, use top 3 by candidate count
            if not selected_paths:
                selected_paths = list(range(min(3, len(logical_paths))))

            # Build final reasoning subgraph from selected patterns instead of
            # replaying only one witness path. This preserves all (h, r, *)
            # evidence matched by the chosen pattern layers.
            selected_pattern_objs = []
            for idx in selected_paths:
                lp = logical_paths[idx]
                selected_pattern_objs.append(lp)
            triples = build_pattern_evidence_triples(
                selected_pattern_objs, ents, rels, h_ids, r_ids, t_ids, anchor_idx,
                max_grouped_lines=120,
            )
            num_triples = len(triples)

            if triples:
                triple_lines = format_grouped_triples(triples, max_lines=80, max_tails=12)
                triples_text = "\n".join(triple_lines)
                # Candidate entity names (no context — triples already provide relations)
                unique_ents = sorted({e for h, r, t in triples for e in (h, t)})
                entity_lines = "\n".join(f"  - {e}" for e in unique_ents)

                reason_prompt = f"""QUESTION: {question}

GRAPH TRIPLES (from Freebase, snapshot circa 2015):
{triples_text}

━━━ REASONING PROTOCOL ━━━
1. GRAPH-BASED ANALYSIS
   - Analyze the graph triples to identify graph-supported candidate answers.
   - Use relation chains, grouped tails, and visible attributes in the triples.

2. PARAMETRIC KNOWLEDGE ANALYSIS
   - Use your own knowledge only to interpret incomplete graph evidence or break ties among graph-supported candidates.
   - Do not introduce a brand-new answer that is not present in the graph triples.
   - The graph reflects Freebase circa 2015, so some attributes may be incomplete.

3. ANSWER SELECTION
   - Prefer entities directly supported by the graph.
   - If the graph is incomplete, use world knowledge only to choose among entities already connected by the graph.
   - If graph + knowledge converge on one graph-supported entity → output it.
   - If multiple graph-supported entities remain plausible → output all plausible ones.
   - Every answer MUST be an EXACT string from the graph triples. No truncation or paraphrase.

━━━ OUTPUT FORMAT ━━━
<reasoning>
  [GRAPH ANALYSIS]
  - One or two sentences: what the graph supports.

  [PARAMETRIC KNOWLEDGE]
  - One sentence: how world knowledge helps interpret or narrow the graph-supported candidates.

  [CANDIDATE EVALUATION]
  - Entity: [Keep/Eliminate] — one short reason per key candidate

  [SPELLING VERIFICATION]
  - Final: "exact string(s) from triples"
</reasoning>
<answer>\\boxed{{exact entity}}</answer>

Multiple plausible: <answer>\\boxed{{cand1}} \\boxed{{cand2}} \\boxed{{cand3}}</answer>
NO text after </answer> tag."""

                llm_raw = await call_llm(session, [
                    {"role": "system", "content": "You are a precise graph QA system over Freebase (circa 2015). Prioritize graph evidence. Use world knowledge only to interpret incomplete graph evidence or choose among graph-supported candidates. Never output an answer that is not explicitly present in the graph triples. Output exact graph strings only."},
                    {"role": "user", "content": reason_prompt},
                ], max_tokens=1200)
                # Extract answer from <answer>...\boxed{...}...</answer> format
                ans_match = re.search(r'<answer>(.*?)</answer>', llm_raw, re.DOTALL)
                if ans_match:
                    boxed = re.findall(r'\\boxed\{([^}]+)\}', ans_match.group(1))
                    if boxed:
                        llm_answer = " | ".join(b.strip() for b in boxed)
                        llm_hit = candidate_hit([b.strip() for b in boxed], gt_answers)
                    else:
                        llm_answer = ans_match.group(1).strip()
                        llm_hit = candidate_hit([llm_answer], gt_answers)
                else:
                    # Legacy fallback
                    ans_match = re.search(r'ANSWER:\s*(.+)', llm_raw, re.IGNORECASE)
                    if ans_match:
                        llm_answer = ans_match.group(1).strip()
                    else:
                        lines = [l.strip() for l in llm_raw.strip().split('\n') if l.strip()]
                        llm_answer = lines[-1] if lines else llm_raw.strip()
                    llm_hit = candidate_hit([llm_answer], gt_answers)
                llm_reasoning_prompt = reason_prompt
                llm_reasoning_full = llm_raw
        except Exception as e:
            print(f"  LLM reasoning error: {type(e).__name__}: {e}")
            llm_answer = None
            llm_hit = False

    return {
        "case_id": pilot_row["case_id"],
        "question": question,
        "gt_answers": gt_answers,
        "decomposition_prompt": DECOMP_PROMPT,
        "decomposition_question": decomp_question,
        "decomposition": raw,
        "steps_parsed": steps,
        "anchor_idx": anchor_idx,
        "anchor_name": ents[anchor_idx] if anchor_idx is not None else None,
        "breakpoints": {k: ents[v] for k, v in breakpoints.items()},
        "step_relations": [list(r) for r in step_relations],
        "entity_retrieval_details": entity_retrieval_details,
        "relation_retrieval_details": relation_retrieval_details,
        "prune_debug": prune_debug_field,
        "layer_diagnostics": final_layer_diagnostics,
        "planning_attempts": planning_attempts,
        "max_depth": max_depth,
        "num_paths": len(paths),
        "answer_candidates": answer_candidates,
        "gt_hit": gt_hit,
        "num_patterns": len(logical_paths),
        "logical_paths": [lp["readable"] for lp in logical_paths[:10]],
        "pattern_details": [{
            "candidates": lp.get("candidates", []),
            "best_tier": lp.get("best_tier", (0, -1, 0)),
            "endpoint": lp.get("endpoint"),
            "witness_nodes": lp.get("best_raw_path", {}).get("nodes", []),
            "witness_relations": lp.get("best_raw_path", {}).get("relations", []),
        } for lp in logical_paths],
        "llm_answer": llm_answer,
        "llm_hit": llm_hit,
        "selected_paths": selected_paths,
        "num_triples": num_triples,
        "attempt_log": attempt_log,
        "llm_reasoning_prompt": llm_reasoning_prompt,
        "llm_reasoning_full": llm_reasoning_full,
    }


async def amain():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-results", default=str(DEFAULT_PILOT))
    parser.add_argument("--cwq-pkl", default=str(DEFAULT_CWQ))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    pilot_rows = json.loads(Path(args.pilot_results).read_text())
    samples = pickle.loads(Path(args.cwq_pkl).read_bytes())
    sample_map = {}
    for s in samples:
        if "id" in s:
            sample_map[s["id"]] = s
        elif "question_id" in s:
            sample_map[s["question_id"]] = s

    rows = []
    async with aiohttp.ClientSession() as session:
        for pilot_row in pilot_rows[:args.limit]:
            sample = sample_map.get(pilot_row["case_id"])
            if not sample:
                # fallback: match by question
                for s in samples:
                    if s.get("question", "") == pilot_row["question"]:
                        sample = s; break
            if not sample:
                print(f"SKIP: {pilot_row['case_id']}")
                continue
            try:
                result = await run_case(session, sample, pilot_row)
            except Exception as e:
                print(f"ERROR: {pilot_row['case_id']}: {type(e).__name__}: {e}")
                continue
            if result is None:
                continue
            rows.append(result)
            status = "HIT" if result["gt_hit"] else "MISS"
            llm_status = "LLM_HIT" if result.get("llm_hit") else "LLM_MISS"
            n_patterns = result.get("num_patterns", 0)
            n_triples = result.get("num_triples", 0)
            cid = result.get('case_id', '?') or '?'
            print(f"{status} | {cid[:30]} | paths={result.get('num_paths',0):6d} | patterns={n_patterns:3d} | triples={n_triples:3d} | {llm_status} | llm={str(result.get('llm_answer',''))[:25]} | gt={result.get('gt_answers')}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    cases = len(rows)
    if cases == 0:
        print("\n=== No valid cases processed ===")
        return
    hits = sum(1 for r in rows if r["gt_hit"])
    llm_hits = sum(1 for r in rows if r.get("llm_hit"))
    avg_paths = sum(r.get("num_paths", 0) for r in rows) / cases
    avg_patterns = sum(r.get("num_patterns", 0) for r in rows) / cases
    print(f"\n=== Summary: GT_recall={hits}/{cases} ({100*hits/cases:.1f}%) | LLM_reason={llm_hits}/{cases} ({100*llm_hits/cases:.1f}%) ===")
    print(f"    Avg paths: {avg_paths:.0f} | Avg compressed patterns: {avg_patterns:.1f}")


if __name__ == "__main__":
    asyncio.run(amain())
