#!/usr/bin/env python3
"""End-to-end test: chain decomposition (v6.2) → GTE retrieval → LLM pruning → graph traversal.

Tests if the v6.2 chain decomposition output works with the downstream pipeline stages.
Does NOT integrate into main pipeline — standalone evaluation only.
"""

import json, re, pickle, time, argparse, sys
from pathlib import Path
import requests

# Import traversal functions from main pipeline
sys.path.insert(0, '/zhaoshu/subgraph/scripts')
from test_chain_decompose import (
    frontier_expand_layers, chain_expand_v2, relation_prior_expand,
    is_cvt_like, normalize, candidate_hit, expand_through_cvt, expand_node,
    extract_xml_tag,
)


# ═══════════════════════════════════════════════════════════════════
# LLM call (synchronous, local vLLM)
# ═══════════════════════════════════════════════════════════════════

def call_llm(messages, model_id="qwen35-9b-local", max_tokens=1536):
    resp = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=90,
    )
    return resp.json()['choices'][0]['message']['content']


# ═══════════════════════════════════════════════════════════════════
# Prompt: v6.2 chain decomposition (relation-semantic definitions)
# ═══════════════════════════════════════════════════════════════════

CHAIN_PROMPT = """You are a knowledge graph reasoning expert. Analyze the question and decompose it into atomic reasoning steps.

## Entities
Named entities from the question: {entities}

## CRITICAL RULES

### Rule 1: Each hop = ONE atomic graph edge
A hop traverses exactly ONE relation. Compound actions MUST be split:
- "find championship year" → TWO hops: find championship events + find event start date
- "find manufacturer's country" → TWO hops: find manufacturer + find country
Attribute lookups (dates, names, counts, types) are ALWAYS separate hops.

### Rule 2: No verification hops for superlatives
Do NOT add hops like "find the most recent", "find the largest", "find the first". These are post-processing filters.

## Step 1: Reasoning — decompose into hops
Think step by step:
1. What is the starting entity? (MUST be from entity list)
2. What is the FIRST piece of information I need to find? (hop 1)
3. Using that result, what is the NEXT piece of information? (hop 2)
4. Continue until I can answer the question. Count total hops needed.

## Step 2: Write the Chain
Format: EntityA -(single action description)-> node -(single action description)-> node

## Step 3: Endpoint Annotation
Look at the entity list and your chain. If any non-anchor entity from the list constrains the answer
(e.g., a specific country, person, or place the question refers to like "in [Place]" or "near [Location]"),
annotate it below. Skip generic type words like "Country" or "Person".
- Output: Endpoints: [Entity] at hop N   (or "none" if no constraint entities)

## Step 4: Per-Hop Analysis
For each hop, provide:
- Keyword: the core concept word (e.g., year, championship, museum, capital)
- Definition: describe the RELATION SEMANTICS — what kind of link this represents (containment, attribution, membership, temporal). NOT just entity types.
  GOOD: "describes the geographic inclusion of countries within a time zone"
  BAD:  "connects a time zone to the countries that use it"

## Output Format
Reasoning: Starting from [entity]. First need to find [X] (hop 1). Then find [Y] (hop 2). N hops total.

Chain:
EntityA -(action 1)-> node -(action 2)-> node

Endpoints: [Entity] at hop 2   (or "none")

Analysis:
1. [action 1 description]
   - Keyword: word
   - Definition: describes the [relation semantics] of [subject]
2. [action 2 description]
   - Keyword: word
   - Definition: describes the [relation semantics] of [subject]

## Examples

Entities: iPhone
Q: Who is the CEO of the company that makes the iPhone?
Reasoning: Start from iPhone. Hop 1: find manufacturer. Hop 2: find CEO. 2 hops.
iPhone -(manufacturer of this product)-> node -(CEO of this organization)-> node
Endpoints: none
Analysis:
1. manufacturer of this product
   - Keyword: manufacturer
   - Definition: describes the production origin of a product, linking it to the manufacturing organization
2. CEO of this organization
   - Keyword: CEO
   - Definition: describes the top executive leadership role of an organization

Entities: Mona Lisa, France
Q: Which museum in France houses the painting that depicts Lisa Gherardini?
Reasoning: Start from Mona Lisa. Hop 1: find museum. Hop 2: find country of museum. 2 hops.
Mona Lisa -(museum housing this artwork)-> node -(country of this museum)-> node
Endpoints: [France] at hop 2
Analysis:
1. museum housing this artwork
   - Keyword: museum
   - Definition: describes the institutional custody or exhibition location of an artwork
2. country of this museum
   - Keyword: country
   - Definition: describes the national jurisdiction containing a cultural institution

Entities: Manchester United
Q: What year did the football team last win the Champions League?
Reasoning: Start from Manchester United. Hop 1: find championships. Hop 2: find start date. 2 hops. "last win" is filter, not hop.
Manchester United -(championship events of this sports team)-> node -(start date of this event)-> node
Endpoints: none
Analysis:
1. championship events of this sports team
   - Keyword: championship
   - Definition: describes the competitive titles or tournament victories achieved by a sports team
2. start date of this event
   - Keyword: date
   - Definition: describes the temporal origin or occurrence date of an event

## WRONG vs CORRECT
Q: What year did the basketball team coached by Brad Stevens win the championship?
WRONG: Brad Stevens -(championship winning year of his team)-> node
CORRECT: Brad Stevens -(team coached by this person)-> node -(championship events of this team)-> node -(start date of this event)-> node

Q: What year did the football team last win the Champions League?
WRONG: championship events -> most recent event -> start date
CORRECT: championship events -> start date  (filter "most recent" in post-processing)

Now analyze:
Entities: {entities}
Q: {question}
"""


# ═══════════════════════════════════════════════════════════════════
# Parse chain decomposition output
# ═══════════════════════════════════════════════════════════════════

def parse_chain(text):
    text = text.strip()
    text = re.sub(r'^(Answer|答案)[：:]\s*', '', text)

    reasoning = ''
    m = re.search(r'Reasoning:\s*(.+?)(?=\n\s*Chain:|\nChain:)', text, re.DOTALL)
    if m:
        reasoning = m.group(1).strip()

    cm = re.search(r'Chain:\s*\n?(.*?)(?=\n\s*(Analysis:|Endpoints:)|\nAnalysis:|\nEndpoints:|$)', text, re.DOTALL)
    if not cm:
        return None
    chain_lines = [l.strip() for l in cm.group(1).split('\n') if '-(' in l]
    if not chain_lines:
        return None
    chain_text = chain_lines[-1]

    am = re.match(r'^(.+?)\s*-\(', chain_text)
    if not am:
        return None
    anchor = am.group(1).strip()
    rest = chain_text[am.end() - 1:]

    hops = []
    for m in re.finditer(r'\(([^)]+)\)\s*(?:\.inv\s*)?->\s*(.+?)(?=\s*-\(|$)', rest):
        node = m.group(2).strip()
        if not node or node.lower() in ('unknown', 'the', 'a'):
            node = 'node'
        hops.append({'relation': m.group(1).strip(), 'endpoint': node, 'keyword': '', 'definition': ''})

    if not hops:
        return None

    # Parse Endpoints: line — e.g., "Endpoints: [France] at hop 2" or "Endpoints: none"
    endpoint_entities = []
    ep_match = re.search(r'Endpoints:\s*(.+)', text)
    if ep_match:
        ep_text = ep_match.group(1).strip()
        if ep_text.lower() != 'none':
            for em in re.finditer(r'\[([^\]]+)\]\s*(?:at\s*hop\s*(\d+))?', ep_text):
                ep_name = em.group(1).strip()
                hop_num = int(em.group(2)) if em.group(2) else None
                endpoint_entities.append({'entity': ep_name, 'hop': hop_num})

    # Parse analysis
    anm = re.search(r'Analysis:\s*\n(.+)', text, re.DOTALL)
    if anm:
        kw = re.findall(r'-\s*Keyword:\s*(.+)', anm.group(1))
        df = re.findall(r'-\s*Definition:\s*(.+)', anm.group(1))
        for i, hop in enumerate(hops):
            if i < len(kw): hop['keyword'] = kw[i].strip()
            if i < len(df): hop['definition'] = df[i].strip()

    return {'anchor': anchor, 'hops': hops, 'reasoning': reasoning,
            'endpoint_entities': endpoint_entities, 'raw': text}


# ═══════════════════════════════════════════════════════════════════
# Entity resolution: match anchor to subgraph entity index
# ═══════════════════════════════════════════════════════════════════

def resolve_anchor(anchor_name, ents, q_entities):
    """Find anchor entity index in subgraph. Try NER entities first, then all."""
    an = normalize(anchor_name)

    # Pass 1: exact match against NER entities
    for i, e in enumerate(ents):
        if normalize(e) == an and not is_cvt_like(e):
            return i

    # Pass 2: substring match against NER entities
    for qe in q_entities:
        qn = normalize(qe)
        if qn in an or an in qn:
            for i, e in enumerate(ents):
                if normalize(e) == qn and not is_cvt_like(e):
                    return i

    # Pass 3: substring match against all entities
    best = None
    for i, e in enumerate(ents):
        en = normalize(e)
        if (an in en or en in an) and not is_cvt_like(e) and len(an) >= 3:
            if best is None or len(e) > len(ents[best]):
                best = i

    return best


def resolve_endpoints(endpoint_entities, ents, q_entities):
    """Resolve endpoint entities from 'Endpoints:' line to entity indices."""
    if not endpoint_entities:
        return []

    targets = []
    for ep_info in endpoint_entities:
        ep = ep_info['entity']
        en = normalize(ep)
        if len(en) < 2:
            continue
        # Pass 1: exact match
        idx = None
        for i, e in enumerate(ents):
            if normalize(e) == en and not is_cvt_like(e):
                idx = i
                break
        # Pass 2: match via NER entity
        if idx is None:
            for qe in q_entities:
                qn = normalize(qe)
                if qn == en or (min(len(qn), len(en)) >= 4 and (qn in en or en in qn)):
                    for i, e in enumerate(ents):
                        if normalize(e) == qn and not is_cvt_like(e):
                            idx = i
                            break
                    if idx is not None:
                        break
        # Pass 3: substring match against all entities
        if idx is None:
            for i, e in enumerate(ents):
                en2 = normalize(e)
                if not is_cvt_like(e) and min(len(en), len(en2)) >= 4 and (en in en2 or en2 in en):
                    idx = i
                    break
        if idx is not None:
            targets.append(idx)
    return targets


# ═══════════════════════════════════════════════════════════════════
# GTE encoding
# ═══════════════════════════════════════════════════════════════════

def make_gte_encode():
    import torch, torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    print("Loading Qwen3-Embedding-0.6B...")
    tokenizer = AutoTokenizer.from_pretrained('/zhaoshu/llm/Qwen3-Embedding-0.6B', padding_side='left', trust_remote_code=True)
    model = AutoModel.from_pretrained('/zhaoshu/llm/Qwen3-Embedding-0.6B', trust_remote_code=True).cuda().eval()

    def last_token_pool(lhs, mask):
        if mask[:, -1].sum() == mask.shape[0]:
            return lhs[:, -1]
        seq_lens = mask.sum(dim=1) - 1
        return lhs[torch.arange(lhs.shape[0], device=lhs.device), seq_lens]

    def encode(texts, bs=64):
        all_e = []
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            bd = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to('cuda')
            with torch.no_grad():
                out = model(**bd)
                embs = last_token_pool(out.last_hidden_state, bd['attention_mask'])
            all_e.append(F.normalize(embs, p=2, dim=1))
        return torch.cat(all_e, dim=0)

    return encode


# ═══════════════════════════════════════════════════════════════════
# LLM pruning
# ═══════════════════════════════════════════════════════════════════

def llm_prune(question, hops, step_candidates, rel_names):
    """LLM selects relevant relations per step from GTE candidates.

    Args:
        question: the question text
        hops: list of hop dicts with 'relation', 'keyword', 'definition'
        step_candidates: dict mapping step_num -> list of (rel_idx, rel_name, score)
        rel_names: list of all relation names

    Returns:
        list of lists: selected relation indices per step
    """
    chain_lines = []
    step_blocks = []
    for hi, hop in enumerate(hops):
        sn = hi + 1
        ep = hop.get('endpoint', 'node')
        ep_str = f" -> {ep}" if ep != 'node' else ''
        df = hop.get('definition', '')
        chain_lines.append(f"  Step {sn}: {hop['relation']}{ep_str}")
        cands = step_candidates.get(sn, [])
        if not cands:
            step_blocks.append(f"Step {sn}: {hop['relation']}\n  Purpose: {df}\n  Candidates: (none)")
            continue
        cand_lines = [f"    {i}. {name}" for i, (idx, name, score) in enumerate(cands, 1)]
        step_blocks.append(
            f"Step {sn}: {hop['relation']}\n  Purpose: {df}\n  Candidates:\n" + "\n".join(cand_lines))

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
2. Select up to 5 relevant relations per step. Fewer is better if only 1-3 are truly relevant.
3. Ignore unrelated attributes
4. If no relations fit a step, output empty list
5. ORDER matters: rank by relevance to the step (most relevant first)

Output format:
<analysis>
One sentence per step: what it needs and which relations fit.
</analysis>
<selected>
step_1: [3, 1]
step_2: [5, 2]
</selected>

Numbers are RANKED: first = most relevant."""

    messages = [
        {"role": "system", "content": "You are a knowledge graph relation selector for multi-step QA. Analyze the full chain, then select relevant relations per step. Output <analysis> and <selected> XML tags."},
        {"role": "user", "content": prompt},
    ]

    raw = call_llm(messages, max_tokens=2000)
    selected_yaml = extract_xml_tag(raw or "", "selected")
    result = {}

    if selected_yaml:
        for line in selected_yaml.split('\n'):
            line = line.strip()
            m = re.match(r'step_(\d+)\s*:\s*\[(.*?)\]', line)
            if m:
                sn = int(m.group(1))
                nums = [int(x.strip()) for x in m.group(2).split(',') if x.strip().isdigit()]
                cands = step_candidates.get(sn, [])
                ranked = []
                seen = set()
                for n in nums:
                    if 1 <= n <= len(cands):
                        idx = cands[n - 1][0]
                        if idx not in seen:
                            ranked.append(idx)
                            seen.add(idx)
                result[sn] = ranked

    # Fallback: top-3 GTE for missing steps
    step_relations = []
    for hi in range(len(hops)):
        sn = hi + 1
        ranked = result.get(sn, [])
        if not ranked:
            cands = step_candidates.get(sn, [])
            ranked = [idx for idx, _, _ in cands[:3]]
        step_relations.append(ranked[:5])

    return step_relations


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def strict_candidate_hit(cands, targets):
    """Strict matching: normalize both sides, require exact match or
    one is a meaningful substring of the other (min length 4 to avoid
    false positives like 'St' matching 'State')."""
    norm_cands = [normalize(c) for c in cands]
    for t in targets:
        nt = normalize(t)
        if len(nt) < 2:
            continue  # skip empty/trivial normalized strings
        for c in norm_cands:
            if len(c) < 2:
                continue
            if c == nt:
                return True
            # Substring allowed only if shorter string >= 4 chars
            shorter = min(len(c), len(nt))
            if shorter >= 4 and (nt in c or c in nt):
                return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=30)
    args = parser.parse_args()

    # Load data
    pkl_map = {d['id']: d for d in pickle.load(open('/zhaoshu/subgraph/data/cwq_processed/test_literal_and_language_fixed.pkl', 'rb'))}
    sparql_map = {d['ID']: d for d in json.load(open('/zhaoshu/subgraph/data/cwq_sparql/test.json'))}
    pilot = json.load(open('/zhaoshu/subgraph/reports/stage_pipeline_test/cwq_50_stage_v7/pilot.json'))

    encode = make_gte_encode()
    task_desc = "Given a knowledge graph question, retrieve relevant graph relations that answer the question"

    stats = {
        'total': 0, 'decomp_ok': 0, 'anchor_ok': 0,
        'gte_hit': 0, 'prune_hit': 0, 'traverse_hit': 0,
    }

    for i in range(min(args.limit, len(pilot))):
        case = pilot[i]
        cid = case['case_id']
        sample = pkl_map.get(cid)
        sq = sparql_map.get(cid)
        if not sample or not sq:
            continue

        stats['total'] += 1
        question = sample.get('question', '')
        q_entities = sample.get('q_entity', [])
        gt_answers = sample.get('a_entity', [])
        ent_str = ", ".join(q_entities) if q_entities else "none"

        # ── Stage 1: Chain Decomposition ──
        prompt = CHAIN_PROMPT.format(entities=ent_str, question=question)
        try:
            raw = call_llm([{"role": "user", "content": prompt}])
        except Exception as e:
            print(f"  Case {i}: LLM error: {e}")
            continue

        parsed = parse_chain(raw)
        if not parsed or not parsed['hops']:
            print(f"  Case {i}: DECOMP FAIL")
            continue
        stats['decomp_ok'] += 1

        anchor = parsed['anchor']
        hops = parsed['hops']
        n_hops = len(hops)

        # ── Stage 2: Entity Resolution ──
        ents = sample.get('text_entity_list', []) + sample.get('non_text_entity_list', [])
        rels = list(sample.get('relation_list', []))
        h_ids = sample.get('h_id_list', [])
        r_ids = sample.get('r_id_list', [])
        t_ids = sample.get('t_id_list', [])

        if not ents or not rels or not h_ids:
            print(f"  Case {i}: NO GRAPH DATA")
            continue

        anchor_idx = resolve_anchor(anchor, ents, q_entities)
        if anchor_idx is None:
            print(f"  Case {i} ({n_hops}h): ANCHOR NOT FOUND '{anchor}'")
            continue
        stats['anchor_ok'] += 1

        anchor_name = ents[anchor_idx]
        hops_str = " -> ".join(f"({h['relation']})->{h['endpoint']}" for h in hops)

        # ── Stage 3: GTE Retrieval (using definition) ──
        r_embs = encode(rels, bs=64)
        step_candidates = {}
        gte_hit = False

        for hi, hop in enumerate(hops):
            sn = hi + 1
            df = hop.get('definition', '')
            rel_desc = hop['relation']

            # Use definition as primary query, description as fallback
            query_text = df if df else rel_desc
            q_text = f'Instruct: {task_desc}\nQuery: {query_text}'
            q_emb = encode([q_text], bs=1)
            scores = (r_embs @ q_emb.T).squeeze(1).cpu().tolist()
            ranked = sorted(enumerate(rels), key=lambda x: -scores[x[0]])

            # Top-10 candidates
            step_candidates[sn] = [(idx, name, scores[idx]) for idx, name in ranked[:10]]

            # Check GTE recall (is any gold relation in top-10?)
            sparql_rels = [r for r in re.findall(r'ns:([a-zA-Z][a-zA-Z0-9_.]+)', sq.get('sparql', ''))
                          if '.' in r and not r.startswith('m.')]

        # Check if any gold relation is in GTE top-10 for any step
        sparql_rels = list(dict.fromkeys(
            r for r in re.findall(r'ns:([a-zA-Z][a-zA-Z0-9_.]+)', sq.get('sparql', ''))
            if '.' in r and not r.startswith('m.')
        ))
        for sr in sparql_rels:
            sn = normalize(sr)
            if len(sn) < 2:
                continue
            for sn_step, cands in step_candidates.items():
                for _, cn, _ in cands:
                    cn_norm = normalize(cn)
                    if len(cn_norm) < 2:
                        continue
                    if sn == cn_norm:
                        gte_hit = True
                        break
                    shorter = min(len(sn), len(cn_norm))
                    if shorter >= 4 and (sn in cn_norm or cn_norm in sn):
                        gte_hit = True
                        break
                if gte_hit:
                    break
            if gte_hit:
                break
        if gte_hit:
            stats['gte_hit'] += 1

        # ── Stage 4: LLM Pruning ──
        step_relations = llm_prune(question, hops, step_candidates, rels)

        # Check if any gold relation survives pruning
        prune_hit = False
        for sr in sparql_rels:
            sn = normalize(sr)
            if len(sn) < 2:
                continue
            for step_rels in step_relations:
                for ri in step_rels:
                    rn = normalize(rels[ri])
                    if len(rn) < 2:
                        continue
                    if sn == rn:
                        prune_hit = True
                        break
                    shorter = min(len(sn), len(rn))
                    if shorter >= 4 and (sn in rn or rn in sn):
                        prune_hit = True
                        break
                if prune_hit:
                    break
            if prune_hit:
                break
        if prune_hit:
            stats['prune_hit'] += 1

        # ── Stage 5: Graph Traversal (using main pipeline functions) ──
        # Resolve [Entity] endpoints to guide traversal
        ner_targets = resolve_endpoints(parsed.get('endpoint_entities', []), ents, q_entities)
        explicit_targets = ner_targets if ner_targets else None
        target_names = [ents[t] for t in ner_targets if 0 <= t < len(ents)]

        # Build steps format expected by frontier_expand_layers
        frontier_steps = []
        for hi, hop in enumerate(hops):
            frontier_steps.append({
                "step": hi + 1,
                "question": hop['relation'],
                "relation_query": hop.get('definition', hop['relation']),
            })

        n_steps = len(hops)
        if n_steps <= 1:
            # Single step: frontier + relation_prior, merge
            paths, max_depth, max_cov = frontier_expand_layers(
                anchor_idx, step_relations, frontier_steps,
                h_ids, r_ids, t_ids, ents)
            rpe_paths, rpe_depth, rpe_cov = relation_prior_expand(
                anchor_idx, [set(rs) for rs in step_relations],
                h_ids, r_ids, t_ids, ents,
                explicit_targets=explicit_targets)
            if rpe_cov > max_cov:
                paths, max_depth, max_cov = rpe_paths, rpe_depth, rpe_cov
            elif rpe_paths:
                existing = {(tuple(p["relations"][:3]), p["nodes"][-1]) for p in paths}
                for rp in rpe_paths:
                    sig = (tuple(rp["relations"][:3]), rp["nodes"][-1])
                    if sig not in existing:
                        paths.append(rp)
        else:
            # Multi-step: frontier first, fallback to relation_prior
            paths, max_depth, max_cov = frontier_expand_layers(
                anchor_idx, step_relations, frontier_steps,
                h_ids, r_ids, t_ids, ents)
            if max_cov < n_steps:
                rpe_paths, rpe_depth, rpe_cov = relation_prior_expand(
                    anchor_idx, [set(rs) for rs in step_relations],
                    h_ids, r_ids, t_ids, ents,
                    explicit_targets=explicit_targets)
                if rpe_cov > max_cov:
                    paths, max_depth, max_cov = rpe_paths, rpe_depth, rpe_cov
                elif rpe_paths:
                    existing = {(tuple(p["relations"][:3]), p["nodes"][-1]) for p in paths}
                    for rp in rpe_paths:
                        sig = (tuple(rp["relations"][:3]), rp["nodes"][-1])
                        if sig not in existing:
                            paths.append(rp)

        # Collect answer candidates from paths
        all_nodes = {anchor_idx}
        for path in paths:
            all_nodes.update(path["nodes"])

        # Also collect HR frontier nodes
        expanded_rels = [set(rs) for rs in step_relations]
        for node_idx in range(len(ents)):
            if node_idx == anchor_idx:
                continue
            for i in range(len(h_ids)):
                if (h_ids[i] == node_idx and r_ids[i] in set().union(*expanded_rels)) or \
                   (t_ids[i] == node_idx and r_ids[i] in set().union(*expanded_rels)):
                    all_nodes.add(node_idx)
                    break

        answer_candidates = []
        for node_idx in sorted(all_nodes):
            if node_idx == anchor_idx:
                continue
            name = ents[node_idx] if 0 <= node_idx < len(ents) else ""
            if is_cvt_like(name):
                for cvt_idx, _ in expand_through_cvt(node_idx, h_ids, r_ids, t_ids, ents):
                    if cvt_idx != anchor_idx and 0 <= cvt_idx < len(ents) and not is_cvt_like(ents[cvt_idx]):
                        answer_candidates.append(ents[cvt_idx])
            else:
                answer_candidates.append(name)

        traverse_hit = strict_candidate_hit(answer_candidates, gt_answers) if answer_candidates else False
        if traverse_hit:
            stats['traverse_hit'] += 1

        # Print result
        n_paths = len(paths)
        mark = "✅" if traverse_hit else "❌"
        print(f"  Case {i} ({n_hops}h) [{mark}] anchor={anchor_name[:20]} | "
              f"paths={n_paths} cov={max_cov}/{n_steps} | cands={len(answer_candidates)} | "
              f"targets={target_names if ner_targets else []} | "
              f"GT={gt_answers[:1]}")

        # Detailed trace for failures
        if not traverse_hit:
            print(f"\n  {'='*60}")
            print(f"  FAIL TRACE — Case {i}")
            print(f"  Q: {question}")
            print(f"  Gold: {gt_answers}")
            print(f"  NER: {q_entities}")
            print(f"  Anchor: '{anchor}' -> idx={anchor_idx} ('{anchor_name}')")
            print(f"  CoT: {parsed.get('reasoning', 'N/A')}")
            print(f"  Chain: {hops_str}")
            print(f"  SPARQL rels: {sparql_rels}")
            print(f"  Steps & pruning:")
            for hi, hop in enumerate(hops):
                sn = hi + 1
                kw = hop.get('keyword', '')
                df = hop.get('definition', '')
                sel = step_relations[hi]
                sel_names = [rels[ri] for ri in sel if ri < len(rels)]
                cands = step_candidates.get(sn, [])
                top5 = [name for _, name, _ in cands[:5]]
                print(f"    Step {sn}: {hop['relation']} -> {hop['endpoint']}")
                print(f"      kw={kw} def={df[:60]}")
                print(f"      GTE top-5: {top5}")
                print(f"      Pruned ({len(sel)}): {sel_names}")
            print(f"  Paths: {n_paths}, cov={max_cov}/{n_steps}")
            if paths:
                for pi, p in enumerate(paths[:5]):
                    p_ents = [ents[n] if 0 <= n < len(ents) else f'idx={n}' for n in p['nodes']]
                    p_rels = [rels[r] if 0 <= r < len(rels) else f'idx={r}' for r in p['relations']]
                    print(f"    Path {pi}: {' --'.join([f'{p_ents[0]}'] + [f'[{pr}]> {pe}' for pr, pe in zip(p_rels, p_ents[1:])])}")
            print(f"  Answer candidates (first 20): {answer_candidates[:20]}")
            print(f"  {'='*60}\n")
        if not traverse_hit and prune_hit:
            print(f"    ⚠️  Prune hit but traverse miss — rels per step: {[len(sr) for sr in step_relations]}")

        time.sleep(0.3)

    # ── Summary ──
    t = stats['total']
    print(f"\n{'='*70}")
    print(f"End-to-End Pipeline Test ({t} cases)")
    print(f"  Decomposition:  {stats['decomp_ok']}/{t} ({stats['decomp_ok']/t*100:.0f}%)")
    print(f"  Anchor resolve: {stats['anchor_ok']}/{t} ({stats['anchor_ok']/t*100:.0f}%)")
    print(f"  GTE recall:     {stats['gte_hit']}/{t} ({stats['gte_hit']/t*100:.0f}%)")
    print(f"  LLM prune:      {stats['prune_hit']}/{t} ({stats['prune_hit']/t*100:.0f}%)")
    print(f"  Traverse hit:   {stats['traverse_hit']}/{t} ({stats['traverse_hit']/t*100:.0f}%)")

    if stats['prune_hit'] > 0:
        print(f"\n  Prune→Traverse conversion: {stats['traverse_hit']}/{stats['prune_hit']} ({stats['traverse_hit']/stats['prune_hit']*100:.0f}%)")


if __name__ == '__main__':
    main()
