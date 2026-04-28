#!/usr/bin/env python3
"""Test decomposition + GTE retrieval with new definition prompt + keyword query.

Reuses Stage 1a from v7 results, re-runs Stage 1b with updated CHAIN_PROMPT,
then tests GTE retrieval with 3 queries (definition, question, keyword).
Compares against golden SPARQL relations.
"""

import asyncio
import aiohttp
import json
import re
import pickle
import time
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

# ── Config ──
LLM_URL = "http://localhost:8000/v1/chat/completions"
LLM_MODEL = "qwen35-9b-local"
GTE_URL = "http://localhost:8003"
GTE_TOP_K = 10

V7_RESULTS = "/zhaoshu/subgraph/reports/stage_pipeline_test/prompt_tune_27b_v7_reselect/results.json"
TEST_DATA = "/zhaoshu/subgraph/data/cwq_processed/test_literal_and_language_fixed.pkl"
SPARQL_DATA = "/zhaoshu/subgraph/data/cwq_sparql/test.json"
OUTPUT_DIR = "/zhaoshu/subgraph/reports/gte_retrieval_v2"

# ── Updated CHAIN_PROMPT (with new definition instruction) ──
CHAIN_PROMPT = """Entities: {entities}
Anchor: {anchor}
Endpoints: {endpoints}
Answer type: {answer_type}
Interpretation: {interpretation}
Rewritten question: {rewritten}
Original question: {question}

Decompose this question into a chain of abstract relation hops.
Each hop describes WHAT kind of graph edge to look for — NOT which specific entity it leads to.
Think first, then write the chain. Do NOT write the chain before doing the per-hop analysis.

IMPORTANT RULES:
1. This is QUESTION DECOMPOSITION, not graph traversal. You do NOT know what entities exist in the graph.
2. Intermediate nodes MUST be written as literally "node". NEVER fill in guessed entity names.
3. Each hop = ONE atomic relation. Compound actions MUST be split into separate hops.
4. No verification hops for superlatives ("find the most recent" → post-processing, not a hop).
5. No circular chains.
6. Endpoints are entities that must be reached or constrain the path.
7. Count hops carefully: each "that/which/who/where" clause typically adds ONE hop. Do NOT merge clauses.
8. "Return the answer", "the result is", "verify it", or "implicit return" is NOT a hop.
9. A constraint entity should appear only at the hop where it is reached or checked.
10. For "X that/which contains/has an airport that serves Y", start from Y with:
    Y -(served by airport)-> node -(airport located in country/region)-> node.
11. For "country/region with [endpoint constraint]", first find the answer node, then add one hop from that
    answer node to the endpoint constraint. Do NOT make the endpoint itself the answer.

Examples:
  Question: "What is the capital of the country where the Eiffel Tower is located?"
  Entities: Eiffel Tower
  Anchor: Eiffel Tower
  Chain: Eiffel Tower -(located in country)-> node -(capital city)-> node  [2 hops]

  Question: "What language is spoken in the country where the leader was appointed to office?"
  Entities: Leader
  Anchor: Leader
  Chain: Leader -(government position held)-> node -(jurisdiction of office)-> node -(language spoken)-> node  [3 hops]

  Question: "What sport does the most popular team in the country containing Paris play?"
  Entities: Paris
  Anchor: Paris
  Chain: Paris -(contained by country)-> node -(popular sport)-> node  [2 hops, "most popular" is post-processing]

  Question: "What country bordering a given country contains an airport that serves a given city?"
  Entities: [city] | [border country]
  Anchor: [city]
  Endpoints: [border country]
  Chain: [city] -(served by airport)-> node -(airport located in country)-> node -(borders country)-> node  [3 hops; endpoint is reached only by the border hop]

  Question: "Which region with a specified time zone contains a given country?"
  Entities: [country] | [time zone]
  Anchor: [country]
  Endpoints: [time zone]
  Chain: [country] -(contained in region)-> node -(has time zone)-> node  [2 hops; answer is the region node, endpoint is the time zone constraint]

WRONG Chain: Leader -(appointed to role in country where language is spoken)-> node  ← merged 3 hops into 1!
WRONG Chain: [city] -(located in country)-> node -(borders endpoint)-> node -(located in country)-> node  ← final "return" hop is fake and creates a loop.
WRONG Chain: [country] -(has time zone)-> node  ← answers the endpoint constraint, not the requested region.

Output (exact format — think first, chain last):
Reasoning: [1-3 short sentences. Identify the answer node and which endpoint constraints must be reached.]

Analysis:
1. [action description]
   - Keyword: core concept word
   - Definition: A concise dictionary-style definition of the relation concept in 15 words or less. Describe what the relation MEANS naturally. Do NOT use graph terminology (no "link", "node", "entity", "edge"). Examples: "neighboring countries sharing a common border", "the sport an athlete plays professionally", "a country's top-level administrative regions", "the official song representing a nation's identity".
   (repeat for each hop)

Chain:
{anchor} -(action description)-> node ... -(final action)-> node"""


# ── SPARQL Golden Relation Parsing ──

def parse_sparql_chain(sparql: str) -> List[Dict]:
    """Parse SPARQL into ordered triples following variable chains."""
    triples = []
    for m in re.finditer(
        r'(\?\w+|ns:\S+)\s+(ns:\S+)\s+(\?\w+|ns:\S+)',
        sparql
    ):
        s, p, o = m.group(1), m.group(2), m.group(3)
        # Skip type/filter triples
        if '.type' in p or p.endswith('.type'):
            continue
        # Only keep triples with ns: predicates (relations)
        if p.startswith('ns:'):
            triples.append({'s': s, 'p': p.replace('ns:', ''), 'o': o})

    if not triples:
        return []

    # Build variable propagation chain
    var_first = {}
    for i, t in enumerate(triples):
        for v in [t['s'], t['o']]:
            if v.startswith('?') and v not in var_first:
                var_first[v] = i

    # Follow chain from first triple
    chain = [triples[0]]
    current_var = triples[0]['o'] if triples[0]['o'].startswith('?') else None
    used = {0}

    for _ in range(len(triples)):
        if not current_var:
            break
        for i, t in enumerate(triples):
            if i in used:
                continue
            if t['s'] == current_var:
                chain.append(t)
                used.add(i)
                current_var = t['o'] if t['o'].startswith('?') else None
                break
            elif t['o'] == current_var:
                chain.append({'s': t['o'], 'p': t['p'], 'o': t['s']})
                used.add(i)
                current_var = t['s'] if t['s'].startswith('?') else None
                break
        else:
            break

    return chain


# ── Chain output parsing ──

def parse_chain(text: str) -> Optional[Dict]:
    text = text.strip()
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
        hops.append({'relation': m.group(1).strip(), 'keyword': '', 'definition': ''})

    if not hops:
        return None

    anm = re.search(r'Analysis:\s*\n(.+)', text, re.DOTALL)
    if anm:
        kw = re.findall(r'-\s*Keyword:\s*(.+)', anm.group(1))
        df = re.findall(r'-\s*Definition:\s*(.+)', anm.group(1))
        for i, hop in enumerate(hops):
            if i < len(kw): hop['keyword'] = kw[i].strip()
            if i < len(df): hop['definition'] = df[i].strip()

    return {'anchor': anchor, 'hops': hops}


# ── Candidate text formatting ──

def rel_to_text_fullshort(rel: str) -> str:
    """Full domain + short property for GTE: 'location contains'"""
    parts = rel.split(".")
    if len(parts) >= 2:
        domain = parts[0].replace("_", " ")
        short = " ".join(p.replace("_", " ") for p in parts[-2:])
        return f"{domain} {short}"
    return rel.replace("_", " ")


# ── API calls ──

async def call_llm(session, messages, temperature=0.3):
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1500,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with session.post(LLM_URL, json=payload) as resp:
        r = await resp.json()
        return r['choices'][0]['message']['content']


async def gte_retrieve(session, query, candidates, candidate_texts=None, top_k=10):
    payload = {
        "query": query,
        "candidates": candidates,
        "top_k": top_k,
    }
    if candidate_texts:
        payload["candidate_texts"] = candidate_texts
    async with session.post(f"{GTE_URL}/retrieve", json=payload) as resp:
        r = await resp.json()
        return r.get('results', [])


# ── Main test ──

async def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    with open(V7_RESULTS) as f:
        v7_results = json.load(f)
    with open(TEST_DATA, 'rb') as f:
        test_data = pickle.load(f)
    with open(SPARQL_DATA) as f:
        sparql_data = json.load(f)

    # Build lookup
    data_by_id = {d['id']: d for d in test_data}
    sparql_by_id = {}
    for s in sparql_data:
        sid = s.get('ID', '')
        # Normalize ID format
        sparql_by_id[sid] = s.get('sparql', '') or s.get('SPARQL', '')

    print(f"Loaded: {len(v7_results)} v7 cases, {len(test_data)} test data, {len(sparql_data)} SPARQL")

    results = []
    sem = asyncio.Semaphore(5)

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as http_session:
        async def process_case(idx, case):
            case_id = case['case_id']
            question = case['question']
            q_entities = data_by_id.get(case_id, {}).get('q_entity', [])

            if not q_entities:
                return None

            # Reuse Stage 1a results from v7
            anchor = case.get('stage_1a_anchor', '') or case.get('anchor_name', '')
            endpoints = case.get('stage_1a_endpoints', 'none') or 'none'
            interpretation = case.get('stage_1a_interpretation', '')
            rewritten = case.get('stage_1a_rewritten', '') or question
            answer_type = case.get('stage_1a_answer_type', 'other')

            if not anchor:
                return None

            ent_str = "\n".join(f"- {e}" for e in q_entities)

            # Run Stage 1b with NEW prompt
            prompt_text = CHAIN_PROMPT.format(
                entities=ent_str, anchor=anchor, endpoints=endpoints,
                answer_type=answer_type, interpretation=interpretation,
                rewritten=rewritten, question=question
            )

            try:
                async with sem:
                    raw = await call_llm(http_session, [{"role": "user", "content": prompt_text}])
            except Exception as e:
                return {"case_id": case_id, "error": f"LLM call failed: {e}"}

            parsed = parse_chain(raw)
            if not parsed or not parsed.get('hops'):
                return {"case_id": case_id, "question": question, "error": "parse failed", "raw": raw[:500]}

            # Get golden relations from SPARQL
            sparql = sparql_by_id.get(case_id, '')
            golden_chain = parse_sparql_chain(sparql) if sparql else []

            # Get subgraph relations
            subgraph = data_by_id.get(case_id, {})
            rels = subgraph.get('relation_list', [])
            rel_texts_fullshort = [rel_to_text_fullshort(r) for r in rels]

            # For each step, run GTE with 3 queries
            step_results = []
            for i, hop in enumerate(parsed['hops']):
                golden_rel = golden_chain[i]['p'] if i < len(golden_chain) else None

                # Build queries (deduplicated)
                queries = {}
                for field_name in ['definition', 'question', 'keyword', 'relation_query']:
                    val = hop.get(field_name, '')
                    if field_name == 'question':
                        val = hop.get('relation', '')  # chain action description
                    if val and val.strip():
                        queries[field_name] = val.strip()

                # Run GTE for each query
                query_results = {}
                for qname, qtext in queries.items():
                    try:
                        async with sem:
                            rows = await gte_retrieve(
                                http_session, qtext, rels,
                                candidate_texts=rel_texts_fullshort, top_k=GTE_TOP_K
                            )
                        query_results[qname] = rows
                    except Exception as e:
                        query_results[qname] = []

                # Find golden relation rank for each query
                golden_rank = {}
                for qname, rows in query_results.items():
                    rank = None
                    for j, r in enumerate(rows):
                        cand = r.get('candidate', '')
                        if golden_rel and cand == golden_rel:
                            rank = j + 1
                            break
                    golden_rank[qname] = rank

                # Best rank across all queries (oracle merge)
                best_rank = None
                best_query = None
                for qname, rank in golden_rank.items():
                    if rank is not None:
                        if best_rank is None or rank < best_rank:
                            best_rank = rank
                            best_query = qname

                # Also check if golden relation exists in subgraph at all
                golden_in_subgraph = golden_rel in rels if golden_rel else False

                step_results.append({
                    "step": i + 1,
                    "hop_relation": hop.get('relation', ''),
                    "keyword": hop.get('keyword', ''),
                    "definition": hop.get('definition', ''),
                    "golden_rel": golden_rel,
                    "golden_in_subgraph": golden_in_subgraph,
                    "golden_rank": golden_rank,
                    "best_rank": best_rank,
                    "best_query": best_query,
                })

            return {
                "case_id": case_id,
                "question": question,
                "anchor": anchor,
                "hops": step_results,
                "golden_chain_len": len(golden_chain),
                "parsed_hops_len": len(parsed['hops']),
            }

        # Process all cases
        tasks = [process_case(i, c) for i, c in enumerate(v7_results)]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter valid results
    valid = [r for r in all_results if r and not isinstance(r, Exception) and 'error' not in r]
    errors = [r for r in all_results if isinstance(r, Exception) or (isinstance(r, dict) and 'error' in r)]

    print(f"\n{'='*80}")
    print(f"RESULTS: {len(valid)} valid, {len(errors)} errors out of {len(v7_results)} cases")
    print(f"{'='*80}")

    # ── Aggregate metrics ──
    # Collect all step-rel pairs where golden relation is in subgraph
    all_pairs = []
    for r in valid:
        for hop in r['hops']:
            if hop['golden_rel'] and hop['golden_in_subgraph']:
                all_pairs.append(hop)

    total = len(all_pairs)

    # Print first few errors for debugging
    if errors:
        print(f"\nFirst 3 errors:")
        for e in errors[:3]:
            if isinstance(e, dict):
                print(f"  {e.get('case_id','?')}: {e.get('error','?')[:200]}")
            else:
                print(f"  {str(e)[:200]}")

    if total == 0:
        print(f"\nNo valid step-rel pairs found. Valid cases: {len(valid)}, Errors: {len(errors)}")
        # Still save what we have
        with open(f"{OUTPUT_DIR}/detailed_results.json", 'w') as f:
            json.dump({"valid": valid, "errors": [str(e) for e in errors]}, f, ensure_ascii=False, indent=2)
        return

    print(f"\nStep-rel pairs with golden in subgraph: {total}")

    # Per-query metrics
    for qname in ['definition', 'keyword', 'question']:
        found = sum(1 for h in all_pairs if h['golden_rank'].get(qname) is not None)
        r1 = sum(1 for h in all_pairs if h['golden_rank'].get(qname) == 1)
        top5 = sum(1 for h in all_pairs if h['golden_rank'].get(qname) is not None and h['golden_rank'][qname] <= 5)
        top10 = sum(1 for h in all_pairs if h['golden_rank'].get(qname) is not None and h['golden_rank'][qname] <= 10)
        not_found = total - found
        print(f"\n  {qname}:")
        print(f"    Rank #1: {r1} ({100*r1/total:.1f}%)")
        print(f"    Top-5:   {top5} ({100*top5/total:.1f}%)")
        print(f"    Top-10:  {top10} ({100*top10/total:.1f}%)")
        print(f"    NOT FOUND: {not_found} ({100*not_found/total:.1f}%)")

    # Oracle merge (best across all queries)
    oracle_r1 = sum(1 for h in all_pairs if h['best_rank'] == 1)
    oracle_top5 = sum(1 for h in all_pairs if h['best_rank'] is not None and h['best_rank'] <= 5)
    oracle_top10 = sum(1 for h in all_pairs if h['best_rank'] is not None and h['best_rank'] <= 10)
    oracle_not_found = sum(1 for h in all_pairs if h['best_rank'] is None)
    print(f"\n  Oracle merge (best of all queries):")
    print(f"    Rank #1: {oracle_r1} ({100*oracle_r1/total:.1f}%)")
    print(f"    Top-5:   {oracle_top5} ({100*oracle_top5/total:.1f}%)")
    print(f"    Top-10:  {oracle_top10} ({100*oracle_top10/total:.1f}%)")
    print(f"    NOT FOUND: {oracle_not_found} ({100*oracle_not_found/total:.1f}%)")

    # Show cases where keyword uniquely finds the golden relation
    keyword_unique = []
    for h in all_pairs:
        kw_rank = h['golden_rank'].get('keyword')
        def_rank = h['golden_rank'].get('definition')
        q_rank = h['golden_rank'].get('question')
        if kw_rank is not None and (kw_rank <= 5):
            others = [def_rank, q_rank]
            others_in_top5 = [r for r in others if r is not None and r <= 5]
            if not others_in_top5:
                keyword_unique.append(h)

    print(f"\n  Keyword uniquely in Top-5 (others miss): {len(keyword_unique)}")
    for h in keyword_unique[:10]:
        print(f"    {h['golden_rel']}: keyword={h['golden_rank']['keyword']}, "
              f"def={h['golden_rank'].get('definition')}, q={h['golden_rank'].get('question')}")

    # Show cases where new definition beats old
    def_found = sum(1 for h in all_pairs if h['golden_rank'].get('definition') is not None)
    print(f"\n  Definition found: {def_found}/{total} ({100*def_found/total:.1f}%)")

    # Save detailed results
    with open(f"{OUTPUT_DIR}/detailed_results.json", 'w') as f:
        json.dump({
            "valid": valid,
            "errors": [str(e) for e in errors],
            "summary": {
                "total_pairs": total,
                "per_query": {
                    qname: {
                        "rank1": sum(1 for h in all_pairs if h['golden_rank'].get(qname) == 1),
                        "top5": sum(1 for h in all_pairs if h['golden_rank'].get(qname) is not None and h['golden_rank'][qname] <= 5),
                        "top10": sum(1 for h in all_pairs if h['golden_rank'].get(qname) is not None and h['golden_rank'][qname] <= 10),
                        "not_found": total - sum(1 for h in all_pairs if h['golden_rank'].get(qname) is not None),
                    }
                    for qname in ['definition', 'keyword', 'question']
                },
                "oracle_merge": {
                    "rank1": oracle_r1, "top5": oracle_top5,
                    "top10": oracle_top10, "not_found": oracle_not_found,
                },
                "keyword_unique_top5": len(keyword_unique),
            }
        }, f, ensure_ascii=False, indent=2)

    print(f"\nDetailed results saved to {OUTPUT_DIR}/detailed_results.json")


if __name__ == "__main__":
    asyncio.run(main())
