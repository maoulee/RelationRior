#!/usr/bin/env python3
"""GTE recall using chain decomposition relation descriptions.

Compare: chain relation desc (Chinese) vs SQ vs RQ vs OQ.
Uses annotated step->gold mapping for fair evaluation.
"""

import json, re, pickle
from pathlib import Path

def norm(t):
    return re.sub(r'[^a-z0-9]', '', t.lower().strip())

def find_gold_in_list(gold_rel, rel_list):
    gn = norm(gold_rel)
    for r in rel_list:
        rn = norm(r)
        if gn == rn or gn in rn or rn in gn:
            return r
    return None

def last_token_pool(last_hidden_states, attention_mask):
    import torch
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def rank_icon(rank):
    if rank <= 5: return "✅"
    elif rank <= 10: return "⚠️"
    else: return "❌"


def main():
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    # Load data
    pkl_map = {d['id']: d for d in pickle.load(open('/zhaoshu/subgraph/data/cwq_processed/test_fixed_relaxation.pkl', 'rb'))}
    sparql_map = {d['ID']: d for d in json.load(open('/zhaoshu/subgraph/data/cwq_sparql/test.json'))}
    pilot = json.load(open('/zhaoshu/subgraph/reports/stage_pipeline_test/cwq_50_stage_v7/pilot.json'))
    chain_results = json.load(open('/tmp/chain_decomp_test.json'))
    annotations = json.load(open('/tmp/annotation_output.json'))
    gte_results = json.load(open('/tmp/gte_recall_v2.json'))

    print("Loading Qwen3-Embedding...")
    tokenizer = AutoTokenizer.from_pretrained('/zhaoshu/llm/Qwen3-Embedding-0.6B', padding_side='left', trust_remote_code=True)
    model = AutoModel.from_pretrained('/zhaoshu/llm/Qwen3-Embedding-0.6B', trust_remote_code=True).cuda().eval()

    task_desc = "Given a knowledge graph question, retrieve relevant graph relations that answer the question"

    def encode_batch(texts, batch_size=64):
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            bd = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to('cuda')
            with torch.no_grad():
                out = model(**bd)
                embs = last_token_pool(out.last_hidden_state, bd['attention_mask'])
            all_embs.append(F.normalize(embs, p=2, dim=1))
        return torch.cat(all_embs, dim=0)

    def get_instruct(query):
        return f'Instruct: {task_desc}\nQuery: {query}'

    # Build annotation lookup and gte lookup
    anno_map = {a['idx']: a for a in annotations}
    gte_map = {c['idx']: c for c in gte_results}

    ks = [1, 3, 5, 10, 20]
    # Chain hop stats
    chain_stats = {k: {'hit': 0, 'total': 0} for k in ks}

    lines = []
    lines.append("# Chain Decomposition GTE Recall vs Sub-question/RQ/OQ")
    lines.append("")
    lines.append("## Format")
    lines.append("- **Chain**: Chinese relation description from chain decomposition")
    lines.append("- **SQ**: sub-question | **RQ**: relation_query | **OQ**: original question")
    lines.append("- ✅ = top-5 | ⚠️ = top-10 | ❌ = miss")
    lines.append("")

    # We need to map chain hops to annotated targets
    # Strategy: for each case, the chain hops should roughly align 1:1 with annotation steps
    # We'll evaluate: for each hop, does its relation desc recall ANY gold rel from the case?

    # Also track per-hop using the annotation mapping (hop i -> step i+1 targets)
    hop_eval_details = []

    for cr in chain_results:
        idx = cr['idx']
        parsed = cr.get('parsed')
        if not parsed:
            continue

        cid = pilot[idx]['case_id']
        sample = pkl_map.get(cid)
        sq = sparql_map.get(cid)
        if not sample or not sq:
            continue

        question = sample.get('question', '')
        all_rels = list(sample.get('relation_list', []))

        import re as re2
        def extract_sparql_rels(sparql_text):
            raw = re2.findall(r'ns:([a-zA-Z][a-zA-Z0-9_.]+)', sparql_text)
            return list(dict.fromkeys(r for r in raw if '.' in r and not r.startswith('m.')))
        sparql_rels = extract_sparql_rels(sq.get('sparql', ''))

        # Gold rels in subgraph
        gold_in_sub = []
        for sr in sparql_rels:
            match = find_gold_in_list(sr, all_rels)
            if match:
                gold_in_sub.append((sr, match))

        if not gold_in_sub or not all_rels:
            continue

        # Encode relations
        r_embs = encode_batch(all_rels, batch_size=64)

        # Get annotation targets for this case
        anno = anno_map.get(idx)
        anno_targets = {}  # step_num -> target_gold_rels
        if anno:
            for s in anno['steps']:
                anno_targets[s['step']] = s['target_gold_rels']

        # Get previous GTE results
        gte_case = gte_map.get(idx, {})
        gte_steps = gte_case.get('steps', [])

        hops = parsed['hops']
        anchor = parsed['anchor']

        lines.append(f"## Case {idx}: {question}")
        lines.append(f"**Chain**: [{anchor}] " + " -> ".join(f"（{h['relation']}）" for h in hops))
        lines.append(f"**Gold in subgraph**: `{[g[0] for g in gold_in_sub]}`")
        if anno and anno.get('unmapped_gold_rels'):
            lines.append(f"**Unmapped**: `{anno['unmapped_gold_rels']}`")
        lines.append("")

        for hi, hop in enumerate(hops):
            rel_desc = hop['relation']
            endpoint = hop['endpoint']

            # Use chain relation desc as query
            query_text = get_instruct(rel_desc)
            q_emb = encode_batch([query_text], batch_size=1)
            scores = (r_embs @ q_emb.T).squeeze(1).cpu().tolist()
            ranked = sorted(enumerate(all_rels), key=lambda x: -scores[x[0]])

            # Find ALL gold rels in top-K
            chain_gold_ranks = {}
            for sr, sr_match in gold_in_sub:
                for rank, (ri, name) in enumerate(ranked):
                    if name == sr_match:
                        chain_gold_ranks[sr] = rank + 1
                        break

            # Get annotated targets for this hop (hop i maps to step i+1)
            step_num = hi + 1
            targets = anno_targets.get(step_num, [])

            chain_best = min(chain_gold_ranks.values()) if chain_gold_ranks else 999
            chain_target_best = min(chain_gold_ranks.get(t, 999) for t in targets) if targets else chain_best

            for k in ks:
                chain_stats[k]['total'] += 1
                if chain_target_best <= k:
                    chain_stats[k]['hit'] += 1

            # Get SQ/RQ/OQ for comparison
            sq_best = rq_best = oq_best = 999
            sq_top5 = []
            if gte_case:
                gte_step = next((gs for gs in gte_steps if gs.get('step') == step_num), None)
                if gte_step:
                    sq_ranking = gte_step.get('rankings', {}).get('sub_question', {})
                    rq_ranking = gte_step.get('rankings', {}).get('relation_query', {})
                    oq_ranking = gte_step.get('rankings', {}).get('original_question', {})
                    sq_top5 = [name for name, _ in sq_ranking.get('top10', [])[:5]]
                    sq_gold = sq_ranking.get('gold_found_at', {})
                    rq_gold = rq_ranking.get('gold_found_at', {})
                    oq_gold = oq_ranking.get('gold_found_at', {})
                    if targets:
                        sq_best = min(sq_gold.get(t, 999) for t in targets)
                        rq_best = min(rq_gold.get(t, 999) for t in targets)
                        oq_best = min(oq_gold.get(t, 999) for t in targets)

            ch_ic = rank_icon(chain_target_best)
            sq_ic = rank_icon(sq_best)
            rq_ic = rank_icon(rq_best)
            oq_ic = rank_icon(oq_best)

            lines.append(f"### Hop {hi+1} [{rel_desc}] -> {endpoint}")
            lines.append(f"- **Target** (step {step_num}): `{targets if targets else 'no annotation'}`")

            # Show chain ranks for targets
            if targets:
                for t in targets:
                    ch_r = chain_gold_ranks.get(t, '—')
                    sq_r = '?'; rq_r = '?'; oq_r = '?'
                    if gte_case:
                        gte_step = next((gs for gs in gte_steps if gs.get('step') == step_num), None)
                        if gte_step:
                            sq_r = gte_step.get('rankings',{}).get('sub_question',{}).get('gold_found_at',{}).get(t, '?')
                            rq_r = gte_step.get('rankings',{}).get('relation_query',{}).get('gold_found_at',{}).get(t, '?')
                            oq_r = gte_step.get('rankings',{}).get('original_question',{}).get('gold_found_at',{}).get(t, '?')
                    lines.append(f"  - `{t}`: Chain{rank_icon(ch_r) if isinstance(ch_r,int) else ''}@{ch_r} | SQ{rank_icon(sq_r) if isinstance(sq_r,int) else ''}@{sq_r} | RQ{rank_icon(rq_r) if isinstance(rq_r,int) else ''}@{rq_r} | OQ{rank_icon(oq_r) if isinstance(oq_r,int) else ''}@{oq_r}")

            chain_top5 = [name for _, name in ranked[:5]]
            lines.append(f"- **Chain Top-5**: `{chain_top5}`")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Summary
    lines.append("## 汇总")
    lines.append("")
    total = chain_stats[5]['total']
    chain_h5 = chain_stats[5]['hit']
    chain_h10 = chain_stats[10]['hit']

    lines.append(f"共 {total} hops with targets")
    lines.append("")
    lines.append("| Metric | Chain | SQ | RQ | OQ |")
    lines.append("|---|---|---|---|---|")
    # Recalculate SQ/RQ/OQ from annotation evaluation (use previous results)
    lines.append(f"| ✅ Hit@5 | {chain_h5} ({chain_h5/total*100:.1f}%) | 39 (61.9%) | 46 (73.0%) | 32 (50.8%) |")
    lines.append(f"| ✅+⚠️ Hit@10 | {chain_h10} ({chain_h10/total*100:.1f}%) | 46 (73.0%) | 47 (74.6%) | 41 (65.1%) |")
    chain_miss = total - chain_h10
    lines.append(f"| ❌ Miss@10 | {chain_miss} ({chain_miss/total*100:.1f}%) | 17 (27.0%) | 16 (25.4%) | 22 (34.9%) |")

    Path('/tmp/chain_gte_review.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f"\nReview -> /tmp/chain_gte_review.md")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Chain GTE Recall ({total} hops):")
    print(f"  Hit@5:  {chain_h5}/{total} = {chain_h5/total*100:.1f}%")
    print(f"  Hit@10: {chain_h10}/{total} = {chain_h10/total*100:.1f}%")
    print(f"  Miss:   {chain_miss}/{total} = {chain_miss/total*100:.1f}%")
    print(f"\n  Previous results for comparison:")
    print(f"  SQ:  Hit@5=61.9% Hit@10=73.0%")
    print(f"  RQ:  Hit@5=73.0% Hit@10=74.6%")
    print(f"  OQ:  Hit@5=50.8% Hit@10=65.1%")


if __name__ == '__main__':
    main()
