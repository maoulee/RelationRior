#!/usr/bin/env python3
"""Test GTE recall v2: detailed per-step evaluation.

Key fixes:
- Only evaluate gold rels that exist in the subgraph's relation_list
- Show detailed top-K rankings for each sub-question
- Per-step: check if ANY gold rel is found in top-K (binary recall)
"""

import argparse, json, re, sys, pickle
from pathlib import Path

def load_pkl(path):
    with open(path, 'rb') as f:
        return {d['id']: d for d in pickle.load(f)}

def norm(t):
    return re.sub(r'[^a-z0-9]', '', t.lower().strip())

def extract_sparql_rels(sparql_text):
    raw = re.findall(r'ns:([a-zA-Z][a-zA-Z0-9_.]+)', sparql_text)
    return list(dict.fromkeys(r for r in raw if '.' in r and not r.startswith('m.')))

def last_token_pool(last_hidden_states, attention_mask):
    import torch
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_instruct(task, query):
    return f'Instruct: {task}\nQuery: {query}'

def find_gold_in_list(gold_rel, rel_list):
    """Check if gold_rel matches any relation in rel_list, return the match."""
    gn = norm(gold_rel)
    for r in rel_list:
        rn = norm(r)
        if gn == rn or gn in rn or rn in gn:
            return r
    return None


def main(args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    pkl_data = load_pkl(args.pkl)
    decomp = json.load(open(args.decomp))
    sparql_map = {d['ID']: d for d in json.load(open(args.sparql))}
    pilot = json.load(open(args.pilot))
    print(f"  pkl: {len(pkl_data)} | decomp: {len(decomp)} | sparql: {len(sparql_map)} | pilot: {len(pilot)}")

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left', trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).cuda().eval()
    print("Model loaded.")

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

    ks = [1, 3, 5, 10, 20]
    # Two evaluation modes:
    # 1. per-step: for each step, does top-K contain at least one gold rel? (binary)
    # 2. per-gold-rel: for each gold rel (in subgraph), is it found by any step's top-K?
    step_binary = {k: 0 for k in ks}  # how many steps found >=1 gold rel
    step_total = 0
    gold_found_by_any_step = {}  # gold_rel -> bool
    gold_total_in_subgraph = 0

    # Also track per-query-type
    query_types = ['sub_question', 'relation_query', 'original_question']
    all_stats = {qt: {
        'step_binary': {k: 0 for k in ks},
        'step_total': 0,
        'gold_found': {},
        'gold_total': 0,
    } for qt in query_types}

    # Detailed output for manual inspection
    detailed_output = []

    for dc in decomp:
        cid = pilot[dc['idx']]['case_id']
        sample = pkl_data.get(cid)
        sq = sparql_map.get(cid)
        if not sample or not sq:
            continue

        question = sample.get('question', '')
        all_rels = list(sample.get('relation_list', []))
        sparql_rels = extract_sparql_rels(sq.get('sparql', ''))

        # Only keep gold rels that are in the subgraph
        gold_in_subgraph = []
        gold_missing = []
        for sr in sparql_rels:
            match = find_gold_in_list(sr, all_rels)
            if match:
                gold_in_subgraph.append((sr, match))
            else:
                gold_missing.append(sr)

        if not gold_in_subgraph:
            continue

        # Encode relations once
        r_embs = encode_batch(all_rels, batch_size=64)

        case_detail = {
            'idx': dc['idx'],
            'question': question,
            'sparql_rels': sparql_rels,
            'gold_in_subgraph': [g[0] for g in gold_in_subgraph],
            'gold_missing': gold_missing,
            'steps': [],
        }

        for qt in query_types:
            all_stats[qt]['gold_total'] += len(gold_in_subgraph)
            for sr, _ in gold_in_subgraph:
                key = (dc['idx'], sr)
                if key not in all_stats[qt]['gold_found']:
                    all_stats[qt]['gold_found'][key] = False

        # For each step, compute rankings per query type
        for si, step in enumerate(dc.get('steps', [])):
            sq_text = step.get('question', '')
            rq_text = step.get('relation_query', '')
            step_type = step.get('type', '')

            step_detail = {
                'step': si + 1,
                'type': step_type,
                'sub_question': sq_text,
                'relation_query': rq_text,
                'rankings': {},
            }

            queries = {}
            if sq_text:
                queries['sub_question'] = get_instruct(task_desc, sq_text)
            if rq_text:
                queries['relation_query'] = get_instruct(task_desc, rq_text)
            queries['original_question'] = get_instruct(task_desc, question)

            for qt, q_text in queries.items():
                q_emb = encode_batch([q_text], batch_size=1)
                scores = (r_embs @ q_emb.T).squeeze(1).cpu().tolist()
                ranked = sorted(enumerate(all_rels), key=lambda x: -scores[x[0]])
                top_indices = {idx for idx, _ in ranked[:20]}
                top_names = [name for _, name in ranked[:20]]
                top_scores = [scores[idx] for idx, _ in ranked[:20]]

                # Which gold rels are in top-K?
                gold_found_at = {}
                for sr, sr_match in gold_in_subgraph:
                    for rank, (idx, name) in enumerate(ranked):
                        if name == sr_match:
                            gold_found_at[sr] = rank + 1
                            break

                # Binary: did we find at least one gold rel in top-K?
                for k in ks:
                    found_any = any(rank <= k for rank in gold_found_at.values())
                    if found_any:
                        all_stats[qt]['step_binary'][k] += 1
                all_stats[qt]['step_total'] += 1

                # Mark gold rels as found
                for sr, rank in gold_found_at.items():
                    if rank <= 10:
                        key = (dc['idx'], sr)
                        all_stats[qt]['gold_found'][key] = True

                step_detail['rankings'][qt] = {
                    'top10': [(name, f'{top_scores[i]:.4f}') for i, name in enumerate(top_names[:10])],
                    'gold_found_at': gold_found_at,
                }

            case_detail['steps'].append(step_detail)

        detailed_output.append(case_detail)

    # === Print Results ===
    print(f"\n{'='*80}")
    print(f"1. Per-Step Binary Recall (step found >=1 gold rel in top-K)")
    print(f"   Denominator: total steps across all cases")
    print(f"{'='*80}")
    print(f"{'Query Type':<22} " + " ".join(f"{'R@'+str(k):>8}" for k in ks))
    print("-" * 80)
    for qt in query_types:
        s = all_stats[qt]
        row = f"{qt:<22} "
        for k in ks:
            pct = s['step_binary'][k] / s['step_total'] * 100 if s['step_total'] > 0 else 0
            row += f"{pct:>7.1f}% "
        row += f" (n={s['step_total']})"
        print(row)

    print(f"\n{'='*80}")
    print(f"2. Per-Gold-Rel Recall (gold rel found by ANY step in top-10)")
    print(f"   Denominator: gold rels that exist in subgraph")
    print(f"{'='*80}")
    for qt in query_types:
        s = all_stats[qt]
        found = sum(1 for v in s['gold_found'].values() if v)
        total = s['gold_total']
        pct = found / total * 100 if total > 0 else 0
        print(f"  {qt:<22}: {found}/{total} = {pct:.1f}%")

    # === Detailed per-case output ===
    print(f"\n{'='*80}")
    print("3. Detailed Per-Case Rankings (top-5 for sub_question)")
    print(f"{'='*80}")
    for case in detailed_output:
        print(f"\n--- Case {case['idx']}: {case['question'][:80]}")
        print(f"    Gold (in subgraph): {case['gold_in_subgraph']}")
        if case['gold_missing']:
            print(f"    Gold (missing): {case['gold_missing']}")
        for step in case['steps']:
            sq_rank = step['rankings'].get('sub_question', {})
            print(f"\n    Step {step['step']} [{step['type']}]: \"{step['sub_question'][:70]}\"")
            print(f"      relation_query: \"{step['relation_query'][:50]}\"")
            if sq_rank.get('top10'):
                print(f"      Top-5 (SQ):")
                for i, (name, score) in enumerate(sq_rank['top10'][:5]):
                    marker = ""
                    for sr, rank in sq_rank.get('gold_found_at', {}).items():
                        if rank == i + 1:
                            marker = f" <-- GOLD: {sr}"
                    print(f"        {i+1}. {name} ({score}){marker}")
                # Show gold positions if not in top-5
                for sr, rank in sq_rank.get('gold_found_at', {}).items():
                    if rank > 5:
                        print(f"        ... gold '{sr}' at rank {rank}")
                    elif sr not in [n for n, _ in sq_rank['top10'][:5]]:
                        pass  # already shown above
            # Also show RQ gold positions
            rq_rank = step['rankings'].get('relation_query', {})
            rq_gold = rq_rank.get('gold_found_at', {})
            if rq_gold:
                gold_positions = [f"{sr}@{r}" for sr, r in rq_gold.items()]
                print(f"      RQ gold positions: {gold_positions}")

    Path(args.output).write_text(json.dumps(detailed_output, indent=2, ensure_ascii=False))
    print(f"\nDetailed JSON -> {args.output}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='/zhaoshu/llm/Qwen3-Embedding-0.6B')
    p.add_argument('--decomp', default='/tmp/decomp_test_50_v6_nl.json')
    p.add_argument('--pkl', default='/zhaoshu/subgraph/data/cwq_processed/test_fixed_relaxation.pkl')
    p.add_argument('--sparql', default='/zhaoshu/subgraph/data/cwq_sparql/test.json')
    p.add_argument('--pilot', default='/zhaoshu/subgraph/reports/stage_pipeline_test/cwq_50_stage_v7/pilot.json')
    p.add_argument('--output', default='/tmp/gte_recall_v2.json')
    main(p.parse_args())
