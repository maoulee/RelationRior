#!/usr/bin/env python3
"""GTE Recall Evaluation v3: Using annotated step -> gold relation mapping.

For each step, only check if its annotated target gold relation(s) are in top-K.
This gives a clean evaluation of per-step GTE recall quality.
"""

import argparse, json, re, pickle
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

def get_instruct(task, query):
    return f'Instruct: {task}\nQuery: {query}'


def main(args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    # Load data
    pkl_map = {d['id']: d for d in pickle.load(open(args.pkl, 'rb'))}
    sparql_map = {d['ID']: d for d in json.load(open(args.sparql))}
    pilot = json.load(open(args.pilot))
    annotations = json.load(open(args.annotation))

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
    query_types = ['sub_question', 'relation_query', 'original_question']

    # Per-step stats: for each step with targets, check if ANY target is in top-K
    stats = {qt: {k: {'hit': 0, 'miss': 0, 'details': []} for k in ks} for qt in query_types}

    for anno in annotations:
        idx = anno['idx']
        cid = pilot[idx]['case_id']
        sample = pkl_map.get(cid)
        if not sample:
            continue

        question = sample.get('question', '')
        all_rels = list(sample.get('relation_list', []))

        # Encode all relations once
        r_embs = encode_batch(all_rels, batch_size=64)

        for step_info in anno['steps']:
            targets = step_info['target_gold_rels']
            if not targets:
                continue

            # Find target relations in subgraph
            target_matches = {}  # sparql_name -> subgraph_name
            for t in targets:
                match = find_gold_in_list(t, all_rels)
                if match:
                    target_matches[t] = match

            if not target_matches:
                # Target gold rels not in subgraph - can't be recalled
                for qt in query_types:
                    for k in ks:
                        stats[qt][k]['miss'] += 1
                        stats[qt][k]['details'].append({
                            'case': idx, 'step': step_info['step'],
                            'reason': f'target not in subgraph: {targets}'
                        })
                continue

            sq_text = step_info.get('sub_question', '')
            rq_text = step_info.get('relation_query', '')

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

                # Find rank of each target
                target_ranks = {}
                for t_sparql, t_sub in target_matches.items():
                    for rank, (ri, name) in enumerate(ranked):
                        if name == t_sub:
                            target_ranks[t_sparql] = rank + 1
                            break

                best_rank = min(target_ranks.values()) if target_ranks else 999

                for k in ks:
                    if best_rank <= k:
                        stats[qt][k]['hit'] += 1
                    else:
                        stats[qt][k]['miss'] += 1
                        stats[qt][k]['details'].append({
                            'case': idx, 'step': step_info['step'],
                            'sq': sq_text[:50],
                            'targets': targets,
                            'target_ranks': target_ranks,
                            'best_rank': best_rank,
                        })

    # === Print Results ===
    total_steps = sum(1 for a in annotations for s in a['steps'] if s['target_gold_rels'])
    print(f"\n{'='*75}")
    print(f"Annotated GTE Recall ({len(annotations)} cases, {total_steps} steps with targets)")
    print(f"{'='*75}")
    print(f"{'Query Type':<22} " + " ".join(f"{'R@'+str(k):>8}" for k in ks))
    print("-" * 75)
    for qt in query_types:
        row = f"{qt:<22} "
        for k in ks:
            s = stats[qt][k]
            total = s['hit'] + s['miss']
            pct = s['hit'] / total * 100 if total > 0 else 0
            row += f"{pct:>7.1f}% "
        print(row)

    # Show missed steps for sub_question
    print(f"\n{'='*75}")
    print("Steps where sub_question MISSED target (R@5):")
    print(f"{'='*75}")
    for d in stats['sub_question'][5]['details']:
        print(f"  Case {d['case']} Step {d['step']}: \"{d['sq']}\"")
        print(f"    targets={d['targets']} best_rank={d.get('best_rank', '?')} ranks={d.get('target_ranks', {})}")

    print(f"\n{'='*75}")
    print("Steps where relation_query MISSED target (R@5):")
    print(f"{'='*75}")
    for d in stats['relation_query'][5]['details']:
        print(f"  Case {d['case']} Step {d['step']}: \"{d['sq']}\"")
        print(f"    targets={d['targets']} best_rank={d.get('best_rank', '?')} ranks={d.get('target_ranks', {})}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='/zhaoshu/llm/Qwen3-Embedding-0.6B')
    p.add_argument('--pkl', default='/zhaoshu/subgraph/data/cwq_processed/test_fixed_relaxation.pkl')
    p.add_argument('--sparql', default='/zhaoshu/subgraph/data/cwq_sparql/test.json')
    p.add_argument('--pilot', default='/zhaoshu/subgraph/reports/stage_pipeline_test/cwq_50_stage_v7/pilot.json')
    p.add_argument('--annotation', default='/tmp/annotation_output.json')
    main(p.parse_args())
