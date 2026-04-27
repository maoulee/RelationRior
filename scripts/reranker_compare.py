#!/usr/bin/env python3
"""Reranker vs LLM: Re-rank GTE candidates and compare with LLM selection.

For each step in the decomposition:
1. GTE returns top-K candidates (already in pipeline results)
2. Reranker re-ranks these same candidates
3. Compare: Reranker top-N vs LLM selected - which captures more SPARQL GT?

Uses Qwen3-Embedding with correct last_token_pool and padding_side='left'.
"""

import argparse, json, re, sys
from pathlib import Path
from collections import defaultdict

def load_pkl(path):
    import pickle
    with open(path, 'rb') as f:
        return {d['id']: d for d in pickle.load(f)}

def load_json(path):
    return json.loads(Path(path).read_text())

def norm(t):
    return re.sub(r'[^a-z0-9]', '', t.lower().strip())

def extract_sparql_rels(sparql_text):
    raw = re.findall(r'ns:([a-zA-Z][a-zA-Z0-9_.]+)', sparql_text)
    return list(dict.fromkeys(r for r in raw if '.' in r and not r.startswith('m.')))

def get_detailed_instruct(task: str, query: str) -> str:
    return f'Instruct: {task}\nQuery:{query}'

def last_token_pool(last_hidden_states, attention_mask):
    import torch
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def eval_recall(selected_names, sparql_rels):
    """Check how many SPARQL GT relations are covered by selected_names."""
    found = 0
    matched = []
    missing = []
    for sr in sparql_rels:
        srn = norm(sr)
        hit = any(srn in norm(s) or norm(s) in srn for s in selected_names)
        if hit:
            found += 1
            matched.append(sr)
        else:
            missing.append(sr)
    return found, matched, missing


def main(args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    print("Loading data...")
    pkl_data = load_pkl(args.pkl_dir + '/test.pkl')
    pilot = load_json(args.pilot)
    results = load_json(args.results)
    sparql_map = {d['ID']: d for d in load_json(args.sparql)}
    result_map = {r['case_id']: r for r in results}
    print(f"  pkl: {len(pkl_data)} | pilot: {len(pilot)} | results: {len(results)}")

    case_indices = args.cases if args.cases else list(range(min(len(pilot), args.limit)))

    # Load Qwen3-Embedding
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left', trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).cuda().eval()
    print("Model loaded.")

    task_desc = "Given a knowledge graph question, retrieve relevant graph relations that answer the question"

    # Stats per-step
    stats = {
        'gte_top5': {'found': 0, 'total': 0},
        'gte_top10': {'found': 0, 'total': 0},
        'reranker_top5': {'found': 0, 'total': 0},
        'reranker_top10': {'found': 0, 'total': 0},
        'llm': {'found': 0, 'total': 0},
    }

    per_case = []

    for idx in case_indices:
        if idx >= len(pilot):
            continue
        case = pilot[idx]
        cid = case['case_id']
        sample = pkl_data.get(cid)
        res = result_map.get(cid)
        sq = sparql_map.get(cid)
        if not sample or not res or not sq:
            continue

        question = sample.get('question', '')
        subgraph_rels = list(sample.get('relation_list', []))
        sparql_rels = extract_sparql_rels(sq.get('sparql', ''))

        # Decomposition steps
        steps = res.get('steps', [])
        rrd = res.get('relation_retrieval_details', [])
        llm_step_rels = res.get('step_relations', [])  # list of list of rel indices

        case_detail = {
            'idx': idx, 'question': question[:70],
            'sparql_rels': sparql_rels,
            'steps': [],
        }

        for si, step_rr in enumerate(rrd):
            if not isinstance(step_rr, dict):
                continue

            # Get GTE candidates for this step (the actual candidates the pipeline saw)
            gte_candidates = []
            for q in step_rr.get('queries', []):
                for item in q.get('top_k', []):
                    name = item.get('candidate', '')
                    score = item.get('score', 0)
                    if name and name not in [c[0] for c in gte_candidates]:
                        gte_candidates.append((name, score))
            # Deduplicate keeping best score
            seen = {}
            for name, score in gte_candidates:
                if name not in seen or score > seen[name]:
                    seen[name] = score
            gte_candidates = sorted(seen.items(), key=lambda x: -x[1])

            if not gte_candidates:
                continue

            # GTE ranked lists
            gte_top5 = [n for n, _ in gte_candidates[:5]]
            gte_top10 = [n for n, _ in gte_candidates[:10]]

            # LLM selected for this step
            llm_rel_indices = llm_step_rels[si] if si < len(llm_step_rels) else []
            llm_rel_names = [subgraph_rels[ri] for ri in llm_rel_indices if ri < len(subgraph_rels)]

            # Step query for reranking
            step_info = steps[si] if si < len(steps) else {}
            step_query = step_info.get('relation_query', '')
            if not step_query:
                step_query = question

            # === Rerank with Qwen3-Embedding ===
            candidate_names = [n for n, _ in gte_candidates]
            query_text = get_detailed_instruct(task_desc, step_query)
            input_texts = [query_text] + candidate_names

            batch_dict = tokenizer(
                input_texts, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            ).to('cuda')

            with torch.no_grad():
                outputs = model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)

            q_emb = embeddings[0]
            c_embs = embeddings[1:]
            scores = (c_embs @ q_emb).cpu().tolist()

            reranked = sorted(zip(candidate_names, scores), key=lambda x: -x[1])
            reranker_top5 = [n for n, _ in reranked[:5]]
            reranker_top10 = [n for n, _ in reranked[:10]]

            # Evaluate each method against SPARQL GT
            g5_f, g5_m, _ = eval_recall(gte_top5, sparql_rels)
            g10_f, _, _ = eval_recall(gte_top10, sparql_rels)
            r5_f, r5_m, _ = eval_recall(reranker_top5, sparql_rels)
            r10_f, _, _ = eval_recall(reranker_top10, sparql_rels)
            l_f, l_m, _ = eval_recall(llm_rel_names, sparql_rels)

            # Only count once per case (not per step) to avoid double counting
            # Actually, let's count per-step to see granularity
            step_detail = {
                'step': si + 1,
                'query': step_query[:60],
                'gte_candidates': len(gte_candidates),
                'gte_top5': gte_top5,
                'reranker_top5': reranker_top5,
                'llm_selected': llm_rel_names,
                'gte5_found': g5_f, 'gte5_matched': g5_m,
                'rerank5_found': r5_f, 'rerank5_matched': r5_m,
                'llm_found': l_f, 'llm_matched': l_m,
            }
            case_detail['steps'].append(step_detail)

            stats['gte_top5']['found'] += g5_f
            stats['gte_top5']['total'] += len(sparql_rels)
            stats['gte_top10']['found'] += g10_f
            stats['gte_top10']['total'] += len(sparql_rels)
            stats['reranker_top5']['found'] += r5_f
            stats['reranker_top5']['total'] += len(sparql_rels)
            stats['reranker_top10']['found'] += r10_f
            stats['reranker_top10']['total'] += len(sparql_rels)
            stats['llm']['found'] += l_f
            stats['llm']['total'] += len(sparql_rels)

        # Per-case summary
        total_sparql = len(sparql_rels)
        case_g5 = sum(s['gte5_found'] for s in case_detail['steps'])
        case_r5 = sum(s['rerank5_found'] for s in case_detail['steps'])
        case_l = sum(s['llm_found'] for s in case_detail['steps'])
        case_detail['summary'] = f"GTE5={case_g5} Rerank5={case_r5} LLM={case_l} / {total_sparql}"
        per_case.append(case_detail)

        print(f"  Case {idx}: {case_detail['summary']} | {question[:50]}")

    # === Summary ===
    print(f"\n{'='*65}")
    print(f"Per-Step SPARQL Relation Recall ({len(case_indices)} cases, {sum(len(c['steps']) for c in per_case)} steps)")
    print(f"{'='*65}")
    print(f"{'Method':<20} {'Found':>8} {'Total':>8} {'%':>8}")
    print(f"{'-'*65}")
    for key, label in [
        ('gte_top5', 'GTE top-5'),
        ('gte_top10', 'GTE top-10'),
        ('reranker_top5', 'Reranker top-5'),
        ('reranker_top10', 'Reranker top-10'),
        ('llm', 'LLM pruning'),
    ]:
        s = stats[key]
        if s['total'] == 0:
            continue
        pct = s['found'] / s['total'] * 100
        print(f"{label:<20} {s['found']:>8} {s['total']:>8} {pct:>7.1f}%")

    # Per-case: where reranker beats LLM or vice versa (at step level)
    print(f"\n{'='*65}")
    print("Per-case Reranker vs LLM (case-level totals):")
    print(f"{'='*65}")
    r_wins = l_wins = ties = 0
    for c in per_case:
        cr = sum(s['rerank5_found'] for s in c['steps'])
        cl = sum(s['llm_found'] for s in c['steps'])
        if cr > cl:
            r_wins += 1
            print(f"  Rerank+ Case {c['idx']}: Rerank={cr} > LLM={cl} | {c['question']}")
        elif cl > cr:
            l_wins += 1
            if cl - cr >= 2:
                print(f"  LLM+ Case {c['idx']}: LLM={cl} > Rerank={cr} | {c['question']}")
        else:
            ties += 1
    print(f"\n  Reranker wins: {r_wins} | LLM wins: {l_wins} | Tie: {ties}")

    # Step-level detail: where reranker picked different (better) relations
    print(f"\n{'='*65}")
    print("Step-level: Reranker found GT that LLM missed:")
    for c in per_case:
        for s in c['steps']:
            r_extra = set(s.get('rerank5_matched', [])) - set(s.get('llm_matched', []))
            l_extra = set(s.get('llm_matched', [])) - set(s.get('rerank5_matched', []))
            if r_extra:
                print(f"  Case {c['idx']} Step {s['step']}: Rerank found {r_extra}")
            if l_extra:
                print(f"  Case {c['idx']} Step {s['step']}: LLM found {l_extra}")

    Path(args.output).write_text(json.dumps(per_case, indent=2, ensure_ascii=False))
    print(f"\nDetailed → {args.output}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='/zhaoshu/llm/Qwen3-Embedding-0.6B')
    p.add_argument('--limit', type=int, default=50)
    p.add_argument('--cases', type=int, nargs='*')
    p.add_argument('--pkl-dir', default=str(Path(__file__).resolve().parent.parent / 'data/cwq_processed'))
    p.add_argument('--pilot', default='/zhaoshu/subgraph/reports/stage_pipeline_test/cwq_50_stage_v7/pilot.json')
    p.add_argument('--results', default='/zhaoshu/subgraph/reports/stage_pipeline_test/chain_decompose_test/results.json')
    p.add_argument('--sparql', default='/zhaoshu/subgraph/data/cwq_sparql/test.json')
    p.add_argument('--output', default='reranker_comparison.json')
    main(p.parse_args())
