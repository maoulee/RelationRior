#!/usr/bin/env python3
"""GTE recall: NER chain decomposition with annotated evaluation."""

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

def rank_icon(rank):
    if rank <= 5: return "✅"
    elif rank <= 10: return "⚠️"
    else: return "❌"


def main():
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    pkl_map = {d['id']: d for d in pickle.load(open('/zhaoshu/subgraph/data/cwq_processed/test_literal_and_language_fixed.pkl', 'rb'))}
    sparql_map = {d['ID']: d for d in json.load(open('/zhaoshu/subgraph/data/cwq_sparql/test.json'))}
    pilot = json.load(open('/zhaoshu/subgraph/reports/stage_pipeline_test/cwq_50_stage_v7/pilot.json'))
    chain_results = json.load(open('/tmp/chain_decomp_ner.json'))
    annotations = json.load(open('/tmp/annotation_output.json'))

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

    anno_map = {a['idx']: a for a in annotations}

    ks = [1, 3, 5, 10, 20]
    short_stats = {k: {'hit': 0, 'total': 0} for k in ks}
    def_stats = {k: {'hit': 0, 'total': 0} for k in ks}

    lines = []
    lines.append("# Chain Decomposition GTE Recall: Short vs Definition")
    lines.append("")

    for cr in chain_results:
        idx = cr['idx']
        parsed = cr.get('parsed')
        if not parsed: continue

        cid = pilot[idx]['case_id']
        sample = pkl_map.get(cid)
        sq = sparql_map.get(cid)
        if not sample or not sq: continue

        question = sample.get('question', '')
        all_rels = list(sample.get('relation_list', []))
        sparql_rels = extract_sparql_rels(sq.get('sparql', ''))

        gold_in_sub = []
        for sr in sparql_rels:
            match = find_gold_in_list(sr, all_rels)
            if match:
                gold_in_sub.append((sr, match))

        if not gold_in_sub or not all_rels: continue

        r_embs = encode_batch(all_rels, batch_size=64)

        anno = anno_map.get(idx)
        anno_targets = {}
        if anno:
            for s in anno['steps']:
                anno_targets[s['step']] = s['target_gold_rels']

        hops = parsed['hops']
        gold_answer = sample.get('a_entity', [])
        q_entities = sample.get('q_entity', [])

        # Identify anchor and endpoints
        anchor = parsed['anchor']
        endpoints = [h['endpoint'] for h in hops if h['endpoint'] != 'node']

        lines.append(f"## Case {idx}: {question}")
        lines.append(f"**Gold Answer**: `{gold_answer}`")
        lines.append(f"**NER Entities**: `{q_entities}`")
        lines.append(f"**Anchor**: `{anchor}`")
        if endpoints:
            lines.append(f"**Endpoints**: `{endpoints}`")
        chains_str = " -> ".join([f"({h['relation']})->{h['endpoint']}" for h in hops])
        lines.append(f"**Chain**: {anchor} {chains_str}")
        lines.append("")

        for hi, hop in enumerate(hops):
            rel_desc = hop['relation']
            rel_def = hop.get('definition', '')

            # --- Short query ---
            q_short = f'Instruct: {task_desc}\nQuery: {rel_desc}'
            q_emb_s = encode_batch([q_short], batch_size=1)
            scores_s = (r_embs @ q_emb_s.T).squeeze(1).cpu().tolist()
            ranked_s = sorted(enumerate(all_rels), key=lambda x: -scores_s[x[0]])

            # --- Definition query ---
            if rel_def:
                q_def = f'Instruct: {task_desc}\nQuery: {rel_def}'
            else:
                q_def = q_short
            q_emb_d = encode_batch([q_def], batch_size=1)
            scores_d = (r_embs @ q_emb_d.T).squeeze(1).cpu().tolist()
            ranked_d = sorted(enumerate(all_rels), key=lambda x: -scores_d[x[0]])

            # Compute gold ranks for both
            gold_ranks_s = {}
            gold_ranks_d = {}
            for sr, sr_match in gold_in_sub:
                for rank, (ri, name) in enumerate(ranked_s):
                    if name == sr_match:
                        gold_ranks_s[sr] = rank + 1
                        break
                for rank, (ri, name) in enumerate(ranked_d):
                    if name == sr_match:
                        gold_ranks_d[sr] = rank + 1
                        break

            step_num = hi + 1
            targets = anno_targets.get(step_num, [])
            best_s = min(gold_ranks_s.get(t, 999) for t in targets) if targets else 999
            best_d = min(gold_ranks_d.get(t, 999) for t in targets) if targets else 999

            for k in ks:
                short_stats[k]['total'] += 1
                def_stats[k]['total'] += 1
                if best_s <= k:
                    short_stats[k]['hit'] += 1
                if best_d <= k:
                    def_stats[k]['hit'] += 1

            ic_s = rank_icon(best_s)
            ic_d = rank_icon(best_d)
            top5_s = [name for _, name in ranked_s[:5]]
            top5_d = [name for _, name in ranked_d[:5]]

            lines.append(f"### Hop {hi+1} ({rel_desc})")
            lines.append(f"- **Definition**: {rel_def[:100] if rel_def else 'N/A'}")
            lines.append(f"- **Target** (step {step_num}): `{targets if targets else 'no annotation'}`")

            if targets:
                for t in targets:
                    rs = gold_ranks_s.get(t, '—')
                    rd = gold_ranks_d.get(t, '—')
                    lines.append(f"  - `{t}`: Short{rank_icon(rs) if isinstance(rs,int) else ''}@{rs} | Def{rank_icon(rd) if isinstance(rd,int) else ''}@{rd}")
            else:
                found_s = [(sr, gold_ranks_s[sr]) for sr in gold_ranks_s if gold_ranks_s[sr] <= 10]
                found_d = [(sr, gold_ranks_d[sr]) for sr in gold_ranks_d if gold_ranks_d[sr] <= 10]
                if found_s or found_d:
                    lines.append(f"  - Short top-10: {[(sr, f'@{r}') for sr, r in found_s]}")
                    lines.append(f"  - Def top-10: {[(sr, f'@{r}') for sr, r in found_d]}")

            lines.append(f"- **Short Top-5** {ic_s}: `{top5_s}`")
            lines.append(f"- **Def Top-5** {ic_d}: `{top5_d}`")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Summary
    total = short_stats[5]['total']
    s_h1 = short_stats[1]['hit']
    s_h3 = short_stats[3]['hit']
    s_h5 = short_stats[5]['hit']
    s_h10 = short_stats[10]['hit']
    d_h1 = def_stats[1]['hit']
    d_h3 = def_stats[3]['hit']
    d_h5 = def_stats[5]['hit']
    d_h10 = def_stats[10]['hit']

    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Short Desc | Definition | Delta |")
    lines.append(f"|---|---|---|---|")
    for k in [1, 3, 5, 10]:
        sh = short_stats[k]['hit']
        dh = def_stats[k]['hit']
        delta = dh - sh
        sign = '+' if delta > 0 else ''
        lines.append(f"| Hit@{k} | {sh} ({sh/total*100:.1f}%) | {dh} ({dh/total*100:.1f}%) | {sign}{delta} |")

    Path('/tmp/chain_ner_gte_review.md').write_text('\n'.join(lines), encoding='utf-8')

    print(f"\n{'='*60}")
    print(f"GTE Recall Comparison ({total} hops):")
    print(f"  {'Metric':<10} {'Short':>10} {'Definition':>12} {'Delta':>8}")
    print(f"  {'-'*42}")
    for k in [1, 3, 5, 10]:
        sh = short_stats[k]['hit']
        dh = def_stats[k]['hit']
        delta = dh - sh
        sign = '+' if delta > 0 else ''
        print(f"  Hit@{k:<6} {sh/total*100:>8.1f}% {dh/total*100:>10.1f}% {sign}{delta:>6}")
    print(f"\nReview -> /tmp/chain_ner_gte_review.md")


if __name__ == '__main__':
    main()
