#!/usr/bin/env python3
"""Chain decomposition v5: model outputs Freebase schema-style relation names directly.

Instead of natural language relation descriptions, the model writes each hop's relation
in Freebase schema format (domain.type.property). This enables direct string matching
against the real relation list.
"""

import json, re, pickle, time
from pathlib import Path
import requests


def call_llm(messages, model_id="qwen35-9b-local", max_tokens=1024):
    resp = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=60,
    )
    data = resp.json()
    return data['choices'][0]['message']['content']


CHAIN_PROMPT_SCHEMA = """You are a knowledge graph reasoning expert. Decompose the question into a chain of hops through a Freebase-style knowledge graph.

## Entities
Named entities from the question: {entities}

## Rules
1. Start with ONE entity as anchor. Prefer specific entities over generic ones. MUST be from the entity list.
2. Each hop follows an outgoing relation to a node. Format: EntityA -(schema.relation.name)-> node
3. Use Freebase-style schema relation names: domain.type_or_topic.property (e.g., sports.sports_team.championships, time.event.start_date, location.location.contains).
4. ONE relation per hop. Each hop is a SINGLE graph edge. Split compound steps into separate hops.
5. Attribute lookups (dates, names, numbers, types) MUST be separate hops. For example, "find championship" and "find its date" are TWO hops.
6. Nodes: write "node" for unknowns. Only write an entity name if it is from the provided entity list.
7. Keep chain minimal. Do NOT include verification hops for superlatives ("most recent", "largest", "first").
8. Do NOT invent entity names. Only use entities from the list.

## Output Format
EntityA -(schema.relation.name)-> node -(schema.relation.name)-> node

## Examples

Entities: Serena Williams, Nike
Q: Which brand sponsors the athlete who won the most Wimbledon singles titles?
Serena Williams -(business.endorsement.brand)-> Nike

Entities: Airbus A380
Q: What country is the manufacturer of the A380 headquartered in?
Airbus A380 -(aviation.aircraft.manufacturer)-> node -(organization.organization.headquarters)-> node

Entities: Lou Seal
Q: Lou Seal is the mascot for the team that last won the World Series when?
Lou Seal -(sports.sports_team.team_mascot).inv -> node -(sports.sports_team.championships)-> node -(time.event.start_date)-> node

Entities: Thames, London
Q: What ocean does the river flowing through London empty into?
Thames -(location.location.contains).inv -> London -(geography.river.mouth)-> node

Entities: Mona Lisa, France
Q: Which museum in France houses the painting that depicts Lisa Gherardini?
Mona Lisa -(visual_art.artwork.owner)-> node -(location.location.country)-> France

Entities: Brad Stevens
Q: What year did the basketball team coached by Brad Stevens win the championship?
Brad Stevens -(sports.sports_team.coach).inv -> node -(sports.sports_team.championships)-> node -(time.event.start_date)-> node

Now decompose:
Entities: {entities}
Q: {question}
"""


def parse_chain(text):
    text = text.strip()
    text = re.sub(r'^(Answer|答案)[：:]\s*', '', text)

    # Remove any trailing explanation text
    lines = text.split('\n')
    chain_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            if chain_lines:
                break
            continue
        # Stop at explanation paragraphs
        if line.startswith(('Note:', 'Explanation:', 'The ', 'This ')) and chain_lines:
            break
        chain_lines.append(line)

    chain_text = ' '.join(chain_lines)

    # Parse: EntityA -(rel1)-> node -(rel2)-> node
    anchor_match = re.match(r'^(.+?)\s*-\(', chain_text)
    if not anchor_match:
        return None
    anchor = anchor_match.group(1).strip()
    rest = chain_text[anchor_match.end() - 1:]

    hops = []
    # Match -(schema.relation.name)-> node   or   -(schema.relation.name).inv -> node
    hop_pattern = r'\(([^)]+)\)\s*(?:\.inv\s*)?->\s*(.+?)(?=\s*-\(|$)'
    for m in re.finditer(hop_pattern, rest):
        rel_raw = m.group(1).strip()
        node = m.group(2).strip()
        if not node:
            node = 'node'

        # Normalize: handle .inv suffix
        inverse = False
        if rel_raw.endswith('.inv'):
            rel_raw = rel_raw[:-4].strip()
            inverse = True

        hops.append({
            'relation': rel_raw,
            'endpoint': node,
            'inverse': inverse,
        })

    if not hops:
        return None

    return {
        'anchor': anchor,
        'hops': hops,
        'raw': text,
        'endpoints': [h['endpoint'] for h in hops if h['endpoint'] != 'node'],
    }


def match_schema_to_real(schema_rel, all_rels):
    """Match a schema relation name against real relation list.
    Returns (best_match, match_type) or (None, None).
    """
    # Exact match
    if schema_rel in all_rels:
        return schema_rel, 'exact'

    # Try without .inv
    base = schema_rel.replace('.inv', '')
    if base in all_rels:
        return base, 'exact_base'

    # Substring match in both directions
    for r in all_rels:
        # Check if schema is substring of real or vice versa
        if base in r or r in base:
            return r, 'substring'
        # Check if last part (property) matches
        schema_parts = base.split('.')
        real_parts = r.split('.')
        if len(schema_parts) >= 2 and len(real_parts) >= 2:
            if schema_parts[-1] == real_parts[-1] and schema_parts[-2] == real_parts[-2]:
                return r, 'suffix_match'
            if schema_parts[-1] == real_parts[-1]:
                return r, 'property_match'

    return None, None


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=30)
    p.add_argument('--output', default='/tmp/chain_schema_results.json')
    args = p.parse_args()

    pkl_map = {d['id']: d for d in pickle.load(open('/zhaoshu/subgraph/data/cwq_processed/test_literal_and_language_fixed.pkl', 'rb'))}
    sparql_map = {d['ID']: d for d in json.load(open('/zhaoshu/subgraph/data/cwq_sparql/test.json'))}
    pilot = json.load(open('/zhaoshu/subgraph/reports/stage_pipeline_test/cwq_50_stage_v7/pilot.json'))

    print(f"Testing schema-based chain decomposition on {args.limit} cases...")

    results = []
    ok = fail = 0
    hop_dist = {}
    total_schema_rels = 0
    matched_schema_rels = 0

    # Gold matching stats
    import re as _re
    def extract_sparql_rels(sparql_text):
        raw = _re.findall(r'ns:([a-zA-Z][a-zA-Z0-9_.]+)', sparql_text)
        return list(dict.fromkeys(r for r in raw if '.' in r and not r.startswith('m.')))

    def find_gold_in_list(gold_rel, rel_list):
        gn = norm(gold_rel)
        for r in rel_list:
            rn = norm(r)
            if gn == rn or gn in rn or rn in gn:
                return r
        return None

    def norm(t):
        return _re.sub(r'[^a-z0-9]', '', t.lower().strip())

    # String match recall stats
    ks = [1, 3, 5, 10]
    str_stats = {k: {'hit': 0, 'total': 0} for k in ks}
    gte_stats = {k: {'hit': 0, 'total': 0} for k in ks}

    # Load annotation if exists
    anno_path = '/tmp/annotation_output.json'
    annotations = json.load(open(anno_path)) if Path(anno_path).exists() else []
    anno_map = {a['idx']: a for a in annotations}

    # Load GTE model for comparison
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    print("Loading Qwen3-Embedding for GTE comparison...")
    tokenizer = AutoTokenizer.from_pretrained('/zhaoshu/llm/Qwen3-Embedding-0.6B', padding_side='left', trust_remote_code=True)
    model_gte = AutoModel.from_pretrained('/zhaoshu/llm/Qwen3-Embedding-0.6B', trust_remote_code=True).cuda().eval()

    def last_token_pool(last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def encode_batch(texts, batch_size=64):
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            bd = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to('cuda')
            with torch.no_grad():
                out = model_gte(**bd)
                embs = last_token_pool(out.last_hidden_state, bd['attention_mask'])
            all_embs.append(F.normalize(embs, p=2, dim=1))
        return torch.cat(all_embs, dim=0)

    task_desc = "Given a knowledge graph question, retrieve relevant graph relations that answer the question"

    lines = []
    lines.append("# Schema-based Chain Decomposition: String Match + GTE Comparison")
    lines.append("")

    for i in range(min(args.limit, len(pilot))):
        case = pilot[i]
        cid = case['case_id']
        sample = pkl_map.get(cid)
        sq = sparql_map.get(cid)
        if not sample or not sq:
            continue

        question = sample.get('question', '')
        q_entities = sample.get('q_entity', [])
        ent_str = ", ".join(q_entities) if q_entities else "none"

        prompt = CHAIN_PROMPT_SCHEMA.format(entities=ent_str, question=question)
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = call_llm(messages)
        except Exception as e:
            print(f"  Case {i}: API error: {e}")
            fail += 1
            results.append({'idx': i, 'question': question, 'error': str(e)})
            continue

        parsed = parse_chain(raw)
        all_rels = list(sample.get('relation_list', []))
        sparql_rels = extract_sparql_rels(sq.get('sparql', ''))

        # Find gold relations in subgraph
        gold_in_sub = []
        for sr in sparql_rels:
            match = find_gold_in_list(sr, all_rels)
            if match:
                gold_in_sub.append((sr, match))

        if not parsed:
            fail += 1
            print(f"  Case {i} PARSE FAIL: {raw[:120]}")
            results.append({'idx': i, 'question': question, 'raw': raw, 'parsed': None})
            continue

        ok += 1
        n = len(parsed['hops'])
        hop_dist[n] = hop_dist.get(n, 0) + 1

        anchor = parsed['anchor']
        anchor_match = any(anchor.lower() in e.lower() or e.lower() in anchor.lower() for e in q_entities)
        ner_mark = "✅" if anchor_match else "❌ NOT_IN_NER"

        hops_str = " -> ".join(f"({h['relation']}){'[inv]' if h.get('inverse') else ''}->{h['endpoint']}" for h in parsed['hops'])
        print(f"  Case {i} OK ({n} hops) [{ner_mark}]: {anchor} {hops_str}")

        # --- Matching ---
        if not gold_in_sub or not all_rels:
            results.append({
                'idx': i, 'question': question, 'q_entity': q_entities,
                'raw': raw, 'parsed': parsed, 'n_hops': n,
            })
            continue

        # Encode all relations for GTE
        r_embs = encode_batch(all_rels, batch_size=64)

        # Annotation targets
        anno = anno_map.get(i)
        anno_targets = {}
        if anno:
            for s in anno['steps']:
                anno_targets[s['step']] = s['target_gold_rels']

        lines.append(f"## Case {i}: {question}")
        lines.append(f"**Gold**: `{sample.get('a_entity', [])}` | **NER**: `{q_entities}` | **Anchor**: `{anchor}`")
        lines.append(f"**SPARQL gold rels**: `{sparql_rels}`")
        lines.append(f"**Chain**: {anchor} {hops_str}")
        lines.append("")

        for hi, hop in enumerate(parsed['hops']):
            schema_rel = hop['relation']
            total_schema_rels += 1

            # --- String matching ---
            str_match, str_type = match_schema_to_real(schema_rel, all_rels)

            if str_match:
                matched_schema_rels += 1
                print(f"    Hop {hi+1}: {schema_rel} → {str_match} ({str_type}) ✅")
            else:
                print(f"    Hop {hi+1}: {schema_rel} → NO STRING MATCH ❌")

            # --- GTE matching using schema name as query ---
            q_gte = f'Instruct: {task_desc}\nQuery: {schema_rel}'
            q_emb = encode_batch([q_gte], batch_size=1)
            scores = (r_embs @ q_emb.T).squeeze(1).cpu().tolist()
            ranked = sorted(enumerate(all_rels), key=lambda x: -scores[x[0]])

            # Compute gold ranks
            gold_ranks_str = {}
            gold_ranks_gte = {}
            for sr, sr_match in gold_in_sub:
                # String match: if our schema_rel matches this gold, rank=1
                if str_match and (sr_match == str_match or norm(schema_rel) in norm(sr_match)):
                    gold_ranks_str[sr] = 1
                # GTE rank
                for rank, (ri, name) in enumerate(ranked):
                    if name == sr_match:
                        gold_ranks_gte[sr] = rank + 1
                        break

            step_num = hi + 1
            targets = anno_targets.get(step_num, [])

            # Best string rank across targets
            best_str = min(gold_ranks_str.get(t, 999) for t in targets) if targets else (1 if gold_ranks_str else 999)
            best_gte = min(gold_ranks_gte.get(t, 999) for t in targets) if targets else min(gold_ranks_gte.values(), default=999)
            # Combined: best of string + GTE
            best_combined = min(best_str, best_gte)

            for k in ks:
                str_stats[k]['total'] += 1
                gte_stats[k]['total'] += 1
                if best_str <= k:
                    str_stats[k]['hit'] += 1
                if best_combined <= k:
                    gte_stats[k]['hit'] += 1

            def rank_icon(r):
                if r <= 5: return "✅"
                elif r <= 10: return "⚠️"
                else: return "❌"

            lines.append(f"### Hop {hi+1}: `{schema_rel}`")
            lines.append(f"- **String match**: {str_match} ({str_type})" if str_match else "- **String match**: None ❌")
            lines.append(f"- **Target**: `{targets if targets else gold_ranks_gte}`")
            lines.append(f"- **String rank**: {rank_icon(best_str)}@{best_str} | **GTE rank**: {rank_icon(best_gte)}@{best_gte} | **Best**: {rank_icon(best_combined)}@{best_combined}")
            lines.append(f"- **GTE Top-5**: `{[name for _, name in ranked[:5]]}`")
            lines.append("")

            hop['str_match'] = str_match
            hop['str_type'] = str_type
            hop['best_str_rank'] = best_str
            hop['best_gte_rank'] = best_gte
            hop['best_combined_rank'] = best_combined

        lines.append("---\n")

        results.append({
            'idx': i, 'question': question, 'q_entity': q_entities,
            'raw': raw, 'parsed': parsed, 'n_hops': n,
        })
        time.sleep(0.3)

    # Summary
    total_hops = sum(len(r['parsed']['hops']) for r in results if r.get('parsed'))
    print(f"\n{'='*60}")
    print(f"Parsing: {ok} OK, {fail} fail / {args.limit} total")
    print(f"Anchor in NER: {sum(1 for r in results if r.get('parsed') and any(r['parsed']['anchor'].lower() in e.lower() for e in r.get('q_entity',[])))}/{ok}")
    print(f"Hop distribution: {dict(sorted(hop_dist.items()))}")
    print(f"String match: {matched_schema_rels}/{total_schema_rels} schema rels matched a real relation")

    total = str_stats[5]['total']
    if total > 0:
        print(f"\nRecall Comparison ({total} hops):")
        print(f"  {'Metric':<10} {'Schema+Str':>12} {'Schema+GTE':>12}")
        print(f"  {'-'*36}")
        for k in [1, 3, 5, 10]:
            sh = str_stats[k]['hit']
            gh = gte_stats[k]['hit']
            print(f"  Hit@{k:<6} {sh/total*100:>10.1f}% {gh/total*100:>10.1f}%")

        lines.append("## Summary\n")
        lines.append(f"| Metric | Schema String | Schema+GTE (Best) |")
        lines.append(f"|---|---|---|")
        for k in [1, 3, 5, 10]:
            sh = str_stats[k]['hit']
            gh = gte_stats[k]['hit']
            lines.append(f"| Hit@{k} | {sh} ({sh/total*100:.1f}%) | {gh} ({gh/total*100:.1f}%) |")

    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    Path('/tmp/chain_schema_review.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f"\nResults -> {args.output}")
    print(f"Review -> /tmp/chain_schema_review.md")


if __name__ == '__main__':
    main()
