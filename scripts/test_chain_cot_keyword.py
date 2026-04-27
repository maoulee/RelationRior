#!/usr/bin/env python3
"""Chain decomposition v6.1: CoT atomic decomposition + keyword + definition.

Changes from v6:
1. Removed hypernym — too generic, hurts GTE performance
2. Entity endpoints: named entities from list used as endpoints, not just "node"
3. Added anti-verification-hop negative example
4. GTE ablation: desc_only / def_only / kw_def (no hypernym)
"""

import json, re, pickle, time, argparse
from pathlib import Path
import requests


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
    data = resp.json()
    return data['choices'][0]['message']['content']


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

CHAIN_PROMPT_V61 = """You are a knowledge graph reasoning expert. Analyze the question and decompose it into atomic reasoning steps.

## Entities
Named entities from the question: {entities}

## CRITICAL RULES

### Rule 1: Each hop = ONE atomic graph edge
A hop traverses exactly ONE relation. Compound actions MUST be split:
- "find championship year" → TWO hops: find championship events + find event start date
- "find manufacturer's country" → TWO hops: find manufacturer + find country
- "find author's birthplace" → TWO hops: find author + find birthplace
- "find CEO of the company that makes X" → TWO hops: find manufacturer + find CEO

Attribute lookups (dates, names, counts, types) are ALWAYS separate hops.

### Rule 2: Use entity names as endpoints when possible
If a named entity from the entity list is the result of a hop, write that entity name as the endpoint instead of "node".
Examples:
- Entities: [Thames, London] → Thames -(river flowing through this city)-> London (NOT node)
- Entities: [Mona Lisa, France] → Mona Lisa -(museum housing this artwork)-> node -(country of this museum)-> France (NOT node)

### Rule 3: No verification hops for superlatives
Do NOT add hops like "find the most recent", "find the largest", "find the first". These are post-processing filters, not graph edges.
- WRONG: championship events -> most recent event -> start date
- RIGHT: championship events -> start date (filter "most recent" in post-processing)

## Step 1: CoT Reasoning (count hops BEFORE writing chain)
Think step by step:
1. What is the starting entity? (MUST be from entity list)
2. What is the FIRST piece of information I need to find? (hop 1)
3. Using that result, what is the NEXT piece of information? (hop 2)
4. Continue until I can answer the question. Count total hops needed.

## Step 2: Write the Chain
Format: EntityA -(single action description)-> Endpoint -(single action description)-> Endpoint
- Each hop describes ONE atomic action
- Use entity names from the list as endpoints when they are the result
- Use "node" for unknown endpoints

## Step 3: Per-Hop Analysis
For each hop, provide:
- Keyword: the core concept word (e.g., year, championship, museum, capital, airport)
- Definition: describe the RELATION SEMANTICS — what kind of link this relation represents. Focus on the action/nature of the relationship (containment, attribution, membership, temporal), NOT just the entity types.
  GOOD: "describes the geographic inclusion of countries within a time zone"
  GOOD: "describes the population count or demographic statistic of a region"
  GOOD: "describes the competitive titles won by a sports team"
  BAD:  "connects a time zone to the countries that use it" (entity-description, no relation semantics)
  BAD:  "connects a country to its population" (too vague, missing the statistic/attribute nature)

## Output Format
Reasoning: Starting from [entity]. First need to find [X] (hop 1). Then find [Y] from that (hop 2). Then find [Z] from that (hop 3). N hops total.

Chain:
EntityA -(action 1)-> endpoint -(action 2)-> endpoint

Analysis:
1. [action 1 description]
   - Keyword: word
   - Definition: describes the [relation semantics] of [subject]
2. [action 2 description]
   - Keyword: word
   - Definition: describes the [relation semantics] of [subject]

## CORRECT Examples

Entities: iPhone
Q: Who is the CEO of the company that makes the iPhone?
Reasoning: Start from iPhone. Hop 1: find which company manufactures it. Hop 2: find the CEO of that company. 2 hops total.

Chain:
iPhone -(manufacturer of this product)-> node -(CEO of this organization)-> node

Analysis:
1. manufacturer of this product
   - Keyword: manufacturer
   - Definition: describes the production origin of a product, linking it to the organization that manufactured it
2. CEO of this organization
   - Keyword: CEO
   - Definition: describes the top executive leadership role of an organization

Entities: Romeo and Juliet
Q: What country is the birthplace of the person who wrote Romeo and Juliet?
Reasoning: Start from Romeo and Juliet. Hop 1: find the author. Hop 2: find the birthplace of that author. Hop 3: find the country of that birthplace. 3 hops total.

Chain:
Romeo and Juliet -(author of this work)-> node -(birthplace of this person)-> node -(country of this location)-> node

Analysis:
1. author of this work
   - Keyword: author
   - Definition: describes the creative authorship linking a written work to its creator
2. birthplace of this person
   - Keyword: birthplace
   - Definition: describes the geographic origin or birth location of a person
3. country of this location
   - Keyword: country
   - Definition: describes the national jurisdiction or sovereign territory containing a geographic location

Entities: Thames, London
Q: What ocean does the river flowing through London empty into?
Reasoning: Start from Thames. The question says it flows through London, so Thames is the river and London is on its path. Hop 1: find where the river empties. 1 hop needed (London is given context, not a hop target).

Chain:
Thames -(body of water this river empties into)-> node

Analysis:
1. body of water this river empties into
   - Keyword: mouth
   - Definition: describes the outflow destination where a waterway terminates or drains into

Entities: Mona Lisa, France
Q: Which museum in France houses the painting that depicts Lisa Gherardini?
Reasoning: Start from Mona Lisa. Hop 1: find the museum housing it. Hop 2: verify the museum's country is France. 2 hops total. France is in the entity list, so use it as endpoint.

Chain:
Mona Lisa -(museum housing this artwork)-> node -(country of this museum)-> France

Analysis:
1. museum housing this artwork
   - Keyword: museum
   - Definition: describes the institutional custody or exhibition location of an artwork
2. country of this museum
   - Keyword: country
   - Definition: describes the national jurisdiction or sovereign territory containing a museum

Entities: Amazon River
Q: Which countries does the Amazon River flow through?
Reasoning: Start from Amazon River. Hop 1: directly find the countries. 1 hop total.

Chain:
Amazon River -(countries this river flows through)-> node

Analysis:
1. countries this river flows through
   - Keyword: countries
   - Definition: describes the geographic inclusion or traversal of sovereign territories by a waterway

Entities: Manchester United
Q: What year did the football team last win the Champions League?
Reasoning: Start from Manchester United. Hop 1: find championship events. Hop 2: find the start date of the event. 2 hops total. "last win" is a filter, NOT a hop.

Chain:
Manchester United -(championship events of this sports team)-> node -(start date of this event)-> node

Analysis:
1. championship events of this sports team
   - Keyword: championship
   - Definition: describes the competitive titles or tournament victories achieved by a sports team
2. start date of this event
   - Keyword: date
   - Definition: describes the temporal origin or occurrence date of an event

## WRONG vs CORRECT (DO NOT repeat these mistakes)

Q: What year did the basketball team coached by Brad Stevens win the championship?
WRONG: Brad Stevens -(championship winning year of his team)-> node
WHY WRONG: "championship winning year" combines finding championships AND extracting a date.

CORRECT:
Brad Stevens -(team coached by this person)-> node -(championship events of this team)-> node -(start date of this event)-> node

Q: What country is the manufacturer of the A380 headquartered in?
WRONG: Airbus A380 -(manufacturer headquarters country)-> node
WHY WRONG: "manufacturer headquarters country" is three edges collapsed into one.

CORRECT:
Airbus A380 -(manufacturer of this product)-> node -(headquarters country of this organization)-> node

Q: What year did the football team last win the Champions League?
WRONG: Manchester United -(championship events)-> node -(most recent event)-> node -(start date)-> node
WHY WRONG: "most recent event" is a FILTER, not a graph edge. No such relation exists in the knowledge graph.

CORRECT:
Manchester United -(championship events)-> node -(start date)-> node

Now analyze:
Entities: {entities}
Q: {question}
"""


# ---------------------------------------------------------------------------
# Parse LLM output
# ---------------------------------------------------------------------------

def parse_output(text):
    """Parse CoT + chain + analysis output."""
    text = text.strip()
    text = re.sub(r'^(Answer|答案)[：:]\s*', '', text)

    # Extract reasoning
    reasoning = ''
    reason_match = re.search(r'Reasoning:\s*(.+?)(?=\n\s*Chain:|\nChain:)', text, re.DOTALL)
    if reason_match:
        reasoning = reason_match.group(1).strip()

    # Extract chain section
    chain_match = re.search(r'Chain:\s*\n?(.*?)(?=\n\s*Analysis:|\nAnalysis:|$)', text, re.DOTALL)
    if not chain_match:
        return None

    chain_text = chain_match.group(1).strip()
    chain_lines = [l.strip() for l in chain_text.split('\n') if '-(' in l]
    if not chain_lines:
        return None
    chain_text = chain_lines[-1]

    # Parse: EntityA -(desc)-> endpoint -(desc)-> endpoint
    anchor_match = re.match(r'^(.+?)\s*-\(', chain_text)
    if not anchor_match:
        return None
    anchor = anchor_match.group(1).strip()
    rest = chain_text[anchor_match.end() - 1:]

    hops = []
    hop_pattern = r'\(([^)]+)\)\s*(?:\.inv\s*)?->\s*(.+?)(?=\s*-\(|$)'
    for m in re.finditer(hop_pattern, rest):
        rel_desc = m.group(1).strip()
        node = m.group(2).strip()
        if not node or node.lower() in ('unknown', 'the', 'a'):
            node = 'node'
        hops.append({
            'relation': rel_desc,
            'endpoint': node,
            'keyword': '',
            'definition': '',
        })

    if not hops:
        return None

    # Extract analysis section
    analysis_match = re.search(r'Analysis:\s*\n(.+)', text, re.DOTALL)
    if analysis_match:
        analysis_text = analysis_match.group(1).strip()
        kw_list = re.findall(r'-\s*Keyword:\s*(.+)', analysis_text)
        df_list = re.findall(r'-\s*Definition:\s*(.+)', analysis_text)

        for idx, hop in enumerate(hops):
            if idx < len(kw_list):
                hop['keyword'] = kw_list[idx].strip()
            if idx < len(df_list):
                hop['definition'] = df_list[idx].strip()

    return {
        'anchor': anchor,
        'hops': hops,
        'reasoning': reasoning,
        'raw': text,
        'endpoints': [h['endpoint'] for h in hops if h['endpoint'] != 'node'],
    }


# ---------------------------------------------------------------------------
# Gold relation extraction
# ---------------------------------------------------------------------------

def norm(t):
    return re.sub(r'[^a-z0-9]', '', t.lower().strip())


def extract_sparql_rels(sparql_text):
    raw = re.findall(r'ns:([a-zA-Z][a-zA-Z0-9_.]+)', sparql_text)
    return list(dict.fromkeys(r for r in raw if '.' in r and not r.startswith('m.')))


def find_gold_in_list(gold_rel, rel_list):
    gn = norm(gold_rel)
    for r in rel_list:
        rn = norm(r)
        if gn == rn or gn in rn or rn in gn:
            return r
    return None


# ---------------------------------------------------------------------------
# GTE encoding
# ---------------------------------------------------------------------------

def make_gte_encode():
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    print("Loading Qwen3-Embedding-0.6B...")
    tokenizer = AutoTokenizer.from_pretrained(
        '/zhaoshu/llm/Qwen3-Embedding-0.6B',
        padding_side='left',
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        '/zhaoshu/llm/Qwen3-Embedding-0.6B',
        trust_remote_code=True,
    ).cuda().eval()

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
                out = model(**bd)
                embs = last_token_pool(out.last_hidden_state, bd['attention_mask'])
            all_embs.append(F.normalize(embs, p=2, dim=1))
        return torch.cat(all_embs, dim=0)

    return encode_batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=30)
    parser.add_argument('--output', default='/tmp/chain_cot_keyword.json')
    parser.add_argument('--review', default='/tmp/chain_cot_keyword_review.md')
    args = parser.parse_args()

    # Load data
    pkl_map = {d['id']: d for d in pickle.load(open('/zhaoshu/subgraph/data/cwq_processed/test_literal_and_language_fixed.pkl', 'rb'))}
    sparql_map = {d['ID']: d for d in json.load(open('/zhaoshu/subgraph/data/cwq_sparql/test.json'))}
    pilot = json.load(open('/zhaoshu/subgraph/reports/stage_pipeline_test/cwq_50_stage_v7/pilot.json'))

    # Load annotations if available
    anno_path = '/tmp/annotation_output.json'
    annotations = json.load(open(anno_path)) if Path(anno_path).exists() else []
    anno_map = {a['idx']: a for a in annotations}

    # GTE
    encode_batch = make_gte_encode()
    task_desc = "Given a knowledge graph question, retrieve relevant graph relations that answer the question"

    # Ablation: 3 GTE query formats (no hypernym)
    ks = [1, 3, 5, 10]
    ablation_keys = ['desc_only', 'def_only', 'kw_def']
    ablation_stats = {k: {ak: {'hit': 0, 'total': 0} for ak in ablation_keys} for k in ks}

    results = []
    ok = fail = 0
    hop_dist = {}
    anchor_in_ner = 0
    entity_endpoint_count = 0
    total_endpoint_count = 0

    lines = []
    lines.append("# CoT Atomic Decomposition v6.1 — Keyword+Definition (No Hypernym)")
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

        prompt = CHAIN_PROMPT_V61.format(entities=ent_str, question=question)
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = call_llm(messages)
        except Exception as e:
            print(f"  Case {i}: API error: {e}")
            fail += 1
            results.append({'idx': i, 'question': question, 'error': str(e)})
            continue

        parsed = parse_output(raw)
        all_rels = list(sample.get('relation_list', []))
        sparql_rels = extract_sparql_rels(sq.get('sparql', ''))

        gold_in_sub = []
        for sr in sparql_rels:
            match = find_gold_in_list(sr, all_rels)
            if match:
                gold_in_sub.append((sr, match))

        if not parsed:
            fail += 1
            print(f"  Case {i} PARSE FAIL: {raw[:150]}")
            results.append({'idx': i, 'question': question, 'raw': raw, 'parsed': None})
            continue

        ok += 1
        n = len(parsed['hops'])
        hop_dist[n] = hop_dist.get(n, 0) + 1

        anchor = parsed['anchor']
        anchor_match = any(anchor.lower() in e.lower() or e.lower() in anchor.lower() for e in q_entities)
        if anchor_match:
            anchor_in_ner += 1

        # Count entity endpoints
        for hop in parsed['hops']:
            total_endpoint_count += 1
            ep = hop['endpoint']
            if ep != 'node' and any(ep.lower() in e.lower() or e.lower() in ep.lower() for e in q_entities):
                entity_endpoint_count += 1

        ner_mark = "OK" if anchor_match else "NOT_NER"
        ent_eps = [h['endpoint'] for h in parsed['hops'] if h['endpoint'] != 'node']
        hops_str = " -> ".join(f"({h['relation']})->{h['endpoint']}" for h in parsed['hops'])
        print(f"  Case {i} ({n}h) [{ner_mark}] eps={ent_eps or 'none'}: {anchor} {hops_str}")

        for hi, hop in enumerate(parsed['hops']):
            kw = hop.get('keyword', '')
            df = hop.get('definition', '')
            status = "OK" if (kw and df) else "INCOMPLETE"
            print(f"    H{hi+1}: kw={kw or '-'} def={df[:50] if df else '-'} [{status}]")

        if not gold_in_sub or not all_rels:
            results.append({
                'idx': i, 'question': question, 'q_entity': q_entities,
                'raw': raw, 'parsed': parsed, 'n_hops': n,
            })
            continue

        r_embs = encode_batch(all_rels, batch_size=64)

        # Gold targets: use ALL SPARQL relations (annotation is unreliable)
        # Each hop checks against all gold relations — best rank across any gold match
        lines.append(f"## Case {i}: {question}")
        lines.append(f"**Gold**: `{sample.get('a_entity', [])}` | **NER**: `{q_entities}` | **Anchor**: `{anchor}`")
        lines.append(f"**SPARQL gold rels**: `{sparql_rels}`")
        lines.append(f"**CoT**: {parsed.get('reasoning', 'N/A')[:150]}")
        lines.append(f"**Chain**: {anchor} {hops_str}")
        lines.append("")

        for hi, hop in enumerate(parsed['hops']):
            rel_desc = hop['relation']
            kw = hop.get('keyword', '')
            df = hop.get('definition', '')

            # Construct 3 GTE queries
            queries = {
                'desc_only': f'Instruct: {task_desc}\nQuery: {rel_desc}',
                'def_only': f'Instruct: {task_desc}\nQuery: {df}' if df else None,
                'kw_def': f'Instruct: {task_desc}\nQuery: {kw} {df}' if (kw and df) else None,
            }

            step_num = hi + 1

            ablation_ranks = {}
            for akey, q_text in queries.items():
                if not q_text:
                    continue
                q_emb = encode_batch([q_text], batch_size=1)
                scores = (r_embs @ q_emb.T).squeeze(1).cpu().tolist()
                ranked = sorted(enumerate(all_rels), key=lambda x: -scores[x[0]])

                # Best rank across ALL SPARQL gold relations in subgraph
                best_gold_rank = 999
                for sr, sr_match in gold_in_sub:
                    for rank, (ri, name) in enumerate(ranked):
                        if name == sr_match:
                            if rank + 1 < best_gold_rank:
                                best_gold_rank = rank + 1
                            break

                ablation_ranks[akey] = best_gold_rank

                for k in ks:
                    ablation_stats[k][akey]['total'] += 1
                    if best_gold_rank <= k:
                        ablation_stats[k][akey]['hit'] += 1

            def ri(r):
                if r <= 5: return "OK"
                elif r <= 10: return "WARN"
                else: return "MISS"

            rank_strs = []
            for akey in ablation_keys:
                r = ablation_ranks.get(akey, 999)
                rank_strs.append(f"{akey}={ri(r)}@{r}")
            lines.append(f"### Hop {hi+1}: `{rel_desc}` -> {hop['endpoint']}")
            lines.append(f"- **Keyword**: {kw} | **Definition**: {df[:100] if df else 'N/A'}")
            lines.append(f"- **SPARQL gold**: `{[g[0] for g in gold_in_sub]}`")
            lines.append(f"- **Ranks**: {' | '.join(rank_strs)}")

            q_ref = queries.get('desc_only', '')
            if q_ref:
                q_emb = encode_batch([q_ref], batch_size=1)
                scores = (r_embs @ q_emb.T).squeeze(1).cpu().tolist()
                ranked = sorted(enumerate(all_rels), key=lambda x: -scores[x[0]])
                lines.append(f"- **Top-5**: `{[name for _, name in ranked[:5]]}`")
            lines.append("")

            hop['ablation_ranks'] = ablation_ranks

        lines.append("---\n")

        results.append({
            'idx': i, 'question': question, 'q_entity': q_entities,
            'raw': raw, 'parsed': parsed, 'n_hops': n,
        })
        time.sleep(0.3)

    # ---- Summary ----
    total_hops = sum(len(r['parsed']['hops']) for r in results if r.get('parsed'))
    hops_with_kw = sum(1 for r in results if r.get('parsed') for h in r['parsed']['hops'] if h.get('keyword'))
    hops_with_df = sum(1 for r in results if r.get('parsed') for h in r['parsed']['hops'] if h.get('definition'))

    print(f"\n{'='*70}")
    print(f"Parsing: {ok} OK, {fail} fail / {args.limit} total")
    print(f"Anchor in NER: {anchor_in_ner}/{ok}")
    print(f"Entity endpoints: {entity_endpoint_count}/{total_endpoint_count} hops use entity from NER list")
    print(f"Hop distribution: {dict(sorted(hop_dist.items()))}")
    print(f"Analysis: kw={hops_with_kw}/{total_hops} def={hops_with_df}/{total_hops}")

    total = ablation_stats[5]['desc_only']['total']
    if total > 0:
        print(f"\nGTE Query Format Ablation ({total} hops):")
        print(f"  {'Format':<16}", end='')
        for k in [1, 3, 5, 10]:
            print(f" {'Hit@'+str(k):>10}", end='')
        print()
        print(f"  {'-'*60}")

        for akey, label in [('desc_only', 'Description'), ('def_only', 'Definition'),
                             ('kw_def', 'Keyword+Def')]:
            print(f"  {label:<16}", end='')
            for k in [1, 3, 5, 10]:
                s = ablation_stats[k][akey]
                pct = s['hit'] / s['total'] * 100 if s['total'] > 0 else 0
                print(f" {s['hit']:>3}({pct:>5.1f}%)", end='')
            print()

        lines.append("## Summary\n")
        lines.append(f"Parsing: {ok} OK, {fail} fail / {args.limit} total")
        lines.append(f"Anchor in NER: {anchor_in_ner}/{ok}")
        lines.append(f"Entity endpoints: {entity_endpoint_count}/{total_endpoint_count}")
        lines.append(f"Hop distribution: {dict(sorted(hop_dist.items()))}")
        lines.append(f"Analysis: kw={hops_with_kw}/{total_hops} def={hops_with_df}/{total_hops}")
        lines.append("")
        lines.append("| Format | Hit@1 | Hit@3 | Hit@5 | Hit@10 |")
        lines.append("|---|---|---|---|---|")
        for akey, label in [('desc_only', 'Description'), ('def_only', 'Definition'),
                             ('kw_def', 'Keyword+Def')]:
            cells = []
            for k in [1, 3, 5, 10]:
                s = ablation_stats[k][akey]
                pct = s['hit'] / s['total'] * 100 if s['total'] > 0 else 0
                cells.append(f"{s['hit']} ({pct:.1f}%)")
            lines.append(f"| {label} | {' | '.join(cells)} |")

        # Failure analysis
        lines.append("\n## Failure Analysis (all methods miss Hit@5)\n")
        fail_count = 0
        for r in results:
            if not r.get('parsed'):
                continue
            for hi, hop in enumerate(r['parsed']['hops']):
                ar = hop.get('ablation_ranks', {})
                if all(ar.get(ak, 999) > 5 for ak in ablation_keys):
                    fail_count += 1
                    lines.append(
                        f"- Case {r['idx']} H{hi+1}: `{hop['relation']}` | "
                        f"Q: {r['question'][:60]} | ranks={ar}"
                    )
        if fail_count == 0:
            lines.append("All hops achieve Hit@5 in at least one method.")
        else:
            lines.append(f"\nTotal failures: {fail_count}")

        # Win analysis
        lines.append("\n## Win Analysis\n")
        for akey1, akey2, label in [
            ('kw_def', 'desc_only', 'Kw+Def vs Desc'),
            ('kw_def', 'def_only', 'Kw+Def vs Def'),
            ('desc_only', 'def_only', 'Desc vs Def'),
        ]:
            wins_1 = wins_2 = ties = 0
            for r in results:
                if not r.get('parsed'):
                    continue
                for hop in r['parsed']['hops']:
                    ar = hop.get('ablation_ranks', {})
                    r1 = ar.get(akey1, 999)
                    r2 = ar.get(akey2, 999)
                    if r1 < r2:
                        wins_1 += 1
                    elif r2 < r1:
                        wins_2 += 1
                    else:
                        ties += 1
            lines.append(f"**{label}**: {akey1} wins={wins_1}, {akey2} wins={wins_2}, ties={ties}")

    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    Path(args.review).write_text('\n'.join(lines), encoding='utf-8')
    print(f"\nResults -> {args.output}")
    print(f"Review -> {args.review}")


if __name__ == '__main__':
    main()
