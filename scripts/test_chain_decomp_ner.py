#!/usr/bin/env python3
"""Chain decomposition v4: with NER entity constraints + relation definitions.

Entity anchor and endpoints MUST come from the provided NER entity list.
After chain decomposition, generates generic definitions for each relation.
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

CHAIN_PROMPT_NER = """You are a knowledge graph reasoning expert. Decompose the question into a chain of relation hops, then provide a generic definition for each relation.

## Entities
Named entities from the question: {entities}

## Rules
1. Start with ONE entity as anchor. Prefer specific entities over generic ones (e.g., a person name over "country"). MUST be from the entity list.
2. Each hop: -(relation description)-> node
3. Use specific, descriptive English for relation descriptions to capture the reasoning context.
4. ONE relation per hop. Each hop is a SINGLE graph edge. Split compound steps into separate hops.
5. Nodes: ALWAYS write "node" for unknowns. Only write an entity name if it is from the provided entity list. Do NOT write entity names not in the list.
6. Keep chain minimal. Do NOT include verification hops for "most recent", "largest", "before year X".
7. Do NOT invent entity names. Only use entities from the list.

## Definitions
After the chain, write a GENERIC, DOMAIN-INDEPENDENT definition for each relation. The definition must:
- Be a universal definition using only generic entity types (e.g., "event", "person", "location", "organization"), NOT domain-specific terms (e.g., NOT "championship", "athlete", "aircraft")
- Describe what it connects in general terms (e.g., "start date of an event" NOT "year a championship was won")
- Mirror the naming style of knowledge graph schema relations (e.g., "connects an event to its start date", "connects an organization to its headquarter location")
- Avoid ANY reference to the specific question domain

## Output Format
EntityA -(relation 1)-> node -(relation 2)-> node

Definitions:
1. relation 1: <generic definition>
2. relation 2: <generic definition>

## Examples

Entities: Serena Williams, Nike
Q: Which brand sponsors the athlete who won the most Wimbledon singles titles?
Serena Williams -(sponsorship brand of this person)-> Nike

Definitions:
1. sponsorship brand of this person: connects a person to a brand or company that sponsors them

Entities: Airbus A380
Q: What country is the manufacturer of the A380 headquartered in?
Airbus A380 -(manufacturer of this product)-> node -(headquarters country of this organization)-> node

Definitions:
1. manufacturer of this product: connects a product to the organization that produces it
2. headquarters country of this organization: connects an organization to the country of its main office

Entities: Thames, London
Q: What ocean does the river flowing through London empty into?
Thames -(waterway passing through this place)-> London -(outlet body of water)-> node

Definitions:
1. waterway passing through this place: connects a waterway to a populated place it passes through
2. outlet body of water: connects a waterway to the body of water it flows into

Entities: Mona Lisa, France
Q: Which museum in France houses the painting that depicts Lisa Gherardini?
Mona Lisa -(institution housing this item)-> node -(country of this institution)-> France

Definitions:
1. institution housing this item: connects an item or exhibit to the institution that holds it
2. country of this institution: connects an institution to the country where it is located

Now decompose:
Entities: {entities}
Q: {question}
"""


def parse_chain(text):
    text = text.strip()
    text = re.sub(r'^(Answer|答案)[：:]\s*', '', text)

    # Split chain line and definitions
    parts = re.split(r'\n\s*Definitions:\s*\n', text, maxsplit=1)
    chain_text = parts[0].strip()
    def_text = parts[1].strip() if len(parts) > 1 else ''

    # Parse chain
    anchor_match = re.match(r'^(.+?)\s*-\(', chain_text)
    if not anchor_match:
        return None
    anchor = anchor_match.group(1).strip()
    rest = chain_text[anchor_match.end() - 1:]

    hops = []
    hop_pattern = r'\(([^)]+)\)\s*->\s*(.+?)(?=\s*-\(|$)'
    for m in re.finditer(hop_pattern, rest):
        rel_desc = m.group(1)
        node = m.group(2).strip()
        if not node:
            node = 'node'
        hops.append({'relation': rel_desc, 'endpoint': node})

    if not hops:
        return None

    # Parse definitions
    definitions = {}
    def_pattern = r'^\d+\.\s*(.+?):\s*(.+)$'
    for line in def_text.split('\n'):
        dm = re.match(def_pattern, line.strip())
        if dm:
            rel_name = dm.group(1).strip()
            rel_def = dm.group(2).strip()
            definitions[rel_name] = rel_def

    # Match definitions to hops
    for hop in hops:
        rel = hop['relation']
        if rel in definitions:
            hop['definition'] = definitions[rel]
        else:
            # Fuzzy match: check if definition key contains or is contained in relation
            for dname, dval in definitions.items():
                if rel.lower() in dname.lower() or dname.lower() in rel.lower():
                    hop['definition'] = dval
                    break
            else:
                hop['definition'] = ''

    return {'anchor': anchor, 'hops': hops, 'raw': text, 'endpoints': [h['endpoint'] for h in hops if h['endpoint'] != 'node']}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=30)
    p.add_argument('--output', default='/tmp/chain_decomp_ner.json')
    args = p.parse_args()

    pkl_map = {d['id']: d for d in pickle.load(open('/zhaoshu/subgraph/data/cwq_processed/test_literal_and_language_fixed.pkl', 'rb'))}
    pilot = json.load(open('/zhaoshu/subgraph/reports/stage_pipeline_test/cwq_50_stage_v7/pilot.json'))

    print(f"Testing NER chain decomposition (v4: with definitions) on {args.limit} cases...")

    results = []
    ok = fail = 0
    hop_dist = {}
    anchor_in_ner = 0

    for i in range(min(args.limit, len(pilot))):
        case = pilot[i]
        cid = case['case_id']
        sample = pkl_map.get(cid)
        if not sample:
            continue

        question = sample.get('question', '')
        q_entities = sample.get('q_entity', [])

        ent_str = ", ".join(q_entities) if q_entities else "none"

        prompt = CHAIN_PROMPT_NER.format(entities=ent_str, question=question)
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = call_llm(messages)
        except Exception as e:
            print(f"  Case {i}: API error: {e}")
            fail += 1
            results.append({'idx': i, 'question': question, 'error': str(e)})
            continue

        parsed = parse_chain(raw)

        if parsed:
            ok += 1
            n = len(parsed['hops'])
            hop_dist[n] = hop_dist.get(n, 0) + 1

            anchor = parsed['anchor']
            anchor_match = any(anchor.lower() in e.lower() or e.lower() in anchor.lower() for e in q_entities)
            if anchor_match:
                anchor_in_ner += 1

            hops_str = " -> ".join(f"({h['relation']})->{h['endpoint']}" for h in parsed['hops'])
            ner_mark = "✅" if anchor_match else "❌ NOT_IN_NER"
            defs_found = sum(1 for h in parsed['hops'] if h.get('definition'))
            print(f"  Case {i} OK ({n} hops, {defs_found} defs) [{ner_mark}]: {anchor} {hops_str}")
            for hi, hop in enumerate(parsed['hops']):
                d = hop.get('definition', '')
                if d:
                    print(f"    Def {hi+1}: {d[:80]}")
        else:
            fail += 1
            print(f"  Case {i} PARSE FAIL: {raw[:120]}")

        results.append({
            'idx': i,
            'question': question,
            'q_entity': q_entities,
            'raw': raw,
            'parsed': parsed,
            'n_hops': len(parsed['hops']) if parsed else 0,
        })
        time.sleep(0.3)

    print(f"\n{'='*60}")
    print(f"Parsing: {ok} OK, {fail} fail / {args.limit} total")
    print(f"Anchor in NER: {anchor_in_ner}/{ok}")
    print(f"Hop distribution: {dict(sorted(hop_dist.items()))}")

    total_hops = sum(len(r['parsed']['hops']) for r in results if r.get('parsed'))
    total_defs = sum(1 for r in results if r.get('parsed') for h in r['parsed']['hops'] if h.get('definition'))
    print(f"Definitions: {total_defs}/{total_hops} hops have definitions")

    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Results -> {args.output}")


if __name__ == '__main__':
    main()
