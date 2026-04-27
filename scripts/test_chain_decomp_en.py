#!/usr/bin/env python3
"""Test chain decomposition v2: English, with endpoint, no case leakage.

Format: [Entity] -(English relation desc)-> node -(desc)-> ... -> [ENDPOINT: target]
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

CHAIN_PROMPT = """You are a knowledge graph reasoning expert. Decompose the question into a chain of entity-relation hops on a knowledge graph.

## Rules
1. Start with the anchor entity in [brackets]. Prefer the entity with the FEWEST possible connections (lowest fan-out).
2. Each hop is a relation traversal: -(relation description)-> node
3. Use concise English phrases for relation descriptions (e.g., "national anthem of", "capital city of").
4. Each "node" is an intermediate result from that hop.
5. Mark the final target clearly as: -> [ENDPOINT: what the answer is]
6. Keep the chain minimal — only include hops necessary to reach the answer.
7. Do NOT include verification/filtering hops for constraints like "most recent", "largest", "before year X". These are resolved after retrieval.

## Output (one line)
[Entity] -(relation desc)-> node -(relation desc)-> node -> [ENDPOINT: description]

## Examples
Q: What is the predominant religion where the leader is Ovadia Yosef?
[Ovadia Yosef] -(religious leadership position)-> node -(jurisdiction of that position)-> node -(religions practiced in that region)-> [ENDPOINT: predominant religion]

Q: What educational institution has a football sports team named Northern Colorado Bears?
[Northern Colorado Bears] -(school of this sports team)-> [ENDPOINT: educational institution]

Q: Who is the governor of Ohio in 2011 that was in the government prior to 2010?
[Ohio] -(governor in 2011)-> node -(government positions held before 2010)-> [ENDPOINT: person name]

Q: What country speaks Germanic languages and uses the East German mark as currency?
[East German mark] -(country using this currency)-> node -(official languages of that country)-> [ENDPOINT: country name]

Q: What actor played the kid in the movie with a character named Jenny's Father?
[Jenny's Father] -(movie featuring this character)-> node -(actor who played kid role in that movie)-> [ENDPOINT: actor name]

Q: For what event did the person educated at Dewitt High School win a gold medal?
[Dewitt High School] -(person educated at this school)-> node -(Olympic medals won by this person)-> [ENDPOINT: event name]

Now decompose:
Q: {question}
"""


def parse_chain(text):
    """Parse: [Entity] -(desc)-> node -(desc)-> ... -> [ENDPOINT: desc]"""
    text = text.strip()
    text = re.sub(r'^(Answer|答案)[：:]\s*', '', text)

    # Extract anchor
    anchor_match = re.match(r'\[([^\]]+)\]', text)
    if not anchor_match:
        return None
    anchor = anchor_match.group(1)
    rest = text[anchor_match.end():]

    # Extract endpoint
    endpoint = None
    ep_match = re.search(r'\[ENDPOINT:\s*([^\]]+)\]', rest)
    if ep_match:
        endpoint = ep_match.group(1).strip()
        rest = rest[:ep_match.start()]

    # Extract hops: -(desc)-> node  (node may be absent if ENDPOINT follows directly)
    hops = []
    hop_pattern = r'-\(([^)]+)\)\s*->\s*(\S+)?'
    for m in re.finditer(hop_pattern, rest):
        rel_desc = m.group(1)
        node = m.group(2) or 'node'
        hops.append({'relation': rel_desc, 'endpoint': node})

    if not hops:
        return None

    return {'anchor': anchor, 'hops': hops, 'endpoint': endpoint, 'raw': text}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=30)
    p.add_argument('--output', default='/tmp/chain_decomp_en.json')
    args = p.parse_args()

    pkl_map = {d['id']: d for d in pickle.load(open('/zhaoshu/subgraph/data/cwq_processed/test_fixed_relaxation.pkl', 'rb'))}
    pilot = json.load(open('/zhaoshu/subgraph/reports/stage_pipeline_test/cwq_50_stage_v7/pilot.json'))

    print(f"Testing EN chain decomposition on {args.limit} cases...")

    results = []
    ok = fail = 0
    hop_dist = {}

    for i in range(min(args.limit, len(pilot))):
        case = pilot[i]
        cid = case['case_id']
        sample = pkl_map.get(cid)
        if not sample:
            continue

        question = sample.get('question', '')
        prompt = CHAIN_PROMPT.format(question=question)
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
            ep = parsed.get('endpoint', '?')
            hops_str = " -> ".join(f"({h['relation']})" for h in parsed['hops'])
            print(f"  Case {i} OK ({n} hops): [{parsed['anchor']}] {hops_str} -> [ENDPOINT: {ep}]")
        else:
            fail += 1
            print(f"  Case {i} PARSE FAIL: {raw[:100]}")

        results.append({
            'idx': i,
            'question': question,
            'raw': raw,
            'parsed': parsed,
            'n_hops': len(parsed['hops']) if parsed else 0,
        })
        time.sleep(0.3)

    print(f"\n{'='*60}")
    print(f"Parsing: {ok} OK, {fail} fail / {args.limit} total")
    print(f"Hop distribution: {dict(sorted(hop_dist.items()))}")

    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Results -> {args.output}")


if __name__ == '__main__':
    main()
