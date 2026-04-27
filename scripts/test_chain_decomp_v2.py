#!/usr/bin/env python3
"""Test chain-style decomposition: [Entity] -(relation)-> node -(relation)-> node ...

Format:
  [Entity] -（relation description）-> node -（relation description）-> node

Then use relation descriptions as GTE queries to test recall against gold relations.
"""

import json, re, pickle, time, sys
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

CHAIN_PROMPT = """You are a knowledge graph question decomposition expert. Decompose the given question into a chain of entity-relation hops.

## Rules
1. Start with the anchor entity in [brackets]. Choose the entity with the FEWEST possible connections (lowest fan-out) as the anchor.
2. Each hop describes a relation traversal using (parentheses) for the relation description.
3. Use Chinese for relation descriptions to capture semantic meaning concisely.
4. Each "node" represents an intermediate entity/value discovered at that hop.
5. Keep the chain minimal - only include hops that are necessary for answering the question.
6. The final node should lead to the answer.

## Output Format (exactly one line)
[Entity] -（relation desc）-> node -（relation desc）-> node

## Examples
Q: What is the predominant religion where the leader is Ovadia Yosef?
[Ovadia Yosef] -（担任的宗教领导职位）-> node -（该职位的管辖地区）-> node -（该地区的主要宗教）-> answer

Q: What country speaks Germanic languages and uses the East German mark as currency?
[East German mark] -（使用该货币的国家）-> node -（该国家的官方语言）-> answer

Q: What actor played the a kid in the movie with a character named Jenny's Father?
[Jenny's Father] -（出场电影）-> node -（该电影中扮演kid角色的演员）-> answer

Q: What educational institution has a football sports team named Northern Colorado Bears?
[Northern Colorado Bears] -（所属的学校）-> answer

Q: Who is the governor of Ohio in 2011 that was in the government prior to 2010?
[Ohio] -（2011年的州长）-> node -（该州长之前的政府职位）-> answer

Now decompose:
Q: {question}
"""


def parse_chain(text):
    """Parse chain format: [Entity] -(desc)-> node -(desc)-> node"""
    text = text.strip()
    # Remove "Answer:" prefix if present
    text = re.sub(r'^(Answer|答案|输出)[：:]\s*', '', text)

    # Extract anchor entity
    anchor_match = re.match(r'\[([^\]]+)\]', text)
    if not anchor_match:
        return None
    anchor = anchor_match.group(1)
    rest = text[anchor_match.end():]

    # Extract hops: -（desc）-> node
    hops = []
    # Match patterns like: -（desc）-> node  or  -(desc)-> node
    hop_pattern = r'-[（(]([^)）]+)[)）]\s*->\s*(\S+)'
    for m in re.finditer(hop_pattern, rest):
        rel_desc = m.group(1)
        node = m.group(2)
        hops.append({'relation': rel_desc, 'endpoint': node})

    if not hops:
        return None

    return {'anchor': anchor, 'hops': hops, 'raw': text}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=30)
    p.add_argument('--output', default='/tmp/chain_decomp_test.json')
    args = p.parse_args()

    # Load data
    pkl_map = {d['id']: d for d in pickle.load(open('/zhaoshu/subgraph/data/cwq_processed/test_fixed_relaxation.pkl', 'rb'))}
    sparql_map = {d['ID']: d for d in json.load(open('/zhaoshu/subgraph/data/cwq_sparql/test.json'))}
    pilot = json.load(open('/zhaoshu/subgraph/reports/stage_pipeline_test/cwq_50_stage_v7/pilot.json'))

    print(f"Testing chain decomposition on {args.limit} cases...")

    results = []
    parse_ok = 0
    parse_fail = 0

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
            parse_fail += 1
            results.append({'idx': i, 'question': question, 'error': str(e)})
            continue

        parsed = parse_chain(raw)

        if parsed:
            parse_ok += 1
            n_hops = len(parsed['hops'])
            rels = [h['relation'] for h in parsed['hops']]
            print(f"  Case {i} OK ({n_hops} hops): [{parsed['anchor']}] " + " -（".join([''] + rels).replace(' -（', ' -> ').replace('）->', ' | ')[:80])
        else:
            parse_fail += 1
            print(f"  Case {i} PARSE FAIL: {raw[:80]}")

        results.append({
            'idx': i,
            'question': question,
            'raw': raw,
            'parsed': parsed,
            'n_hops': len(parsed['hops']) if parsed else 0,
        })

        time.sleep(0.3)

    print(f"\n{'='*60}")
    print(f"Parsing: {parse_ok} OK, {parse_fail} fail / {args.limit} total")

    hop_counts = {}
    for r in results:
        n = r.get('n_hops', 0)
        hop_counts[n] = hop_counts.get(n, 0) + 1
    print(f"Hop distribution: {dict(sorted(hop_counts.items()))}")

    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Results -> {args.output}")


if __name__ == '__main__':
    main()
