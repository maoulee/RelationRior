#!/usr/bin/env python3
"""End-to-end test for Agent 3.5 (Winner+Filter) → Agent 4 (Final Answer).

Uses existing v3 results to get real backend data (plan output, action results,
filter results), feeds them through new Agent 3.5 and Agent 4 prompts,
and scores final answers with F1 against ground truth.
"""

import json
import os
import re
import sys

sys.path.insert(0, '/zhaoshu/subgraph')
from config.subagent_prompts import (
    AGENT35_SYSTEM, AGENT35_USER_TEMPLATE,
    AGENT37_SYSTEM, AGENT37_USER_TEMPLATE,
    AGENT4_SYSTEM_A, AGENT4_USER_TEMPLATE_A,
    AGENT4_SYSTEM_B, AGENT4_USER_TEMPLATE,
)

import requests

LLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = os.getenv("TEST_MODEL", "qwen35-9b-local")
RESULTS_PATH = "/zhaoshu/subgraph/reports/skill_v3_test/v3_full_1494/results.json"


# --- Utility ---

def normalize(s):
    if not isinstance(s, str):
        s = str(s)
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch.isalnum() or ch in " .-_")
    return " ".join(s.split())


def call_llm(system_prompt, user_content, temperature=0.3, max_tokens=1024):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = requests.post(LLM_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def compute_f1(pred_list, gt_list):
    pred_norm = {normalize(e) for e in pred_list}
    gt_norm = {normalize(g) for g in gt_list}
    if not pred_norm and not gt_norm:
        return 1.0
    if not pred_norm or not gt_norm:
        return 0.0
    common = pred_norm & gt_norm
    p = len(common) / len(pred_norm)
    r = len(common) / len(gt_norm)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def extract_boxed_answers(text):
    """Extract entities from \\boxed{...} in Agent 4 output."""
    answers = re.findall(r'\\boxed\{([^}]+)\}', text)
    return [a.strip() for a in answers if a.strip()]


def extract_candidates_from_action(response_text):
    """Extract entities from action backend response.

    Priority: CVT-Expanded Entities > Leaf Entities > Target Entities.
    CVT-Expanded contains human-readable names extracted from CVT nodes.
    """
    # Try CVT-Expanded first (human-readable names from CVT properties)
    for pat in [
        r'CVT-Expanded Entities\s*\(\d+[^)]*\):\s*\n\s*\[([^\]]+)\]',
    ]:
        match = re.search(pat, response_text)
        if match:
            entities = _parse_entity_list(match.group(1))
            if entities:
                return entities

    # Fallback to Leaf Entities
    for pat in [
        r'Leaf Entities\s*\(\d+[^)]*\):\s*\n\s*\[([^\]]+)\]',
        r'Target Entities\s*:?\s*\n\s*\[([^\]]+)\]',
    ]:
        match = re.search(pat, response_text)
        if match:
            entities = _parse_entity_list(match.group(1))
            if entities:
                return entities
    return []


def _parse_entity_list(list_str):
    """Parse a bracketed entity list string into a list of entity names."""
    entities = []
    for item in re.findall(r"'([^']*)'|\"([^\"]*)\"", list_str):
        val = item[0] or item[1]
        if val and val.strip():
            entities.append(val.strip())
    if not entities:
        for part in list_str.split(','):
            cleaned = part.strip().strip("'\"")
            if cleaned:
                entities.append(cleaned)
    return entities


# --- Data Extraction ---

def extract_case_data(case):
    """Extract plan, actions, filters from a v3 result case."""
    question = case.get('question', '')
    gt = case.get('ground_truth', [])

    plan_text = ''
    actions = []  # list of {path_str, entities, response_text}
    filters = []  # list of {args, response_text}
    sub_questions = []  # from original trajectory

    for turn in case.get('trajectory', []):
        eq = turn.get('executed_queries', [])
        br = turn.get('backend_results', [])

        for r in br:
            if r.get('tool_name') == 'plan' and not plan_text:
                plan_text = r.get('response_text', '')

        for q, r in zip(eq, br):
            tool = q.get('tool_name', '')

            if tool == 'action':
                path = q.get('arguments', {}).get('path', [])
                path_str = " -> ".join(
                    f"{p.get('relation', '?')}({p.get('direction', '?')})" for p in path
                )
                entities = extract_candidates_from_action(r.get('response_text', ''))
                actions.append({
                    'path_str': path_str,
                    'entities': entities,
                    'response_text': r.get('response_text', ''),
                })

            if tool == 'filter':
                filters.append({
                    'args': q.get('arguments', {}),
                    'response_text': r.get('response_text', ''),
                })

    return {
        'question': question,
        'ground_truth': gt,
        'plan_text': plan_text,
        'actions': actions,
        'filters': filters,
    }


def format_action_results_for_agent35(actions):
    """Format action results for Agent 3.5 input."""
    lines = []
    for i, act in enumerate(actions):
        aid = f"A{i+1}"
        lines.append(f"► Action {aid}:")
        lines.append(f"  Path: {act['path_str']}")
        lines.append(f"  Entities ({len(act['entities'])}): {', '.join(act['entities'][:20])}")
        if len(act['entities']) > 20:
            lines.append(f"  ... and {len(act['entities'])-20} more")
        # Include Node Details and CVT-Expanded sections from raw response
        resp = act['response_text']
        # Extract Node Details section
        node_details_match = re.search(r'\[Node Details\]:.*?(?=\n-{5,}|\nCVT-Expanded|\Z)', resp, re.DOTALL)
        if node_details_match:
            for sl in node_details_match.group(0).strip().split('\n')[:20]:
                lines.append(f"  {sl}")
        # Extract CVT-Expanded Entities
        cvt_match = re.search(r'CVT-Expanded Entities\s*\(\d+[^)]*\):\s*\n\s*\[([^\]]+)\]', resp)
        if cvt_match:
            lines.append(f"  CVT-Expanded Entities: {cvt_match.group(0).split(chr(10))[-1].strip()}")
        lines.append("")
    return "\n".join(lines)


def format_filter_results_for_agent4(filters):
    """Format filter results for Agent 4 input."""
    if not filters:
        return "No filter results available."
    lines = []
    for i, f in enumerate(filters):
        lines.append(f"Filter {i+1}:")
        lines.append(f"  Args: {json.dumps(f['args'], ensure_ascii=False)}")
        lines.append(f"  Results: {f['response_text'][:500]}")
        lines.append("")
    return "\n".join(lines)


# --- Test Cases Selection ---

def select_test_cases(all_results, count=100):
    """Select diverse test cases with action results that have relevant entities."""
    cases = []
    seen_questions = set()

    # Only select cases where at least one action has F1>0 (contains GT entity)
    for case in all_results:
        data = extract_case_data(case)
        if data['question'] in seen_questions:
            continue
        if not data['actions']:
            continue
        # Check if any action has relevant entities
        has_relevant = False
        for act in data['actions']:
            if compute_f1(act['entities'], data['ground_truth']) > 0:
                has_relevant = True
                break
        if not has_relevant:
            continue
        cases.append((case, data))
        seen_questions.add(data['question'])

    # Prioritize: multi-action > with-filters > single-action
    multi = [c for c in cases if len(c[1]['actions']) >= 2]
    with_filters = [c for c in cases if len(c[1]['filters']) >= 1 and c not in multi]
    single = [c for c in cases if c not in multi and c not in with_filters]

    ordered = multi + with_filters + single
    return ordered[:count]


def main():
    with open(RESULTS_PATH) as f:
        all_results = json.load(f)

    test_cases = select_test_cases(all_results, count=100)

    print("=" * 80)
    print("Agent 3.5 → Agent 4 End-to-End Test")
    print(f"Model: {MODEL} | Cases: {len(test_cases)}")
    print("=" * 80)

    results = {}
    for idx, (case, data) in enumerate(test_cases):
        cid = case.get('case_id', f'case_{idx}')
        question = data['question']
        gt = data['ground_truth']

        # Compact header
        print(f"\n[{idx+1}/{len(test_cases)}] {cid} | {question[:60]}")
        print(f"  GT: {gt[:3]}{'...' if len(gt)>3 else ''} | Actions: {len(data['actions'])} | Filters: {len(data['filters'])}")

        # Show action F1 summary (compact)
        action_f1s = []
        for i, act in enumerate(data['actions']):
            f1 = compute_f1(act['entities'], gt)
            action_f1s.append(f"A{i+1}={f1:.1f}")
        best_action_f1 = max(compute_f1(act['entities'], gt) for act in data['actions'])
        print(f"  Actions: {', '.join(action_f1s)} (best={best_action_f1:.2f})")

        try:
            # --- Agent 3.5: Single-Winner Selection ---
            action_results_text = format_action_results_for_agent35(data['actions'])

            sub_questions_text = "No FOLLOW-UP sub-questions."
            if '<sub_questions>' in data['plan_text']:
                sq_match = re.search(r'<sub_questions>(.*?)</sub_questions>',
                                     data['plan_text'], re.DOTALL)
                if sq_match:
                    sub_questions_text = sq_match.group(1).strip()

            user_35 = AGENT35_USER_TEMPLATE.format(
                question=question,
                execution_results=action_results_text,
                sub_questions=sub_questions_text,
            )

            resp_35 = call_llm(AGENT35_SYSTEM, user_35, max_tokens=1024)

            # Parse winner action ID
            winner_line = ""
            winner_aid = None
            for line in resp_35.split('\n'):
                if '[WINNER]' in line:
                    winner_line = line.strip()[:80]
                    m = re.search(r'\b(A\d+)\b', line)
                    if m:
                        winner_aid = m.group(1)
                    break

            # Extract candidates from Agent 3.5 output
            candidates_match = re.search(r'<candidates>\s*\n(.*?)\n\s*</candidates>',
                                          resp_35, re.DOTALL)
            if candidates_match:
                candidates = [l.strip().lstrip('- ') for l in candidates_match.group(1).split('\n')
                              if l.strip().startswith('- ')]
            else:
                # Fallback: use winning action's entities (or all if no winner parsed)
                if winner_aid:
                    aidx = int(winner_aid[1:]) - 1
                    candidates = data['actions'][aidx]['entities'] if aidx < len(data['actions']) else []
                else:
                    candidates = []
                    for act in data['actions']:
                        candidates.extend(act['entities'])

            # Parse post_filter flag
            post_filter_match = re.search(r'<post_filter>\s*(YES|NO)\s*</post_filter>',
                                           resp_35, re.IGNORECASE)
            post_filter = post_filter_match.group(1).upper() if post_filter_match else "NO"

            # --- Build context from winning action only ---
            if winner_aid:
                aidx = int(winner_aid[1:]) - 1
                winner_text = data['actions'][aidx]['response_text'][:2000] if aidx < len(data['actions']) else ""
            else:
                # No winner parsed — merge all
                winner_text = ""
                for act in data['actions']:
                    winner_text += act['response_text'][:800] + "\n---\n"
                winner_text = winner_text[:2000]

            # --- Agent 3.7: skipped (not yet implemented) ---
            attr_values_text = "Not available (post-filter not yet implemented)."

            # --- ABLATION: Run BOTH Agent 4 variants on same candidates ---

            # Variant A: minimal context (question + candidates + KG data only)
            user_4a = AGENT4_USER_TEMPLATE_A.format(
                question=question,
                candidates="\n".join(f"- {c}" for c in candidates),
                kg_data=winner_text,
            )
            resp_4a = call_llm(AGENT4_SYSTEM_A, user_4a, max_tokens=1024)
            pred_a = extract_boxed_answers(resp_4a)
            f1_a = compute_f1(pred_a, gt)

            # Variant B: original full context (question + plan + filter + candidates)
            user_4b = AGENT4_USER_TEMPLATE.format(
                question=question,
                execution_results=winner_text,
                attr_values=attr_values_text,
                candidates="\n".join(f"- {c}" for c in candidates),
            )
            resp_4b = call_llm(AGENT4_SYSTEM_B, user_4b, max_tokens=1536)
            pred_b = extract_boxed_answers(resp_4b)
            f1_b = compute_f1(pred_b, gt)

            # Compact output
            sa = "✓" if f1_a >= 0.99 else "△" if f1_a > 0 else "✗"
            sb = "✓" if f1_b >= 0.99 else "△" if f1_b > 0 else "✗"
            best = "A" if f1_a >= f1_b else "B"
            print(f"  3.5: {winner_line[:60]}")
            print(f"    Cands={len(candidates)} PF={post_filter}")
            print(f"    4A{sa} F1={f1_a:.2f} Pred: {pred_a[:3]}{'...' if len(pred_a)>3 else ''}")
            print(f"    4B{sb} F1={f1_b:.2f} Pred: {pred_b[:3]}{'...' if len(pred_b)>3 else ''}")
            print(f"    Best: 4{best}")

            results[cid] = {
                "question": question,
                "ground_truth": gt,
                "candidates_35": candidates,
                "winner_aid": winner_aid,
                "post_filter": post_filter,
                "f1_a": f1_a,
                "pred_a": pred_a,
                "f1_b": f1_b,
                "pred_b": pred_b,
                "best_variant": best,
                "response_35": resp_35,
                "response_4a": resp_4a,
                "response_4b": resp_4b,
                "n_actions": len(data['actions']),
                "n_filters": len(data['filters']),
            }

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[cid] = {"question": question, "error": str(e)}

    # --- Summary ---
    print(f"\n\n{'=' * 80}")
    print("ABLATION SUMMARY: Single-Winner 3.5 × Agent 4 Variant A vs B")
    print(f"{'=' * 80}")

    valid = {k: v for k, v in results.items() if 'f1_a' in v}
    total = len(valid)

    if total > 0:
        f1s_a = [v['f1_a'] for v in valid.values()]
        f1s_b = [v['f1_b'] for v in valid.values()]
        exact_a = sum(1 for f in f1s_a if f >= 0.99)
        exact_b = sum(1 for f in f1s_b if f >= 0.99)
        hits_a = sum(1 for f in f1s_a if f > 0)
        hits_b = sum(1 for f in f1s_b if f > 0)
        avg_a = sum(f1s_a) / len(f1s_a)
        avg_b = sum(f1s_b) / len(f1s_b)

        print(f"\n{'Metric':<30} {'4A':>8} {'4B':>8} {'Delta':>8}")
        print("-" * 58)
        print(f"{'Cases':<30} {total:>8} {total:>8}")
        print(f"{'Exact match (F1=1.0)':<30} {exact_a:>8} {exact_b:>8} {exact_b-exact_a:>+8}")
        print(f"{'Partial match (F1>0)':<30} {hits_a:>8} {hits_b:>8} {hits_b-hits_a:>+8}")
        print(f"{'Avg F1':<30} {avg_a:>8.3f} {avg_b:>8.3f} {avg_b-avg_a:>+8.3f}")

        # Per-case delta analysis
        a_wins = sum(1 for v in valid.values() if v['f1_a'] > v['f1_b'])
        b_wins = sum(1 for v in valid.values() if v['f1_b'] > v['f1_a'])
        ties = sum(1 for v in valid.values() if v['f1_a'] == v['f1_b'])
        print(f"\nPer-case: A wins={a_wins}, B wins={b_wins}, ties={ties}")

        # Show biggest deltas
        deltas = [(cid, v['f1_a'] - v['f1_b'], v['f1_a'], v['f1_b'], v['question'])
                  for cid, v in valid.items()]
        deltas.sort(key=lambda x: x[1])

        print(f"\nTop 10 cases where B > A:")
        for cid, delta, fa, fb, q in deltas[:10]:
            if delta >= 0:
                break
            print(f"  {cid:<20} 4A={fa:.2f} 4B={fb:.2f} delta={delta:+.2f} | {q[:50]}")

        print(f"\nTop 10 cases where A > B:")
        for cid, delta, fa, fb, q in reversed(deltas[:10]):
            if delta <= 0:
                break
            print(f"  {cid:<20} 4A={fa:.2f} 4B={fb:.2f} delta={delta:+.2f} | {q[:50]}")

        # Per-case breakdown sorted by best variant
        print(f"\n{'CID':<20} {'4A':>5} {'4B':>5} {'Best':>5}  Question")
        print("-" * 75)
        for cid, v in sorted(valid.items(), key=lambda x: -(x[1]['f1_a'] - x[1]['f1_b'])):
            print(f"{cid:<20} {v['f1_a']:>5.2f} {v['f1_b']:>5.2f} {'4'+v['best_variant']:>5}  {v['question'][:50]}")

    # Save
    out = "/zhaoshu/subgraph/scripts/agent35_4_ablation_test.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
