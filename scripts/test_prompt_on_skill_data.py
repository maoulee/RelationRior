#!/usr/bin/env python3
"""Evaluate our Agent 3.5→4A prompts on skill-enhanced version's action spaces.

Isolates prompt quality: same cases, same backend data, only prompt differs.
Compares our decomposed reasoning vs the skill-enhanced monolithic pipeline's F1.
"""

import json
import os
import re
import sys

sys.path.insert(0, '/zhaoshu/subgraph')
from config.subagent_prompts import (
    AGENT35_SYSTEM, AGENT35_USER_TEMPLATE,
    AGENT4_SYSTEM_A, AGENT4_USER_TEMPLATE_A,
)

import requests

LLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = os.getenv("TEST_MODEL", "qwen35-9b-local")
SKILL_RESULTS_PATH = sys.argv[1] if len(sys.argv) > 1 else "/zhaoshu/subgraph/reports/skill_enhanced_test/val100_skill_top3/results.json"
OUT_SUFFIX = "_webq" if "val100" in SKILL_RESULTS_PATH else "_cwq"


def load_cwq_questions(path="/zhaoshu/subgraph/data/cwq/cwq_test.jsonl"):
    """Load CWQ questions indexed by case_id."""
    questions = {}
    try:
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                cid = d.get('id', '')
                msgs = d.get('messages', [])
                for m in msgs:
                    if m['role'] == 'user':
                        content = m['content']
                        # Extract question after "Question:\n"
                        qm = re.search(r'Question:\s*\n(.+?)(?:\n\n|\n\[PHASE)', content, re.DOTALL)
                        if qm:
                            questions[cid] = qm.group(1).strip()
                        break
    except FileNotFoundError:
        pass
    return questions

CWQ_QUESTIONS = load_cwq_questions()


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
    answers = re.findall(r'\\boxed\{([^}]+)\}', text)
    return [a.strip() for a in answers if a.strip()]


def extract_candidates_from_action(response_text):
    """Extract entities from action backend response."""
    for pat in [
        r'CVT-Expanded Entities\s*\(\d+[^)]*\):\s*\n\s*\[([^\]]+)\]',
    ]:
        match = re.search(pat, response_text)
        if match:
            entities = _parse_entity_list(match.group(1))
            if entities:
                return entities

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
    """Extract plan, actions, filters from skill-enhanced trajectory."""
    cid = case.get('case_id', '')
    # Use CWQ question if available, else fall back to stored question
    question = CWQ_QUESTIONS.get(cid, case.get('question', ''))
    gt = case.get('ground_truth', [])
    skill_f1 = case.get('f1', 0.0)
    skill_pred = case.get('predicted', [])

    plan_text = ''
    actions = []
    filters = []
    sub_questions = []

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
        'skill_f1': skill_f1,
        'skill_pred': skill_pred,
        'plan_text': plan_text,
        'actions': actions,
        'filters': filters,
    }


def format_action_results_for_agent35(actions):
    lines = []
    for i, act in enumerate(actions):
        aid = f"A{i+1}"
        lines.append(f"► Action {aid}:")
        lines.append(f"  Path: {act['path_str']}")
        lines.append(f"  Entities ({len(act['entities'])}): {', '.join(act['entities'][:20])}")
        if len(act['entities']) > 20:
            lines.append(f"  ... and {len(act['entities'])-20} more")
        resp = act['response_text']
        node_details_match = re.search(r'\[Node Details\]:.*?(?=\n-{5,}|\nCVT-Expanded|\Z)', resp, re.DOTALL)
        if node_details_match:
            for sl in node_details_match.group(0).strip().split('\n')[:20]:
                lines.append(f"  {sl}")
        cvt_match = re.search(r'CVT-Expanded Entities\s*\(\d+[^)]*\):\s*\n\s*\[([^\]]+)\]', resp)
        if cvt_match:
            lines.append(f"  CVT-Expanded Entities: {cvt_match.group(0).split(chr(10))[-1].strip()}")
        lines.append("")
    return "\n".join(lines)


# --- Main ---

def main():
    with open(SKILL_RESULTS_PATH) as f:
        all_results = json.load(f)

    print("=" * 80)
    print("Prompt Evaluation on Skill-Enhanced Action Spaces")
    print(f"Model: {MODEL} | Cases: {len(all_results)}")
    print("=" * 80)

    our_results = {}
    skill_f1s = []

    for idx, case in enumerate(all_results):
        cid = case.get('case_id', f'case_{idx}')
        data = extract_case_data(case)
        question = data['question']
        gt = data['ground_truth']
        skill_f1 = data['skill_f1']

        # Skip cases with no actions
        if not data['actions']:
            print(f"\n[{idx+1}/{len(all_results)}] {cid} | SKIP (no actions)")
            continue

        # Show action F1 summary
        action_f1s = []
        for i, act in enumerate(data['actions']):
            f1 = compute_f1(act['entities'], gt)
            action_f1s.append(f"A{i+1}={f1:.2f}")
        best_action_f1 = max(compute_f1(act['entities'], gt) for act in data['actions'])

        compact = f"[{idx+1}/{len(all_results)}] {cid} | {question[:60]}"
        print(f"\n{compact}")
        print(f"  GT: {gt[:3]}{'...' if len(gt)>3 else ''} | Actions: {len(data['actions'])}")
        print(f"  Action F1s: {', '.join(action_f1s)} (best={best_action_f1:.2f}) | Skill F1: {skill_f1:.2f}")

        try:
            # --- Agent 3.5: Winner Selection ---
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

            # Parse winner
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
                if winner_aid:
                    aidx = int(winner_aid[1:]) - 1
                    candidates = data['actions'][aidx]['entities'] if aidx < len(data['actions']) else []
                else:
                    candidates = []
                    for act in data['actions']:
                        candidates.extend(act['entities'])

            # Build context from winning action
            if winner_aid:
                aidx = int(winner_aid[1:]) - 1
                winner_text = data['actions'][aidx]['response_text'][:2000] if aidx < len(data['actions']) else ""
            else:
                winner_text = ""
                for act in data['actions']:
                    winner_text += act['response_text'][:800] + "\n---\n"
                winner_text = winner_text[:2000]

            # --- Agent 4A: Final Answer ---
            user_4a = AGENT4_USER_TEMPLATE_A.format(
                question=question,
                candidates="\n".join(f"- {c}" for c in candidates),
                kg_data=winner_text,
            )
            resp_4a = call_llm(AGENT4_SYSTEM_A, user_4a, max_tokens=1024)
            pred_a = extract_boxed_answers(resp_4a)
            f1_a = compute_f1(pred_a, gt)

            # Compact output
            sa = "✓" if f1_a >= 0.99 else "△" if f1_a > 0 else "✗"
            ss = "✓" if skill_f1 >= 0.99 else "△" if skill_f1 > 0 else "✗"
            delta = f1_a - skill_f1
            print(f"  3.5: {winner_line[:60]}")
            print(f"  Ours{sa} F1={f1_a:.2f} Pred: {pred_a[:3]}{'...' if len(pred_a)>3 else ''}")
            print(f"  Skill{ss} F1={skill_f1:.2f} Pred: {data['skill_pred'][:3]}{'...' if len(data['skill_pred'])>3 else ''}")
            print(f"  Delta: {delta:+.2f}")

            our_results[cid] = {
                "question": question,
                "ground_truth": gt,
                "candidates_35": candidates,
                "winner_aid": winner_aid,
                "f1_ours": f1_a,
                "pred_ours": pred_a,
                "f1_skill": skill_f1,
                "pred_skill": data['skill_pred'],
                "delta": delta,
                "best_action_f1": best_action_f1,
                "n_actions": len(data['actions']),
                "response_35": resp_35,
                "response_4a": resp_4a,
            }
            skill_f1s.append(skill_f1)

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            our_results[cid] = {"question": question, "error": str(e)}

    # --- Summary ---
    print(f"\n\n{'=' * 80}")
    print("COMPARISON: Our Prompt vs Skill-Enhanced (Same Action Spaces)")
    print(f"{'=' * 80}")

    valid = {k: v for k, v in our_results.items() if 'f1_ours' in v}
    total = len(valid)

    if total > 0:
        f1s_ours = [v['f1_ours'] for v in valid.values()]
        f1s_skill = [v['f1_skill'] for v in valid.values()]
        exact_ours = sum(1 for f in f1s_ours if f >= 0.99)
        exact_skill = sum(1 for f in f1s_skill if f >= 0.99)
        hits_ours = sum(1 for f in f1s_ours if f > 0)
        hits_skill = sum(1 for f in f1s_skill if f > 0)
        avg_ours = sum(f1s_ours) / len(f1s_ours)
        avg_skill = sum(f1s_skill) / len(f1s_skill)

        print(f"\n{'Metric':<30} {'Ours':>8} {'Skill':>8} {'Delta':>8}")
        print("-" * 58)
        print(f"{'Cases':<30} {total:>8} {total:>8}")
        print(f"{'Exact match (F1=1.0)':<30} {exact_ours:>8} {exact_skill:>8} {exact_skill-exact_ours:>+8}")
        print(f"{'Partial match (F1>0)':<30} {hits_ours:>8} {hits_skill:>8} {hits_skill-hits_ours:>+8}")
        print(f"{'Avg F1':<30} {avg_ours:>8.3f} {avg_skill:>8.3f} {avg_skill-avg_ours:>+8.3f}")

        # Per-case comparison
        ours_better = sum(1 for v in valid.values() if v['f1_ours'] > v['f1_skill'])
        skill_better = sum(1 for v in valid.values() if v['f1_skill'] > v['f1_ours'])
        ties = sum(1 for v in valid.values() if v['f1_ours'] == v['f1_skill'])
        print(f"\nPer-case: Ours better={ours_better}, Skill better={skill_better}, Ties={ties}")

        # Cases where skill wins by >0.3
        print(f"\nCases where Skill >> Ours (delta < -0.3):")
        deltas = [(cid, v['delta'], v['f1_ours'], v['f1_skill'], v['question'])
                  for cid, v in valid.items() if v['delta'] < -0.3]
        deltas.sort(key=lambda x: x[1])
        for cid, delta, fo, fs, q in deltas[:15]:
            print(f"  {cid:<20} Ours={fo:.2f} Skill={fs:.2f} delta={delta:+.2f} | {q[:50]}")

        print(f"\nCases where Ours >> Skill (delta > 0.3):")
        deltas_pos = [(cid, v['delta'], v['f1_ours'], v['f1_skill'], v['question'])
                      for cid, v in valid.items() if v['delta'] > 0.3]
        deltas_pos.sort(key=lambda x: -x[1])
        for cid, delta, fo, fs, q in deltas_pos[:15]:
            print(f"  {cid:<20} Ours={fo:.2f} Skill={fs:.2f} delta={delta:+.2f} | {q[:50]}")

        # Full case list sorted by delta
        print(f"\n{'CID':<20} {'Ours':>5} {'Skill':>5} {'Delta':>6}  Question")
        print("-" * 80)
        all_sorted = sorted(valid.items(), key=lambda x: x[1]['delta'])
        for cid, v in all_sorted:
            print(f"{cid:<20} {v['f1_ours']:>5.2f} {v['f1_skill']:>5.2f} {v['delta']:>+6.2f}  {v['question'][:50]}")

    # Save
    out = f"/zhaoshu/subgraph/scripts/prompt_eval{OUT_SUFFIX}.json"
    with open(out, "w") as f:
        json.dump(our_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
