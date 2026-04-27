#!/usr/bin/env python3
"""Test Agent 3 (Action Selection) with reward scoring.

Uses existing test results to get real action spaces, presents them to Agent 3,
and evaluates selections using action reward (F1 × coverage).
"""

import json
import os
import re
import sys

sys.path.insert(0, '/zhaoshu/subgraph')
from config.subagent_prompts import AGENT3_SYSTEM as AGENT3_SYSTEM_COT

import requests

LLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = os.getenv("TEST_MODEL", "qwen35-9b-local")
RESULTS_PATH = "/zhaoshu/subgraph/reports/skill_v3_test/v3_full_1494/results.json"


# --- Reward functions ---

def normalize(s):
    if not isinstance(s, str):
        s = str(s)
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch.isalnum() or ch in " .-_")
    return " ".join(s.split())


def extract_action_spaces(plan_text):
    """Parse action spaces from plan response text."""
    actions = {}
    # Match patterns like:
    # ► From Relation: "rel" (Anchor: X)
    #   1. Action: action(anchor="X", path=[...])
    #      Logic Pattern: ...
    #      Analogical Example: ...

    blocks = re.split(r'► From Relation:', plan_text)
    for i, block in enumerate(blocks[1:], 1):  # skip first split (before any ►)
        aid = f"A{i}"

        # Extract relation name
        rel_match = re.match(r'\s*"([^"]+)"', block)
        relation = rel_match.group(1) if rel_match else "unknown"

        # Extract anchor
        anchor_match = re.search(r'Anchor:\s*([^\n)]+)', block)
        anchor = anchor_match.group(1).strip() if anchor_match else "unknown"

        # Extract path
        path_matches = re.findall(
            r'\{"relation":\s*"([^"]+)",\s*"direction":\s*"([^"]+)"\}', block)
        path_str = " -> ".join(f"{r}({d})" for r, d in path_matches) if path_matches else relation

        # Extract logic pattern
        logic_match = re.search(r'Logic Pattern:\s*\n((?:\s+Step.*\n?)+)', block)
        logic = logic_match.group(0).strip() if logic_match else ""

        # Extract analogical example
        example_match = re.search(r'Analogical Example:\s*\n\s*(.+)', block)
        example = example_match.group(1).strip() if example_match else ""

        actions[aid] = {
            "relation": relation,
            "anchor": anchor,
            "path": path_str,
            "logic": logic,
            "example": example,
            "raw_block": block.strip(),
        }

    return actions


def extract_entities_from_action(response_text):
    """Extract entities from action backend response."""
    entities = []
    for pat in [
        r'CVT-Expanded Entities\s*\(\d+\):\s*\n\s*\[([^\]]+)\]',
        r'Leaf Entities\s*\(\d+[^)]*\):\s*\n\s*\[([^\]]+)\]',
        r'Target Entities\s*:?\s*\n\s*\[([^\]]+)\]',
    ]:
        match = re.search(pat, response_text)
        if match:
            list_str = match.group(1)
            for item in re.findall(r"'([^']*)'|\"([^\"]*)\"", list_str):
                val = item[0] or item[1]
                if val and val.strip():
                    entities.append(val.strip())
            if not entities:
                for part in list_str.split(','):
                    cleaned = part.strip().strip("'\"")
                    if cleaned:
                        entities.append(cleaned)
            break
    return entities


def compute_action_relevance(action_entities, ground_truth):
    """Compute relevance score for an action against ground truth.

    Returns precision, recall, F1, and matched entities.
    """
    pred_norm = {normalize(e) for e in action_entities}
    gt_norm = {normalize(g) for g in ground_truth}

    if not pred_norm or not gt_norm:
        return {"precision": 0, "recall": 0, "f1": 0, "matched": []}

    common = pred_norm & gt_norm
    precision = len(common) / len(pred_norm) if pred_norm else 0
    recall = len(common) / len(gt_norm) if gt_norm else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched": [e for e in action_entities if normalize(e) in gt_norm],
    }


def format_action_spaces(actions):
    """Format action spaces for Agent 3 input."""
    lines = []
    for aid, info in actions.items():
        lines.append(f"► Action {aid}: From Relation: \"{info['relation']}\" (Anchor: {info['anchor']})")
        lines.append(f"  Path: {info['path']}")
        if info['logic']:
            for ll in info['logic'].split('\n'):
                if ll.strip():
                    lines.append(f"  {ll.strip()}")
        if info['example']:
            lines.append(f"  Analogical Example: {info['example']}")
        lines.append("")
    return "\n".join(lines)


def call_llm(system_prompt, messages, temperature=0.3, max_tokens=1024):
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = requests.post(LLM_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def extract_selected_actions(response_text):
    """Extract selected action IDs from Agent 3 response."""
    for pat in [r'select_action\(action_id="([^"]+)"\)', r"select_action\(action_id='([^']+)'\)"]:
        matches = re.findall(pat, response_text)
        if matches:
            return matches
    return []


def main():
    # Load existing results
    with open(RESULTS_PATH) as f:
        all_results = json.load(f)

    # Target cases
    target_ids = ['WebQTest-832', 'WebQTrn-557', 'WebQTest-743', 'WebQTrn-1841',
                  'WebQTest-576', 'WebQTrn-3376', 'WebQTest-1785', 'WebQTrn-493']

    # Build case data from existing results
    cases = {}
    for case in all_results:
        cid = case.get('case_id', '').split('_')[0]
        if cid not in target_ids:
            continue

        gt = case.get('ground_truth', [])
        question = case.get('question', '')

        # Extract plan response and action results
        plan_text = None
        action_results = {}  # action_id -> {entities, relevance}

        for turn in case.get('trajectory', []):
            br = turn.get('backend_results', [])
            eq = turn.get('executed_queries', [])

            # Get plan response
            for r in br:
                if r.get('tool_name') == 'plan' and not plan_text:
                    plan_text = r.get('response_text', '')

        if not plan_text:
            continue

        # Parse action spaces from plan
        action_spaces = extract_action_spaces(plan_text)
        if not action_spaces:
            continue

        # Get executed action results and compute relevance
        for turn in case.get('trajectory', []):
            eq = turn.get('executed_queries', [])
            br = turn.get('backend_results', [])
            for q, r in zip(eq, br):
                if q.get('tool_name') == 'action':
                    args = q.get('arguments', {})
                    path = args.get('path', [])
                    path_sig = " -> ".join(
                        f"{p.get('relation', '?')}({p.get('direction', '?')})" for p in path
                    )
                    entities = extract_entities_from_action(r.get('response_text', ''))
                    relevance = compute_action_relevance(entities, gt)
                    # Match to action space by path
                    for aid, info in action_spaces.items():
                        if info['path'] == path_sig:
                            action_spaces[aid]['entities'] = entities
                            action_spaces[aid]['relevance'] = relevance
                            break

        cases[cid] = {
            'question': question,
            'ground_truth': gt,
            'plan_text': plan_text,
            'action_spaces': action_spaces,
        }

    # --- Run test ---
    print("=" * 80)
    print("Agent 3 (Action Selection) Test with Reward Scoring")
    print(f"Model: {MODEL} | Cases: {len(cases)}")
    print("=" * 80)

    results = {}
    for cid, cdata in sorted(cases.items()):
        question = cdata['question']
        gt = cdata['ground_truth']
        actions = cdata['action_spaces']

        print(f"\n{'═' * 70}")
        print(f"[{cid}] {question}")
        print(f"GT: {gt[:5]}{'...' if len(gt)>5 else ''}")
        print(f"Action Spaces: {len(actions)}")

        # Print action space relevance
        for aid, info in actions.items():
            rel = info.get('relevance', {})
            matched = rel.get('matched', [])
            f1 = rel.get('f1', 0)
            n_ent = len(info.get('entities', []))
            marker = " ★" if f1 > 0 else ""
            print(f"  {aid}: {info['relation'][:50]} | {n_ent} entities | F1={f1:.2f}{marker}")
            if matched:
                print(f"     Matched: {matched[:3]}{'...' if len(matched)>3 else ''}")

        # Format for Agent 3
        action_space_text = format_action_spaces(actions)
        user_msg = f"Question: {question}\n\n[STAGE: SELECT ACTION SPACE]\n\n{cdata['plan_text']}\n\nSelect up to 2 actions."

        try:
            # Call Agent 3
            s3 = call_llm(AGENT3_SYSTEM_COT, [{"role": "user", "content": user_msg}])
            selected = extract_selected_actions(s3)

            print(f"\n[Agent 3 Response]")
            for line in s3.split('\n'):
                if line.strip():
                    print(f"  {line.strip()}")

            # Compute selection reward
            selected_f1 = []
            selected_coverage = []
            for aid in selected:
                if aid in actions:
                    rel = actions[aid].get('relevance', {})
                    selected_f1.append(rel.get('f1', 0))
                    coverage = len(rel.get('matched', [])) / len(gt) if gt else 0
                    selected_coverage.append(coverage)

            # Overall reward: max F1 across selected × avg coverage
            best_f1 = max(selected_f1) if selected_f1 else 0
            avg_coverage = sum(selected_coverage) / len(selected_coverage) if selected_coverage else 0
            reward = best_f1 * avg_coverage if avg_coverage > 0 else best_f1

            # Check if best action was selected
            best_aid = max(actions.keys(),
                          key=lambda a: actions[a].get('relevance', {}).get('f1', 0))
            hit = best_aid in selected

            print(f"\n[EVALUATION]")
            print(f"  Selected: {selected}")
            print(f"  Best action: {best_aid} (F1={actions[best_aid].get('relevance',{}).get('f1',0):.2f})")
            print(f"  Hit best: {'YES' if hit else 'NO'}")
            print(f"  Reward: {reward:.3f} (best_F1={best_f1:.2f} × coverage={avg_coverage:.2f})")

            results[cid] = {
                "question": question,
                "ground_truth": gt,
                "selected": selected,
                "best_action": best_aid,
                "hit_best": hit,
                "reward": reward,
                "response": s3,
            }

        except Exception as e:
            print(f"ERROR: {e}")
            results[cid] = {"question": question, "error": str(e)}

    # --- Summary ---
    print(f"\n\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    total = len(results)
    hits = sum(1 for r in results.values() if r.get('hit_best'))
    rewards = [r.get('reward', 0) for r in results.values() if 'reward' in r]

    print(f"Cases: {total}")
    print(f"Hit best action: {hits}/{total} ({hits/total*100:.0f}%)")
    if rewards:
        print(f"Avg reward: {sum(rewards)/len(rewards):.3f}")
        print(f"Max reward: {max(rewards):.3f}")
        print(f"Min reward: {min(rewards):.3f}")

    # Save
    out = "/zhaoshu/subgraph/scripts/agent3_test.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
