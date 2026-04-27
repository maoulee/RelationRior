#!/usr/bin/env python3
"""Inspect trajectories for specific cases, comparing Arm A vs Arm B.
Outputs compact summaries of tool calls, reasoning, and final answers.
"""
import json, sys
from pathlib import Path

arm_a_path = Path("reports/skill_enhanced_test/prompt_tuning_v1/decision_consistency_full_armA/results.json")
arm_b_path = Path("reports/skill_enhanced_test/prompt_tuning_v1/decision_consistency_full_armB/results.json")

with open(arm_a_path) as f:
    arm_a = {c["case_id"]: c for c in json.load(f)}
with open(arm_b_path) as f:
    arm_b = {c["case_id"]: c for c in json.load(f)}

def extract_tool_calls(trajectory):
    """Extract tool call sequences from trajectory."""
    calls = []
    for step in trajectory:
        turn = step.get("turn", "?")
        raw = step.get("raw_response", "")
        parsed = step.get("parsed_output", {})
        queries = parsed.get("queries", [])
        tool_names = []
        for q in queries:
            if isinstance(q, dict):
                tool_names.extend(q.keys())
            elif isinstance(q, str):
                tool_names.append(q)
        calls.append({
            "turn": turn,
            "tools": tool_names,
            "raw_preview": raw[:300] if raw else "",
        })
    return calls

def extract_consistency_detail(case_data):
    """Extract consistency details from arm B."""
    cons = case_data.get("consistency", {})
    details = []
    for t in cons.get("turns", []):
        if t.get("consistency_used"):
            details.append({
                "turn": t.get("consistency_turn_number"),
                "agreed": t.get("consistency_agreed_initially"),
                "votes": t.get("consistency_votes", {}),
                "chosen": t.get("consistency_chosen_signature", ""),
                "candidates": t.get("consistency_candidate_signatures", []),
            })
    return details

def summarize_case(case_id):
    a = arm_a.get(case_id)
    b = arm_b.get(case_id)
    if not a or not b:
        print(f"  MISSING: {case_id}")
        return

    print(f"\n{'='*80}")
    print(f"CASE: {case_id}")
    print(f"Q: {a['question']}")
    print(f"A F1={a['f1']:.2f} predicted={a.get('predicted',[])} | B F1={b['f1']:.2f} predicted={b.get('predicted',[])}")
    print(f"GT: {a.get('ground_truth',[])}")
    print(f"A turns={a['turns']}, B turns={b['turns']}")
    print(f"A fe={a.get('frontend_errors',0)}, B fe={b.get('frontend_errors',0)}")

    # Tool calls comparison
    a_calls = extract_tool_calls(a.get("trajectory", []))
    b_calls = extract_tool_calls(b.get("trajectory", []))

    print(f"\n  ARM A tool calls ({len(a_calls)} turns):")
    for c in a_calls:
        print(f"    T{c['turn']}: {', '.join(c['tools']) if c['tools'] else '(none)'}")

    print(f"\n  ARM B tool calls ({len(b_calls)} turns):")
    for c in b_calls:
        print(f"    T{c['turn']}: {', '.join(c['tools']) if c['tools'] else '(none)'}")

    # Consistency details (arm B only)
    cons_details = extract_consistency_detail(b)
    if cons_details:
        print(f"\n  ARM B consistency interventions ({len(cons_details)}):")
        for d in cons_details:
            vote_str = "; ".join(f"{k[:60]}={v}" for k, v in d["votes"].items())
            print(f"    T{d['turn']}: agreed={d['agreed']} | votes: {vote_str}")
            if len(d["candidates"]) > 1:
                print(f"      chosen: {d['chosen'][:100]}")

    # Last turn reasoning (final answer)
    a_last = a["trajectory"][-1] if a.get("trajectory") else {}
    b_last = b["trajectory"][-1] if b.get("trajectory") else {}
    print(f"\n  ARM A final reasoning: {(a_last.get('raw_response','') or '')[:200]}")
    print(f"  ARM B final reasoning: {(b_last.get('raw_response','') or '')[:200]}")

# Cases to inspect
top_imp = [
    "WebQTest-1179", "WebQTest-1250", "WebQTest-1313", "WebQTest-146",
    "WebQTest-1479", "WebQTest-1480", "WebQTest-1482", "WebQTest-1537",
    "WebQTest-1586", "WebQTest-16"
]
top_reg = [
    "WebQTest-1047", "WebQTest-1072", "WebQTest-1092", "WebQTest-1102",
    "WebQTest-111", "WebQTest-1119", "WebQTest-1160", "WebQTest-1209",
    "WebQTest-1213", "WebQTest-1248"
]

print("### TOP 10 IMPROVEMENTS ###")
for cid in top_imp:
    summarize_case(cid)

print("\n\n### TOP 10 REGRESSIONS ###")
for cid in top_reg:
    summarize_case(cid)
