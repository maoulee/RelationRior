#!/usr/bin/env python3
"""Inspect small-delta cases to distinguish noise from mechanism effects."""
import json, csv
from pathlib import Path

csv_path = Path("reports/skill_enhanced_test/prompt_tuning_v1/full_consistency_case_comparison.csv")
arm_a_path = Path("reports/skill_enhanced_test/prompt_tuning_v1/decision_consistency_full_armA/results.json")
arm_b_path = Path("reports/skill_enhanced_test/prompt_tuning_v1/decision_consistency_full_armB/results.json")

with open(arm_a_path) as f:
    arm_a = {c["case_id"]: c for c in json.load(f)}
with open(arm_b_path) as f:
    arm_b = {c["case_id"]: c for c in json.load(f)}

rows = []
with open(csv_path) as f:
    for r in csv.DictReader(f):
        rows.append(r)

# Small positive deltas (0 < delta <= 0.1)
small_pos = [r for r in rows if 0 < float(r["delta"]) <= 0.1]
# Small negative deltas (-0.1 <= delta < 0)
small_neg = [r for r in rows if -0.1 <= float(r["delta"]) < 0]
# Medium positive (0.1 < delta <= 0.5)
med_pos = [r for r in rows if 0.1 < float(r["delta"]) <= 0.5]
# Medium negative (-0.5 <= delta < -0.1)
med_neg = [r for r in rows if -0.5 <= float(r["delta"]) < -0.1]

print(f"Small pos (0<d<=0.1): {len(small_pos)}")
print(f"Small neg (-0.1<=d<0): {len(small_neg)}")
print(f"Med pos (0.1<d<=0.5): {len(med_pos)}")
print(f"Med neg (-0.5<=d<-0.1): {len(med_neg)}")

def inspect_small(case_id, label=""):
    a = arm_a.get(case_id)
    b = arm_b.get(case_id)
    if not a or not b:
        return
    print(f"\n  {label}{case_id}: Q={a['question'][:60]}")
    print(f"    A: F1={a['f1']:.2f} pred={a.get('predicted',[])} | B: F1={b['f1']:.2f} pred={b.get('predicted',[])}")
    print(f"    GT={a.get('ground_truth',[])}")
    print(f"    A turns={a['turns']} fe={a.get('frontend_errors',0)} | B turns={b['turns']} fe={b.get('frontend_errors',0)}")
    # Compare tool call counts
    a_traj = a.get("trajectory", [])
    b_traj = b.get("trajectory", [])
    a_tool_counts = [len(s.get("parsed_output",{}).get("queries",[])) for s in a_traj]
    b_tool_counts = [len(s.get("parsed_output",{}).get("queries",[])) for s in b_traj]
    print(f"    A tool_counts/turn: {a_tool_counts}")
    print(f"    B tool_counts/turn: {b_tool_counts}")
    # Check consistency disagreements
    cons = b.get("consistency", {})
    dis = cons.get("disagreement_turns", 0)
    used = cons.get("used_turns", 0)
    print(f"    B: dis={dis} used={used}")
    # Check if tool sequence differs
    a_tool_names = []
    for s in a_traj:
        for q in s.get("parsed_output",{}).get("queries",[]):
            if isinstance(q, dict):
                a_tool_names.extend(q.keys())
    b_tool_names = []
    for s in b_traj:
        for q in s.get("parsed_output",{}).get("queries",[]):
            if isinstance(q, dict):
                b_tool_names.extend(q.keys())
    same_tools = a_tool_names == b_tool_names
    print(f"    Same tool sequence: {same_tools}")
    if not same_tools:
        print(f"    A tools: {a_tool_names}")
        print(f"    B tools: {b_tool_names}")

print("\n=== SMALL POSITIVE DELTAS (likely noise candidates) ===")
for r in small_pos[:5]:
    inspect_small(r["case_id"], "[S+] ")

print("\n=== SMALL NEGATIVE DELTAS (likely noise candidates) ===")
for r in small_neg[:5]:
    inspect_small(r["case_id"], "[S-] ")

print("\n=== MEDIUM POSITIVE DELTAS (real mechanism candidates) ===")
for r in med_pos[:5]:
    inspect_small(r["case_id"], "[M+] ")

print("\n=== MEDIUM NEGATIVE DELTAS (real mechanism candidates) ===")
for r in med_neg[:5]:
    inspect_small(r["case_id"], "[M-] ")

# Also look at frontend-error-specific cases
print("\n=== FRONTEND ERROR TRANSITIONS ===")
fe_fixed_cases = [r for r in rows if int(r["a_frontend_errors"]) > 0 and int(r["b_frontend_errors"]) == 0]
fe_created_cases = [r for r in rows if int(r["a_frontend_errors"]) == 0 and int(r["b_frontend_errors"]) > 0]
print(f"FE fixed (A>0, B=0): {len(fe_fixed_cases)}")
for r in fe_fixed_cases[:5]:
    inspect_small(r["case_id"], "[FE-fixed] ")
print(f"\nFE created (A=0, B>0): {len(fe_created_cases)}")
for r in fe_created_cases[:5]:
    inspect_small(r["case_id"], "[FE-created] ")

# Analyze: how many of the improved/regressed cases also had FE changes?
print("\n=== F1 DELTA vs FE CORRELATION ===")
improved_with_fe_fix = sum(1 for r in rows if float(r["delta"]) > 0 and int(r["a_frontend_errors"]) > 0 and int(r["b_frontend_errors"]) == 0)
improved_with_fe_create = sum(1 for r in rows if float(r["delta"]) > 0 and int(r["a_frontend_errors"]) == 0 and int(r["b_frontend_errors"]) > 0)
regressed_with_fe_fix = sum(1 for r in rows if float(r["delta"]) < 0 and int(r["a_frontend_errors"]) > 0 and int(r["b_frontend_errors"]) == 0)
regressed_with_fe_create = sum(1 for r in rows if float(r["delta"]) < 0 and int(r["a_frontend_errors"]) == 0 and int(r["b_frontend_errors"]) > 0)
print(f"Improved + FE fixed: {improved_with_fe_fix}")
print(f"Improved + FE created: {improved_with_fe_create}")
print(f"Regressed + FE fixed: {regressed_with_fe_fix}")
print(f"Regressed + FE created: {regressed_with_fe_create}")
