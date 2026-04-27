#!/usr/bin/env python3
"""Compare Arm A vs Arm B decision consistency results.
Outputs ranked improvements/regressions and a full case-level CSV.
"""
import json, csv, sys
from pathlib import Path

arm_a_path = Path("reports/skill_enhanced_test/prompt_tuning_v1/decision_consistency_full_armA/results.json")
arm_b_path = Path("reports/skill_enhanced_test/prompt_tuning_v1/decision_consistency_full_armB/results.json")
out_dir = Path("reports/skill_enhanced_test/prompt_tuning_v1")
out_dir.mkdir(parents=True, exist_ok=True)

with open(arm_a_path) as f:
    arm_a = {c["case_id"]: c for c in json.load(f)}
with open(arm_b_path) as f:
    arm_b = {c["case_id"]: c for c in json.load(f)}

rows = []
for cid in sorted(arm_a.keys()):
    a = arm_a[cid]
    b = arm_b.get(cid)
    if not b:
        continue
    a_f1 = a["f1"]
    b_f1 = b["f1"]
    delta = round(b_f1 - a_f1, 4)
    b_dis = b.get("consistency", {}).get("disagreement_turns", 0)
    b_used = b.get("consistency", {}).get("used_turns", 0)
    a_fe = a.get("frontend_errors", 0)
    b_fe = b.get("frontend_errors", 0)
    rows.append({
        "case_id": cid,
        "question": a["question"],
        "a_f1": a_f1,
        "b_f1": b_f1,
        "delta": delta,
        "b_disagreement_turns": b_dis,
        "b_used_turns": b_used,
        "a_frontend_errors": a_fe,
        "b_frontend_errors": b_fe,
        "a_turns": a["turns"],
        "b_turns": b["turns"],
        "a_predicted": json.dumps(a.get("predicted", [])),
        "b_predicted": json.dumps(b.get("predicted", [])),
        "ground_truth": json.dumps(a.get("ground_truth", [])),
    })

improved = sorted([r for r in rows if r["delta"] > 0], key=lambda x: -x["delta"])
regressed = sorted([r for r in rows if r["delta"] < 0], key=lambda x: x["delta"])
unchanged = [r for r in rows if r["delta"] == 0]

print(f"Total: {len(rows)}, Improved: {len(improved)}, Regressed: {len(regressed)}, Unchanged: {len(unchanged)}")

# Write full CSV
csv_path = out_dir / "full_consistency_case_comparison.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f"CSV: {csv_path}")

# Write top 20 improvements and regressions as JSON for downstream use
top_imp = improved[:20]
top_reg = regressed[:20]
summary_path = out_dir / "top_improvements_regressions.json"
with open(summary_path, "w") as f:
    json.dump({"top_20_improvements": top_imp, "top_20_regressions": top_reg}, f, indent=2)
print(f"Summary: {summary_path}")

# Print top 20 improvements
print("\n=== TOP 20 IMPROVEMENTS ===")
print(f"{'Case':<18} {'A F1':>6} {'B F1':>6} {'Delta':>7} {'B Dis':>5} {'B Used':>6}")
for r in top_imp:
    print(f"{r['case_id']:<18} {r['a_f1']:>6.2f} {r['b_f1']:>6.2f} {r['delta']:>+7.2f} {r['b_disagreement_turns']:>5} {r['b_used_turns']:>6}")

# Print top 20 regressions
print("\n=== TOP 20 REGRESSIONS ===")
print(f"{'Case':<18} {'A F1':>6} {'B F1':>6} {'Delta':>7} {'B Dis':>5} {'B Used':>6}")
for r in top_reg:
    print(f"{r['case_id']:<18} {r['a_f1']:>6.2f} {r['b_f1']:>6.2f} {r['delta']:>+7.2f} {r['b_disagreement_turns']:>5} {r['b_used_turns']:>6}")

# Stats on delta distribution
deltas = [r["delta"] for r in rows]
pos_deltas = [d for d in deltas if d > 0]
neg_deltas = [d for d in deltas if d < 0]
print(f"\nDelta stats: mean={sum(deltas)/len(deltas):.4f}")
print(f"Positive deltas: {len(pos_deltas)} (mean +{sum(pos_deltas)/len(pos_deltas):.4f})" if pos_deltas else "")
print(f"Negative deltas: {len(neg_deltas)} (mean {sum(neg_deltas)/len(neg_deltas):.4f})" if neg_deltas else "")

# Bucket by delta magnitude
tiny = sum(1 for d in deltas if abs(d) > 0 and abs(d) <= 0.1)
small = sum(1 for d in deltas if 0.1 < abs(d) <= 0.3)
medium = sum(1 for d in deltas if 0.3 < abs(d) <= 0.6)
large = sum(1 for d in deltas if abs(d) > 0.6)
print(f"\nDelta magnitudes (non-zero): <=0.1: {tiny}, 0.1-0.3: {small}, 0.3-0.6: {medium}, >0.6: {large}")

# Frontend error comparison
a_fe_total = sum(r["a_frontend_errors"] for r in rows)
b_fe_total = sum(r["b_frontend_errors"] for r in rows)
fe_fixed = sum(1 for r in rows if r["a_frontend_errors"] > 0 and r["b_frontend_errors"] == 0)
fe_created = sum(1 for r in rows if r["a_frontend_errors"] == 0 and r["b_frontend_errors"] > 0)
print(f"\nFrontend errors: A={a_fe_total}, B={b_fe_total}, fixed={fe_fixed}, created={fe_created}")
