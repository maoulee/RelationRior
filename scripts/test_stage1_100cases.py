#!/usr/bin/env python3
"""Batch Stage 1 decomposition test on 100 CWQ cases."""

import json
import requests
import time
import sys

LLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "qwen35-9b-local"

STAGE1_SYSTEM = r"""You break down a complex question into simple atomic questions.

Rules:
- Each atomic question should be as simple as possible
- Keep original entity names exactly as they appear
- Separate each atomic question with |
- Mark the main search question with [MAIN]
- Mark all other questions with [FOLLOW-UP]
- There must be exactly one [MAIN] question
- [MAIN] = the primary action or relation to search for
- [FOLLOW-UP] = location filters, attribute conditions, comparisons, or secondary lookups
- When a question has the pattern 'What ENTITY in LOCATION did ACTION', split LOCATION into a [FOLLOW-UP]
- Output ONLY the questions separated by |, nothing else
- Do NOT add explanation, reasoning, or extra text"""

USER_TEMPLATE = "Question: {question}"

FEW_SHOT = [
    {"role": "user", "content": "Question: Which lighthouse near Cape Aurora was built before 1890?"},
    {"role": "assistant", "content": "[MAIN] Which lighthouses are near Cape Aurora? | [FOLLOW-UP] which one was built before 1890?"},
    {"role": "user", "content": "Question: What material is the statue carved by Elena Voss made of?"},
    {"role": "assistant", "content": "[MAIN] Which statue was carved by Elena Voss? | [FOLLOW-UP] what material is it made of?"},
    {"role": "user", "content": "Question: Which spacecraft launched by the Horizon program visited Mars first?"},
    {"role": "assistant", "content": "[MAIN] Which spacecraft were launched by the Horizon program? | [FOLLOW-UP] which ones visited Mars? | [FOLLOW-UP] which one visited first?"},
    {"role": "user", "content": "Question: Who painted the mural inside Crystal Dawn Cathedral?"},
    {"role": "assistant", "content": "[MAIN] Which mural is inside Crystal Dawn Cathedral? | [FOLLOW-UP] who painted it?"},
    {"role": "user", "content": "Question: The mascot Sparky represents the team that won the Galactic Cup in 2187 — who was the captain?"},
    {"role": "assistant", "content": "[MAIN] Which team does mascot Sparky represent? | [FOLLOW-UP] did that team win the Galactic Cup in 2187? | [FOLLOW-UP] who was the captain of that team?"},
    {"role": "user", "content": "Question: Which popular sports team in Arcadia won the 2099 Continental Cup?"},
    {"role": "assistant", "content": "[MAIN] Which sports team in Arcadia won the 2099 Continental Cup? | [FOLLOW-UP] is it considered popular?"},
    # Location split pattern — LAST for recency effect
    {"role": "user", "content": "Question: What city in the Northern Region hosted the 2050 Tech Summit?"},
    {"role": "assistant", "content": "[MAIN] Which city hosted the 2050 Tech Summit? | [FOLLOW-UP] is that city in the Northern Region?"},
    {"role": "user", "content": "Question: What province in the Coastal Zone elected Mara Voss as governor?"},
    {"role": "assistant", "content": "[MAIN] Which province elected Mara Voss as governor? | [FOLLOW-UP] is that province in the Coastal Zone?"},
]


def call_llm(question, temperature=0.3, max_tokens=512):
    messages = list(FEW_SHOT)
    messages.append({"role": "user", "content": USER_TEMPLATE.format(question=question)})
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": STAGE1_SYSTEM}] + messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = requests.post(LLM_URL, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_decomposition(raw):
    """Parse output into structured parts."""
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    main_q = None
    followups = []
    for p in parts:
        if "[MAIN]" in p:
            main_q = p.replace("[MAIN]", "").strip()
        elif "[FOLLOW-UP]" in p:
            followups.append(p.replace("[FOLLOW-UP]", "").strip())
        else:
            # No tag — treat as untagged
            if main_q is None:
                main_q = p.strip()
            else:
                followups.append(p.strip())
    return main_q, followups, len(parts)


def main():
    # Load 100 cases
    src = "/zhaoshu/subgraph/reports/skill_enhanced_test/agg_markdown_v2_fixed_100samples/results.json"
    data = json.load(open(src))
    cases = [(c["case_id"], c["question"]) for c in data]

    print(f"Stage 1 Decomposition — 100 CWQ Cases")
    print(f"Model: {MODEL}")
    print("=" * 80)

    results = {}
    errors = 0
    t0 = time.time()

    for i, (cid, question) in enumerate(cases):
        pct = (i + 1) / len(cases) * 100
        try:
            raw = call_llm(question)
            main_q, followups, n_parts = parse_decomposition(raw)
            results[cid] = {
                "question": question,
                "raw_output": raw,
                "main": main_q,
                "followups": followups,
                "n_parts": n_parts,
                "has_main": main_q is not None,
                "status": "ok",
            }
            tag = "M" if main_q and followups else ("S" if main_q and not followups else "?")
            print(f"[{i+1:3d}/100 {pct:5.1f}%] {tag} {cid}: {raw[:90]}")
        except Exception as e:
            errors += 1
            results[cid] = {"question": question, "raw_output": str(e), "status": "error"}
            print(f"[{i+1:3d}/100 {pct:5.1f}%] ERR {cid}: {e}")

    elapsed = time.time() - t0

    # Statistics
    ok = [r for r in results.values() if r["status"] == "ok"]
    multi = [r for r in ok if r["followups"]]
    single = [r for r in ok if not r["followups"]]
    no_main = [r for r in ok if not r["has_main"]]

    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Total:    {len(cases)}")
    print(f"OK:       {len(ok)}")
    print(f"Errors:   {errors}")
    print(f"Multi-Q:  {len(multi)} ({len(multi)/max(len(ok),1)*100:.1f}%)")
    print(f"Single-Q: {len(single)} ({len(single)/max(len(ok),1)*100:.1f}%)")
    print(f"No [MAIN]:{len(no_main)}")
    print(f"Time:     {elapsed:.1f}s ({elapsed/len(cases):.2f}s/case)")

    # Save
    out_path = "/zhaoshu/subgraph/scripts/stage1_100cases_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
