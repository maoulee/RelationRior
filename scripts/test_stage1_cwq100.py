#!/usr/bin/env python3
"""Stage 1 decomposition test on 100 randomly sampled CWQ cases."""

import json
import requests
import time
import random

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
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    main_q = None
    followups = []
    for p in parts:
        if "[MAIN]" in p:
            main_q = p.replace("[MAIN]", "").strip()
        elif "[FOLLOW-UP]" in p:
            followups.append(p.replace("[FOLLOW-UP]", "").strip())
        else:
            if main_q is None:
                main_q = p.strip()
            else:
                followups.append(p.strip())
    return main_q, followups, len(parts)


def classify_complexity(question):
    q = question.lower()
    n_subord = sum(1 for k in [' that ', ' who ', ' which ', ' whose ', ' where '] if k in q)
    has_superlative = any(k in q for k in ['latest', 'first', 'last', 'most', 'oldest', 'newest'])
    has_temporal = any(k in q for k in ['before', 'after', 'when', 'during', 'since'])
    if n_subord >= 2:
        return "hard"
    elif n_subord == 1 or has_superlative or has_temporal:
        return "medium"
    else:
        return "easy"


def main():
    # Load CWQ test data
    all_cases = []
    with open("/zhaoshu/subgraph/data/cwq/cwq_test.jsonl") as f:
        for line in f:
            d = json.loads(line)
            case_id = d["id"].split("_")[0]
            user_msg = d["messages"][1]["content"]
            for l in user_msg.split("\n"):
                l = l.strip()
                if '?' in l and not l.startswith('-') and not l.startswith('[') and len(l) > 15:
                    if 'Question:' not in l and not any(kw in l.lower() for kw in ['how to', 'do you', 'can you', 'should you', 'must you']):
                        all_cases.append((case_id, l))
                        break

    # Classify complexity
    by_complexity = {"hard": [], "medium": [], "easy": []}
    for cid, q in all_cases:
        c = classify_complexity(q)
        by_complexity[c].append((cid, q))

    print(f"CWQ total: {len(all_cases)} → hard={len(by_complexity['hard'])}, medium={len(by_complexity['medium'])}, easy={len(by_complexity['easy'])}")

    # Sample: 30 hard + 40 medium + 30 easy = 100
    random.seed(42)
    sample = (
        random.sample(by_complexity["hard"], min(30, len(by_complexity["hard"]))) +
        random.sample(by_complexity["medium"], min(40, len(by_complexity["medium"]))) +
        random.sample(by_complexity["easy"], min(30, len(by_complexity["easy"])))
    )
    random.shuffle(sample)
    print(f"Sampled: {len(sample)} cases")
    print("=" * 80)

    results = {}
    errors = 0
    t0 = time.time()

    for i, (cid, question) in enumerate(sample):
        pct = (i + 1) / len(sample) * 100
        cx = classify_complexity(question)
        try:
            raw = call_llm(question)
            main_q, followups, n_parts = parse_decomposition(raw)
            tag = "M" if followups else "S"
            results[cid] = {
                "question": question,
                "complexity": cx,
                "raw_output": raw,
                "main": main_q,
                "followups": followups,
                "n_parts": n_parts,
                "has_main": main_q is not None,
                "status": "ok",
            }
            print(f"[{i+1:3d}/{len(sample)} {pct:5.1f}%] {tag}/{cx:5s} {cid}: {raw[:100]}")
        except Exception as e:
            errors += 1
            results[cid] = {
                "question": question,
                "complexity": cx,
                "raw_output": str(e),
                "status": "error",
            }
            print(f"[{i+1:3d}/{len(sample)} {pct:5.1f}%] ERR/{cx:5s} {cid}: {e}")

    elapsed = time.time() - t0

    # Statistics by complexity
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    for cx in ["hard", "medium", "easy"]:
        cx_results = [r for r in results.values() if r.get("complexity") == cx and r["status"] == "ok"]
        cx_multi = [r for r in cx_results if r["followups"]]
        cx_single = [r for r in cx_results if not r["followups"]]
        cx_err = [r for r in results.values() if r.get("complexity") == cx and r["status"] == "error"]
        print(f"{cx:6s}: {len(cx_results):3d} ok, {len(cx_err)} err | Multi={len(cx_multi)}, Single={len(cx_single)}")

    ok = [r for r in results.values() if r["status"] == "ok"]
    multi = [r for r in ok if r["followups"]]
    print(f"\nTotal: {len(sample)} | OK={len(ok)} | Errors={errors}")
    print(f"Multi-Q: {len(multi)} ({len(multi)/max(len(ok),1)*100:.1f}%) | Single-Q: {len(ok)-len(multi)}")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(sample):.2f}s/case)")

    # Save
    out_path = "/zhaoshu/subgraph/scripts/stage1_cwq100_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
