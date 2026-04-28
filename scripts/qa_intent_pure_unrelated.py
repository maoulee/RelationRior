#!/usr/bin/env python3
"""Filter truly intent-unrelated cases from the 498 'unrelated' results.

Many 'unrelated' cases are actually factual errors (wrong person, wrong country)
where the answer TYPE matches but the specific entity is wrong. We want to find
cases where the answer type itself is completely different from what the question asks.

E.g., truly unrelated: Q asks "what sport" → A is "Spanish language"
      NOT truly unrelated: Q asks "what country in ASEAN" → A is "India" (wrong country, but still a country)
"""
import asyncio, aiohttp, json, os

INPUT_FILE = "/zhaoshu/subgraph/reports/question_answer_consistency/intent_relevance_results.json"
OUTPUT_DIR = "/zhaoshu/subgraph/reports/question_answer_consistency"
LLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "qwen35-9b-local"
BATCH_SIZE = 15

def build_batch_prompt(cases):
    lines = [
        "Determine if the answer is the WRONG TYPE entirely (not just wrong fact).",
        "",
        "Rules:",
        "- 'wrong_type' = Answer type is fundamentally different from what question asks.",
        "  Examples: Q asks for a sport, A gives a language. Q asks for a person, A gives a date. Q asks for a country, A gives a sport.",
        "- 'wrong_fact' = Answer is the correct TYPE but wrong specific entity.",
        "  Examples: Q asks for a country in ASEAN, A gives India. Q asks for US president, A gives wrong president.",
        "- 'question_bad' = Question itself is garbled, has typos, or is grammatically broken beyond comprehension.",
        "",
        "Output ONLY a JSON array, one object per case, in order:",
        '[{"label": "wrong_type"/"wrong_fact"/"question_bad", "q_type": "what the question asks for", "a_type": "what the answer is", "explain": "one sentence"}]',
        "",
    ]
    for i, c in enumerate(cases):
        q = c["question"]
        a = c.get("answers", []) or []
        lines.append(f"{i+1}. Q: {q}")
        lines.append(f"   A: {json.dumps(a[:5])}")
        lines.append("")
    return "\n".join(lines)

async def check_batch(session, sem, cases):
    prompt = build_batch_prompt(cases)
    for attempt in range(3):
        try:
            async with sem:
                async with session.post(LLM_URL, json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 4000,
                    "chat_template_kwargs": {"enable_thinking": False},
                }, timeout=aiohttp.ClientTimeout(total=90)) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    text = await resp.text()
                    data = json.loads(text)
                    if "error" in data:
                        return [{"id": c["id"], "question": c["question"], "answers": c.get("answers", []),
                                 "label": "error", "q_type": "", "a_type": "", "explain": str(data["error"])[:80]} for c in cases]
                    content = data["choices"][0]["message"]["content"].strip()
                    start = content.find("[")
                    end = content.rfind("]") + 1
                    if start < 0 or end <= 0:
                        return [{"id": c["id"], "question": c["question"], "answers": c.get("answers", []),
                                 "label": "parse_error", "q_type": "", "a_type": "", "explain": "no json"} for c in cases]
                    results = json.loads(content[start:end])
                    while len(results) < len(cases):
                        results.append({"label": "wrong_fact", "q_type": "", "a_type": "", "explain": "missing"})
                    results = results[:len(cases)]
                    out = []
                    for c, r in zip(cases, results):
                        out.append({
                            "id": c["id"],
                            "question": c["question"],
                            "answers": c.get("answers", []) or [],
                            "label": r.get("label", "wrong_fact"),
                            "q_type": r.get("q_type", ""),
                            "a_type": r.get("a_type", ""),
                            "explain": r.get("explain", ""),
                        })
                    return out
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
            else:
                return [{"id": c["id"], "question": c["question"], "answers": c.get("answers", []),
                         "label": "error", "q_type": "", "a_type": "", "explain": str(e)[:80]} for c in cases]
    return [{"id": c["id"], "question": c["question"], "answers": c.get("answers", []),
             "label": "error", "q_type": "", "a_type": "", "explain": "max retries"} for c in cases]

async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(INPUT_FILE) as f:
        all_results = json.load(f)

    # Filter to only 'unrelated' category
    unrelated = [c for c in all_results if c.get("category") == "unrelated"]
    print(f"Re-evaluating {len(unrelated)} 'unrelated' cases for true type mismatch...")

    sem = asyncio.Semaphore(3)
    async with aiohttp.ClientSession() as session:
        batches = [unrelated[i:i+BATCH_SIZE] for i in range(0, len(unrelated), BATCH_SIZE)]
        all_out = await asyncio.gather(*[check_batch(session, sem, b) for b in batches])
        results = []
        for br in all_out:
            results.extend(br)

    cats = {}
    for r in results:
        cats[r["label"]] = cats.get(r["label"], 0) + 1

    wrong_type = [r for r in results if r["label"] == "wrong_type"]
    wrong_fact = [r for r in results if r["label"] == "wrong_fact"]
    question_bad = [r for r in results if r["label"] == "question_bad"]

    # Save
    with open(f"{OUTPUT_DIR}/true_wrong_type_cases.json", "w") as f:
        json.dump(wrong_type, f, ensure_ascii=False, indent=2)
    with open(f"{OUTPUT_DIR}/wrong_fact_cases.json", "w") as f:
        json.dump(wrong_fact, f, ensure_ascii=False, indent=2)
    with open(f"{OUTPUT_DIR}/intent_pure_summary.json", "w") as f:
        json.dump({"total_unrelated": len(unrelated), "labels": cats,
                    "wrong_type": len(wrong_type), "wrong_fact": len(wrong_fact),
                    "question_bad": len(question_bad)}, f, indent=2)

    print(f"\nLabels: {cats}")
    print(f"\nTrue wrong type (答非所问): {len(wrong_type)}")
    print(f"Wrong fact (correct type, wrong entity): {len(wrong_fact)}")
    print(f"Bad question: {len(question_bad)}")

    print(f"\n=== Sample WRONG TYPE cases (true 答非所问) ===")
    for r in wrong_type[:20]:
        print(f"  Q: {r['question'][:80]}")
        print(f"  A: {r['answers'][:3]}")
        print(f"  Q wants: {r['q_type']} | A gives: {r['a_type']}")
        print(f"  {r['explain']}")
        print()

if __name__ == "__main__":
    asyncio.run(main())
