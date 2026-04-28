#!/usr/bin/env python3
"""Re-evaluate flagged cases: distinguish type mismatch (related) vs intent-unrelated (答非所问)."""
import asyncio, aiohttp, json, os

FLAGGED_FILE = "/zhaoshu/subgraph/reports/question_answer_consistency/flagged_cases.json"
OUTPUT_DIR = "/zhaoshu/subgraph/reports/question_answer_consistency"
LLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "qwen35-9b-local"
BATCH_SIZE = 15

def build_batch_prompt(cases):
    lines = [
        "For each case, judge whether the answer is RELATED to the question's intent, even if the type is slightly off.",
        "",
        "Categories:",
        "  'related' - Answer addresses the question's intent, possibly with type granularity difference (e.g., asked for sport, got specific team; asked for person, got person's title; asked for location, got containing region).",
        "  'unrelated' - Answer has NOTHING to do with what the question asks (e.g., asked about geography, answer is about music; asked about a person, answer is a date; asked for a sport, answer is a language).",
        "  'time_event' - Question asks 'when' and answer is an event name instead of a date (e.g., '2014 World Series' for 'when'). This is acceptable — the event implies the time.",
        "  'ambiguous' - Question is too vague/ambiguous to determine relevance.",
        "",
        "Output ONLY a JSON array with one object per case, in order:",
        '[{"category": "related"/"unrelated"/"time_event"/"ambiguous", "reason": "one sentence"}]',
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
                                 "category": "error", "reason": str(data["error"])[:80]} for c in cases]
                    content = data["choices"][0]["message"]["content"].strip()
                    start = content.find("[")
                    end = content.rfind("]") + 1
                    if start < 0 or end <= 0:
                        return [{"id": c["id"], "question": c["question"], "answers": c.get("answers", []),
                                 "category": "parse_error", "reason": "no json"} for c in cases]
                    results = json.loads(content[start:end])
                    while len(results) < len(cases):
                        results.append({"category": "ambiguous", "reason": "missing"})
                    results = results[:len(cases)]
                    out = []
                    for c, r in zip(cases, results):
                        out.append({
                            "id": c["id"],
                            "question": c["question"],
                            "answers": c.get("answers", []) or [],
                            "issue": c.get("issue"),
                            "category": r.get("category", "ambiguous"),
                            "reason": r.get("reason", ""),
                        })
                    return out
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
            else:
                return [{"id": c["id"], "question": c["question"], "answers": c.get("answers", []),
                         "category": "error", "reason": str(e)[:80]} for c in cases]
    return [{"id": c["id"], "question": c["question"], "answers": c.get("answers", []),
             "category": "error", "reason": "max retries"} for c in cases]

async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(FLAGGED_FILE) as f:
        flagged = json.load(f)

    print(f"Total flagged cases to re-evaluate: {len(flagged)}")

    sem = asyncio.Semaphore(3)
    async with aiohttp.ClientSession() as session:
        batches = [flagged[i:i+BATCH_SIZE] for i in range(0, len(flagged), BATCH_SIZE)]
        all_results = await asyncio.gather(*[check_batch(session, sem, b) for b in batches])
        results = []
        for br in all_results:
            results.extend(br)

    # Categorize
    cats = {}
    for r in results:
        cat = r["category"]
        cats[cat] = cats.get(cat, 0) + 1

    unrelated = [r for r in results if r["category"] == "unrelated"]
    time_event = [r for r in results if r["category"] == "time_event"]

    # Save
    with open(f"{OUTPUT_DIR}/intent_relevance_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(f"{OUTPUT_DIR}/unrelated_cases.json", "w") as f:
        json.dump(unrelated, f, ensure_ascii=False, indent=2)
    with open(f"{OUTPUT_DIR}/intent_relevance_summary.json", "w") as f:
        json.dump({"total_flagged": len(flagged), "categories": cats,
                    "unrelated_count": len(unrelated), "time_event_count": len(time_event)}, f, indent=2)

    print(f"\nCategories: {cats}")
    print(f"\nUnrelated (答非所问): {len(unrelated)}")
    print(f"Time/event (acceptable): {len(time_event)}")

    print(f"\n=== Sample unrelated cases ===")
    for r in unrelated[:15]:
        print(f"  Q: {r['question'][:80]}")
        print(f"  A: {r['answers'][:3]}")
        print(f"  Reason: {r.get('reason')}")
        print()

if __name__ == "__main__":
    asyncio.run(main())
