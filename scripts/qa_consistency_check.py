#!/usr/bin/env python3
"""Check question-answer consistency for CWQ test data using batch LLM calls."""
import asyncio, aiohttp, json, pickle, os

TEST_DATA = "/zhaoshu/subgraph/data/cwq_processed/test_literal_and_language_fixed.pkl"
OUTPUT_DIR = "/zhaoshu/subgraph/reports/question_answer_consistency"
LLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "qwen35-9b-local"
BATCH_SIZE = 20  # cases per LLM call

def build_batch_prompt(cases):
    lines = [
        "Evaluate whether each question clearly asks for an answer of the given type.",
        "Consider: missing type words, ambiguous referents, answer type mismatches.",
        "",
        "Output ONLY a JSON array, one object per case, in order:",
        '[{"consistent": true/false, "issue": "description or null", "fix": "corrected question or null"}]',
        "",
    ]
    for i, c in enumerate(cases):
        q = c["question"]
        a = c.get("a_entity", []) or c.get("answers", [])
        lines.append(f"{i+1}. Question: {q}")
        lines.append(f"   Answer(s): {json.dumps(a[:5])}")
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
                }, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    text = await resp.text()
                    data = json.loads(text)
                    if "error" in data:
                        return [{"id": c["id"], "question": c["question"], "consistent": True, "issue": f"API: {data['error']}"} for c in cases]
                    content = data["choices"][0]["message"]["content"].strip()
                    # Extract JSON array
                    start = content.find("[")
                    end = content.rfind("]") + 1
                    if start < 0 or end <= 0:
                        return [{"id": c["id"], "question": c["question"], "consistent": True, "issue": "parse error"} for c in cases]
                    results = json.loads(content[start:end])
                    # Pad or trim to match cases
                    while len(results) < len(cases):
                        results.append({"consistent": True, "issue": "missing"})
                    results = results[:len(cases)]
                    out = []
                    for c, r in zip(cases, results):
                        a = c.get("a_entity", []) or c.get("answers", [])
                        out.append({
                            "id": c["id"],
                            "question": c["question"],
                            "answers": a[:5],
                            "consistent": r.get("consistent", True),
                            "issue": r.get("issue"),
                            "fix": r.get("fix"),
                        })
                    return out
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
            else:
                return [{"id": c["id"], "question": c["question"], "consistent": True, "issue": f"Err: {str(e)[:80]}"} for c in cases]
    return [{"id": c["id"], "question": c["question"], "consistent": True, "issue": "Max retries"} for c in cases]

async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(TEST_DATA, "rb") as f:
        data = pickle.load(f)

    # Filter cases with answers
    valid = [c for c in data if (c.get("a_entity") or c.get("answers"))]
    print(f"Total: {len(data)}, With answers: {len(valid)}")

    sem = asyncio.Semaphore(3)
    results = []

    async with aiohttp.ClientSession() as session:
        batches = [valid[i:i+BATCH_SIZE] for i in range(0, len(valid), BATCH_SIZE)]
        # Submit all at once — vLLM handles batching internally
        all_results = await asyncio.gather(*[check_batch(session, sem, b) for b in batches])
        for br in all_results:
            results.extend(br)

    flagged = [r for r in results if not r.get("consistent", True)]
    print(f"\nTotal: {len(valid)} | Flagged: {len(flagged)}")

    # Categorize
    categories = {}
    for r in flagged:
        issue = (r.get("issue") or "unknown").lower()
        if any(w in issue for w in ["missing", "specif", "type word"]):
            cat = "missing_type"
        elif any(w in issue for w in ["ambig", "unclear", "vague"]):
            cat = "ambiguous"
        elif any(w in issue for w in ["mismatch", "wrong", "ask", "expects"]):
            cat = "type_mismatch"
        else:
            cat = "other"
        categories[cat] = categories.get(cat, 0) + 1

    with open(f"{OUTPUT_DIR}/flagged_cases.json", "w") as f:
        json.dump(flagged, f, ensure_ascii=False, indent=2)
    with open(f"{OUTPUT_DIR}/summary.json", "w") as f:
        json.dump({"total": len(valid), "flagged": len(flagged), "categories": categories}, f, indent=2)

    print(f"Categories: {categories}")
    print(f"Results saved to {OUTPUT_DIR}/")

    # Show some examples
    print(f"\n=== Sample flagged cases ===")
    for r in flagged[:10]:
        print(f"  Q: {r['question'][:70]}")
        print(f"  A: {r['answers'][:3]}")
        print(f"  Issue: {r.get('issue')}")
        print(f"  Fix: {r.get('fix')}")
        print()

asyncio.run(main())
