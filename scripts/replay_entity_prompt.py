#!/usr/bin/env python3
"""Replay failed Stage 8 cases with entity-centric prompt to compare against default.

Uses the same graph evidence from previous runs, only changes the reasoning prompt.
"""
import asyncio, aiohttp, json, re, sys, os, time

RESULTS_FILE = "/zhaoshu/subgraph/reports/stage_pipeline_test/gte_v2_subquestion/results.json"
LLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "qwen35-9b-local"

def extract_graph_evidence(prompt_text):
    """Extract the GRAPH EVIDENCE section from a reasoning prompt."""
    # Try both markers
    for start_marker in ["GRAPH EVIDENCE (from Freebase, snapshot circa 2015):\n", "GRAPH EVIDENCE:\n"]:
        idx = prompt_text.find(start_marker)
        if idx < 0:
            continue
        start = idx + len(start_marker)
        # Find end marker
        for end_marker in ["\n━━━ REASONING TASK", "\n━━━ EVIDENCE-BASED", "\n━━━ ENTITY-CENTRIC"]:
            end = prompt_text.find(end_marker, start)
            if end > 0:
                return prompt_text[start:end].strip()
    # Fallback: try to find between GRAPH EVIDENCE and next section
    idx = prompt_text.find("GRAPH EVIDENCE")
    if idx >= 0:
        # Find the colon and newline
        colon = prompt_text.find(":\n", idx)
        if colon >= 0:
            start = colon + 2
            # Find next major section
            for marker in ["━━━", "STEP 1", "RULES:", "=== "]:
                end = prompt_text.find(marker, start)
                if end > 0:
                    return prompt_text[start:end].strip()
    return ""


def build_entity_prompt(question, pattern_text, candidates, answer_type, rewritten, logical_paths, selected_paths):
    """Build entity-centric 3-step prompt from extracted data."""
    cand_names = list(dict.fromkeys(candidates[:60])) if candidates else []
    cand_list = "\n".join(f"  - {c}" for c in cand_names) if cand_names else "  (see graph evidence above)"

    # Extract relation names from logical paths (string format: "A --[rel]--> B --[rel2]--> C")
    rel_names = set()
    if logical_paths and selected_paths is not None:
        for i in selected_paths:
            if i < len(logical_paths):
                lp = logical_paths[i]
                if isinstance(lp, str):
                    for m in re.finditer(r'\[([^\]]+)\]', lp):
                        rel_names.add(m.group(1))
    rel_list = ", ".join(sorted(rel_names)[:12]) if rel_names else "(see evidence)"

    answer_type_hint = f"\nAnswer type: {answer_type}" if answer_type else ""
    rewritten_hint = f"\nRewritten: {rewritten}" if rewritten and rewritten != question else ""

    prompt = f"""QUESTION: {question}
{answer_type_hint}{rewritten_hint}

GRAPH EVIDENCE:
{pattern_text}

CANDIDATE ENTITIES (traversal endpoints):
{cand_list}

NOTE: Any entity name appearing in GRAPH EVIDENCE (including intermediate nodes, CVT attribute values, and ← also lines) is also a valid answer. Do NOT restrict answers to the list above only.

━━━ ENTITY-CENTRIC REASONING ━━━

STEP 1 — QUESTION UNDERSTANDING
Answer type: what kind of entity the question seeks (person, country, event, year, etc.).
Available relations in evidence: {rel_list}
Basic fact: the core factual requirement that candidate entities must satisfy to be relevant at all.
  Example: "What team won the Super Bowl in 2009?" → basic fact = "team that won the Super Bowl"
  Example: "Who is the governor of Texas?" → basic fact = "person who held the governor position of Texas"
  Example: "What languages are spoken in Brazil?" → basic fact = "language used in Brazil"
Explicit constraints from the question (time, location, superlatives, quantities, etc.): ____
Implicit constraint heuristics (pick one):
  - UNIQUE ROLE ("the governor/president/leader") without time qualifier → MOST RECENT only
  - EVENTS / ACHIEVEMENTS ("wins/championships/movies/albums") → return ALL matching
  - ATTRIBUTES / PROPERTIES ("languages/religions/currency") → return ALL that apply
  - GROUP MEMBERSHIP ("countries in / states in / members of") → return ALL matching members

STEP 2 — ENTITY SCREENING + CONSTRAINT CHECK
2a. From GRAPH EVIDENCE, list ALL entities (from CANDIDATE ENTITIES list AND from evidence tree nodes/attributes) that satisfy the basic fact. For each, cite the supporting pattern/relation.
  Format: entity — supported by [relation/pattern] — one fact from evidence
  If NO entity satisfies the basic fact, output <answer>None</answer> and stop.

2b. For each constraint (explicit + implicit), check which entities from 2a satisfy it, citing specific attributes from evidence.
  Format: Constraint "[description]": entity1 PASS (evidence: ...), entity2 FAIL (evidence: ...)
  If graph evidence does NOT show failure → KEEP the entity.
  No dates in evidence → do NOT filter by time.

STEP 3 — OUTPUT
Collect entities that PASS ALL constraints.
- 1 entity → output it
- Multiple + unique role heuristic → pick MOST RECENT by graph dates; no dates → output ALL
- Multiple + events/attributes/group → output ALL
- Zero entities passed → <answer>None</answer>

RULES:
- Graph evidence ONLY. No outside knowledge.
- ANY entity in GRAPH EVIDENCE is a valid answer — not just CANDIDATE ENTITIES list. This includes intermediate nodes, CVT attribute values, and entities in ← also lines.
- Answer may be at intermediate hop, not only terminal node.
- "When" questions → output event NAME (e.g. "2014 World Series"), NOT raw timestamp.
- Singular/plural phrasing alone does NOT determine answer count.
- If unsure whether an entity satisfies a constraint → KEEP it.
- Over-output is better than discarding valid answers.

━━━ EXAMPLES (illustrate METHOD only) ━━━

Example A — Basic fact + unique role:
Q: "Who is the president of France?"
Basic fact: person who holds the president position of France.
2a: Emmanuel Macron (president_of → France), François Hollande (president_of → France).
2b: Constraint "most recent (unique role)": Macron PASS (term start 2017-05-14), Hollande FAIL (term start 2012-05-15).
3: Macron.

Example B — Events → ALL:
Q: "What movies did the actor who played Forrest Gump star in?"
Basic fact: movies starring the actor who played Forrest Gump.
2a: Forrest Gump (starred_in ← Tom Hanks), Saving Private Ryan (starred_in ← Tom Hanks), Cast Away (starred_in ← Tom Hanks).
2b: No additional constraints beyond basic fact. Events heuristic → ALL.
3: Forrest Gump | Saving Private Ryan | Cast Away.

━━━ OUTPUT FORMAT ━━━
<reasoning>
Step 1: answer type, basic fact, constraints.
Step 2a: entities satisfying basic fact with evidence.
Step 2b: constraint check per entity.
Step 3: passing entities and output decision.
</reasoning>
<answer>\\boxed{{exact entity}}</answer>
Multiple: <answer>\\boxed{{e1}} \\boxed{{e2}}</answer>
No valid entity: <answer>None</answer>
NO text after </answer> tag."""

    return prompt


def normalize(s):
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9%.' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def candidate_hit(predictions, ground_truths):
    if not predictions or not ground_truths:
        return False
    p_set = {normalize(p) for p in predictions}
    g_set = {normalize(g) for g in ground_truths}
    return bool(p_set & g_set)


def compute_f1(predictions, ground_truths):
    if not predictions or not ground_truths:
        return 0.0, 0.0, 0.0
    p_set = {normalize(p) for p in predictions}
    g_set = {normalize(g) for g in ground_truths}
    hits = len(p_set & g_set)
    recall = hits / len(g_set) if g_set else 0
    precision = hits / len(p_set) if p_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1, recall, precision


async def call_llm(session, messages, max_tokens=2400):
    async with session.post(LLM_URL, json={
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=aiohttp.ClientTimeout(total=120)) as resp:
        data = json.loads(await resp.text())
        return data["choices"][0]["message"]["content"].strip()


def extract_answer(raw):
    """Extract answer from <answer>...\boxed{...}...</answer> format."""
    ans_match = re.search(r'<answer>(.*?)</answer>', raw, re.DOTALL)
    if ans_match:
        boxed = re.findall(r'\\boxed\{([^}]+)\}', ans_match.group(1))
        if boxed:
            return " | ".join(b.strip() for b in boxed), True
        text = ans_match.group(1).strip()
        if text.lower() == "none":
            return "None", True
        return text, True
    # Fallback: last \boxed{}
    boxed = re.findall(r'\\boxed\{([^}]+)\}', raw)
    if boxed:
        return " | ".join(b.strip() for b in boxed), False
    return raw[:100], False


async def main():
    results = json.load(open(RESULTS_FILE))

    # Filter: LLM miss cases with reasoning prompt
    miss_cases = [c for c in results if not c.get("llm_hit") and c.get("llm_reasoning_prompt")]

    # Also include partial hit cases (F1 < 1.0) — answer completeness issues
    partial_cases = [c for c in results if c.get("llm_hit") and c.get("llm_f1", 1.0) < 1.0 and c.get("llm_reasoning_prompt")]

    # Also include a few full hit cases to check for regressions
    full_hit_cases = [c for c in results if c.get("llm_hit") and c.get("llm_f1", 1.0) >= 1.0 and c.get("llm_reasoning_prompt")][:5]

    all_cases = miss_cases + partial_cases + full_hit_cases
    print(f"Replaying {len(miss_cases)} miss + {len(partial_cases)} partial + {len(full_hit_cases)} full hit = {len(all_cases)} total")
    print("=" * 80)

    sem = asyncio.Semaphore(3)
    async with aiohttp.ClientSession() as session:
        outcomes = []

        for ci, case in enumerate(all_cases):
            q = case["question"]
            gt = case.get("gt_answers", [])
            old_answer = case.get("llm_answer", "")
            old_hit = case.get("llm_hit", False)

            # Extract graph evidence from the old prompt
            old_prompt = case["llm_reasoning_prompt"]
            pattern_text = extract_graph_evidence(old_prompt)
            if not pattern_text:
                print(f"[{ci+1}] SKIP (no evidence extracted): {q[:60]}")
                continue

            candidates = case.get("answer_candidates", [])
            answer_type = case.get("stage_1a_answer_type", "")
            rewritten = case.get("stage_1a_rewritten", "")
            logical_paths = case.get("logical_paths", [])
            selected_paths = case.get("selected_paths", [])

            new_prompt = build_entity_prompt(q, pattern_text, candidates, answer_type, rewritten, logical_paths, selected_paths)

            system_msg = "You are a precise graph QA system using entity-centric reasoning. First identify the basic fact, then screen entities against it, then check constraints. Any entity in GRAPH EVIDENCE is a valid answer, not just CANDIDATE ENTITIES — including intermediate nodes and CVT attribute values. It is acceptable to output None if no entity satisfies the basic fact. Use graph evidence only."

            try:
                async with sem:
                    raw = await call_llm(session, [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": new_prompt},
                    ])
            except Exception as e:
                print(f"[{ci+1}] ERROR: {e}")
                continue

            new_answer, has_tag = extract_answer(raw)
            new_preds = new_answer.split(" | ")
            new_hit = candidate_hit(new_preds, gt)
            new_f1, new_recall, new_prec = compute_f1(new_preds, gt)
            old_f1 = case.get("llm_f1", 0) or 0
            old_recall = case.get("llm_recall", 0) or 0
            f1_change = new_f1 - old_f1

            label = "MISS→HIT!" if (not old_hit and new_hit) else ("HIT→MISS!" if (old_hit and not new_hit) else ("HIT" if new_hit else "MISS"))
            change = "↑" if (not old_hit and new_hit) else ("↓" if (old_hit and not new_hit) else "=")

            print(f"\n[{ci+1}] {label} {change}")
            print(f"  Q: {q[:80]}")
            print(f"  GT ({len(gt)}): {gt[:3]}")
            print(f"  Old ({'HIT' if old_hit else 'MISS'}): F1={old_f1:.2f} R={old_recall:.2f} | {old_answer[:60]}")
            print(f"  New ({'HIT' if new_hit else 'MISS'}): F1={new_f1:.2f} R={new_recall:.2f} | {new_answer[:80]}")

            outcomes.append({
                "case_id": case.get("case_id"),
                "question": q,
                "gt_answers": gt,
                "old_answer": old_answer,
                "old_hit": old_hit,
                "old_f1": old_f1,
                "old_recall": old_recall,
                "new_answer": new_answer,
                "new_hit": new_hit,
                "new_f1": new_f1,
                "new_recall": new_recall,
                "new_prec": new_prec,
                "f1_change": f1_change,
                "label": label,
            })

    # Summary
    print("\n" + "=" * 80)
    improved = sum(1 for o in outcomes if o["label"] == "MISS→HIT!")
    regressed = sum(1 for o in outcomes if o["label"] == "HIT→MISS")
    same_hit = sum(1 for o in outcomes if o["label"] == "HIT" and o["old_hit"])
    same_miss = sum(1 for o in outcomes if o["label"] == "MISS" and not o["old_hit"])

    # F1 changes
    f1_up = sum(1 for o in outcomes if o["f1_change"] > 0.01)
    f1_down = sum(1 for o in outcomes if o["f1_change"] < -0.01)
    avg_old_f1 = sum(o["old_f1"] for o in outcomes) / len(outcomes) if outcomes else 0
    avg_new_f1 = sum(o["new_f1"] for o in outcomes) / len(outcomes) if outcomes else 0

    print(f"Total replayed: {len(outcomes)}")
    print(f"  MISS→HIT (improved): {improved}")
    print(f"  HIT→MISS (regressed): {regressed}")
    print(f"  HIT→HIT (stable):     {same_hit}")
    print(f"  MISS→MISS (still wrong): {same_miss}")
    print(f"  Net change: {'+' if improved - regressed >= 0 else ''}{improved - regressed}")
    print(f"  F1 improved: {f1_up}, F1 regressed: {f1_down}")
    print(f"  Avg F1: {avg_old_f1:.3f} → {avg_new_f1:.3f} (Δ={avg_new_f1 - avg_old_f1:+.3f})")

    # Save detailed results
    out_dir = os.path.dirname(RESULTS_FILE)
    out_path = os.path.join(out_dir, "entity_prompt_replay.json")
    with open(out_path, "w") as f:
        json.dump(outcomes, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
