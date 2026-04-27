#!/usr/bin/env python3
"""Test Stage 2 Discovery: analyze anchors + domains from Stage 1 decomposition."""

import json
import requests
import time

LLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "qwen35-9b-local"

AVAILABLE_DOMAINS = "award, base, baseball, business, common, freebase, location, organization, sports, time, type"

# Stage 2 system prompt
STAGE2_SYSTEM = r"""You analyze decomposed questions to identify anchor entities and relevant knowledge graph domains.

Input: A set of sub-questions separated by |, where the first is [MAIN] and others are [FOLLOW-UP].

For each sub-question, identify:
1. anchor: the entity name to search in the knowledge graph (keep original spelling)
2. domain: the most relevant domain(s) from the available list
3. intent: what relation/information to look for (brief phrase)

Available domains: award, base, baseball, business, common, freebase, location, organization, sports, time, type

Output format (one line per sub-question):
[MAIN] anchor | domain | intent
[FOLLOW-UP] anchor | domain | intent
...

Rules:
- anchor must be a specific entity name from the question, not a category
- If no specific entity exists, use the question word (who/what/which) as placeholder
- domain must be from the available list, pick the most specific one
- Keep intent as a short phrase describing the relation to explore
- Output ONLY the tagged lines, nothing else"""

# Stage 1 decomposition prompt (same as tested)
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

STAGE1_FEW_SHOT = [
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

# Stage 2 few-shot
STAGE2_FEW_SHOT = [
    {
        "role": "user",
        "content": "[MAIN] Which lighthouses are near Cape Aurora? | [FOLLOW-UP] which one was built before 1890?"
    },
    {
        "role": "assistant",
        "content": "[MAIN] Cape Aurora | location | find nearby lighthouses\n[FOLLOW-UP] lighthouse candidates | time | check build date before 1890"
    },
    {
        "role": "user",
        "content": "[MAIN] Which statue was carved by Elena Voss? | [FOLLOW-UP] what material is it made of?"
    },
    {
        "role": "assistant",
        "content": "[MAIN] Elena Voss | base | find statues by sculptor\n[FOLLOW-UP] statue | common | lookup material attribute"
    },
    {
        "role": "user",
        "content": "[MAIN] Which team does mascot Sparky represent? | [FOLLOW-UP] did that team win the Galactic Cup in 2187? | [FOLLOW-UP] who was the captain of that team?"
    },
    {
        "role": "assistant",
        "content": "[MAIN] Sparky | sports | find mascot team\n[FOLLOW-UP] team | sports | check championship wins\n[FOLLOW-UP] team | sports | find captain"
    },
]


# Test cases: pick 10 diverse CWQ questions
TEST_CASES = [
    ("WebQTest-832", "Lou Seal is the mascot for the team that last won the World Series when?"),
    ("WebQTest-361", "What museum established before 1971 is there to see in Vienna, Austria?"),
    ("WebQTrn-557", "In the film with teh character named Winged Monkey #7, who played Dorothy?"),
    ("WebQTest-743", "Which of JFK's brother held the latest governmental position?"),
    ("WebQTrn-1841", "Which popular sports team in Spain, that won the 2014 Eurocup Finals championship?"),
    ("WebQTest-1785", "What TV series that had 3 episodes was Kim Richards in?"),
    ("WebQTrn-493", "What region is the country whose national anthem is Brabançonne located in?"),
    ("WebQTest-576", "What country in Central America appointed Otto Pérez Molina to a governmental position?"),
    ("WebQTrn-3376", "What educational institution with men's basketball sports team named Temple Owls did Kevin Hart go to school?"),
    ("WebQTest-1528", "In which movies, does Logan Lerman act in, that was production designed by Andrew Menzies?"),
]


def call_llm(system_prompt, messages, temperature=0.3, max_tokens=512):
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = requests.post(LLM_URL, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def stage1_decompose(question):
    messages = list(STAGE1_FEW_SHOT)
    messages.append({"role": "user", "content": f"Question: {question}"})
    return call_llm(STAGE1_SYSTEM, messages)


def stage2_discover(decomposition):
    messages = list(STAGE2_FEW_SHOT)
    messages.append({"role": "user", "content": decomposition})
    return call_llm(STAGE2_SYSTEM, messages)


def main():
    print("=" * 80)
    print("Stage 1 + Stage 2 (Discovery) Pipeline Test")
    print(f"Model: {MODEL} | Cases: {len(TEST_CASES)}")
    print("=" * 80)

    results = {}
    for case_id, question in TEST_CASES:
        print(f"\n{'═' * 60}")
        print(f"[{case_id}] {question}")
        print(f"{'─' * 60}")
        try:
            # Stage 1
            s1 = stage1_decompose(question)
            print(f"Stage 1: {s1}")

            # Stage 2
            s2 = stage2_discover(s1)
            print(f"Stage 2:")
            for line in s2.split("\n"):
                if line.strip():
                    print(f"  {line.strip()}")

            results[case_id] = {
                "question": question,
                "stage1": s1,
                "stage2": s2,
                "status": "ok",
            }
        except Exception as e:
            print(f"ERROR: {e}")
            results[case_id] = {"question": question, "error": str(e), "status": "error"}

    out_path = "/zhaoshu/subgraph/scripts/stage2_discovery_test.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
