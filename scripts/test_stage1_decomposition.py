#!/usr/bin/env python3
"""Test Stage 1 (Semantic Decomposition) — marked main/follow-up questions."""

import json
import requests

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
    {
        "role": "user",
        "content": "Question: Which lighthouse near Cape Aurora was built before 1890?"
    },
    {
        "role": "assistant",
        "content": "[MAIN] Which lighthouses are near Cape Aurora? | [FOLLOW-UP] which one was built before 1890?"
    },
    {
        "role": "user",
        "content": "Question: What material is the statue carved by Elena Voss made of?"
    },
    {
        "role": "assistant",
        "content": "[MAIN] Which statue was carved by Elena Voss? | [FOLLOW-UP] what material is it made of?"
    },
    {
        "role": "user",
        "content": "Question: Which spacecraft launched by the Horizon program visited Mars first?"
    },
    {
        "role": "assistant",
        "content": "[MAIN] Which spacecraft were launched by the Horizon program? | [FOLLOW-UP] which ones visited Mars? | [FOLLOW-UP] which one visited first?"
    },
    {
        "role": "user",
        "content": "Question: Who painted the mural inside Crystal Dawn Cathedral?"
    },
    {
        "role": "assistant",
        "content": "[MAIN] Which mural is inside Crystal Dawn Cathedral? | [FOLLOW-UP] who painted it?"
    },
    {
        "role": "user",
        "content": "Question: The mascot Sparky represents the team that won the Galactic Cup in 2187 — who was the captain?"
    },
    {
        "role": "assistant",
        "content": "[MAIN] Which team does mascot Sparky represent? | [FOLLOW-UP] did that team win the Galactic Cup in 2187? | [FOLLOW-UP] who was the captain of that team?"
    },
    {
        "role": "user",
        "content": "Question: Which popular sports team in Arcadia won the 2099 Continental Cup?"
    },
    {
        "role": "assistant",
        "content": "[MAIN] Which sports team in Arcadia won the 2099 Continental Cup? | [FOLLOW-UP] is it considered popular?"
    },
    # Location split pattern — placed LAST for recency effect
    {
        "role": "user",
        "content": "Question: What city in the Northern Region hosted the 2050 Tech Summit?"
    },
    {
        "role": "assistant",
        "content": "[MAIN] Which city hosted the 2050 Tech Summit? | [FOLLOW-UP] is that city in the Northern Region?"
    },
    {
        "role": "user",
        "content": "Question: What province in the Coastal Zone elected Mara Voss as governor?"
    },
    {
        "role": "assistant",
        "content": "[MAIN] Which province elected Mara Voss as governor? | [FOLLOW-UP] is that province in the Coastal Zone?"
    },
]

TEST_CASES = [
    ("WebQTest-361", "What museum established before 1971 is there to see in Vienna, Austria?"),
    ("WebQTrn-60", "What Portuguese-speaking country had the net migration number of -10000?"),
    ("WebQTrn-557", "In the film with teh character named Winged Monkey #7, who played Dorothy?"),
    ("WebQTest-743", "Which of JFK's brother held the latest governmental position?"),
    ("WebQTest-832", "Lou Seal is the mascot for the team that last won the World Series when?"),
    ("WebQTrn-1841", "Which popular sports team in Spain, that won the 2014 Eurocup Finals championship?"),
    ("WebQTest-1785", "What TV series that had 3 episodes was Kim Richards in?"),
    ("WebQTrn-493", "What region is the country whose national anthem is Braban\u00e7onne located in?"),
    ("WebQTest-576", "What country in Central America appointed Otto P\u00e9rez Molina to a governmental position?"),
    ("WebQTrn-3376", "What educational institution with men's basketball sports team named Temple Owls did Kevin Hart go to school?"),
]


def call_llm(system_prompt, messages, temperature=0.3, max_tokens=1024):
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


def run_test(case_id, question):
    messages = list(FEW_SHOT)
    messages.append({"role": "user", "content": USER_TEMPLATE.format(question=question)})
    return call_llm(STAGE1_SYSTEM, messages)


def main():
    print("=" * 80)
    print("Stage 1 Decomposition Test — [MAIN]/[FOLLOW-UP] Marked")
    print(f"Model: {MODEL} | Cases: {len(TEST_CASES)}")
    print("=" * 80)

    results = {}
    for case_id, question in TEST_CASES:
        print(f"\n{'─' * 60}")
        print(f"[{case_id}] {question}")
        print(f"{'─' * 60}")
        try:
            output = run_test(case_id, question)
            print(output)
            results[case_id] = {"question": question, "output": output, "status": "ok"}
        except Exception as e:
            print(f"ERROR: {e}")
            results[case_id] = {"question": question, "output": str(e), "status": "error"}

    out_path = "/zhaoshu/subgraph/scripts/stage1_decomposition_test.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
