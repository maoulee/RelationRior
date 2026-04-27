#!/usr/bin/env python3
"""Stage 2 Discovery v2: use actual subgraph domains per case."""

import json
import requests

LLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "qwen35-9b-local"

# Stage 1 (same as tested)
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

# Stage 2 system — domain comes from input
STAGE2_SYSTEM = r"""You analyze decomposed questions to identify anchor entities and relevant knowledge graph domains.

Input:
- Decomposed sub-questions with [MAIN] and [FOLLOW-UP] tags
- Available domains from the subgraph

For each sub-question, output one line:
[TAG] anchor | domain | intent

Where:
- TAG: [MAIN] or [FOLLOW-UP] (same as input)
- anchor: specific entity name to search (from question text)
- domain: pick exactly ONE from the available domains, choose the most relevant
- intent: brief phrase for what relation to explore

Rules:
- domain MUST be from the provided available domains list, never invent one
- anchor must be a specific named entity from the question
- Keep intent short and actionable
- Output ONLY the tagged lines, nothing else"""

STAGE2_FEW_SHOT = [
    {
        "role": "user",
        "content": "Available domains: location, common, time, base\n\n[MAIN] Which lighthouses are near Cape Aurora? | [FOLLOW-UP] which one was built before 1890?"
    },
    {
        "role": "assistant",
        "content": "[MAIN] Cape Aurora | location | find nearby lighthouses\n[FOLLOW-UP] lighthouse candidates | time | check build date before 1890"
    },
    {
        "role": "user",
        "content": "Available domains: base, common, type\n\n[MAIN] Which statue was carved by Elena Voss? | [FOLLOW-UP] what material is it made of?"
    },
    {
        "role": "assistant",
        "content": "[MAIN] Elena Voss | base | find statues by sculptor\n[FOLLOW-UP] statue | common | lookup material attribute"
    },
    {
        "role": "user",
        "content": "Available domains: sports, base, common, time\n\n[MAIN] Which team does mascot Sparky represent? | [FOLLOW-UP] did that team win the Galactic Cup in 2187? | [FOLLOW-UP] who was the captain of that team?"
    },
    {
        "role": "assistant",
        "content": "[MAIN] Sparky | sports | find mascot team\n[FOLLOW-UP] team | sports | check championship wins\n[FOLLOW-UP] team | sports | find captain"
    },
]


# Load per-case domains from CWQ data
def load_case_domains():
    case_map = {}
    with open("/zhaoshu/subgraph/data/cwq/cwq_test.jsonl") as f:
        for line in f:
            d = json.loads(line)
            case_id = d["id"].split("_")[0]
            user_msg = d["messages"][1]["content"]
            for l in user_msg.split("\n"):
                stripped = l.strip()
                if stripped and not stripped.startswith("[") and not stripped.startswith("Available") and "," in stripped and len(stripped.split(",")) > 5:
                    # Filter out meta domains
                    domains = [d.strip() for d in stripped.split(",") if not d.strip().startswith(("owl", "rdf"))]
                    if case_id not in case_map:
                        case_map[case_id] = ", ".join(domains)
                    break
    return case_map


TEST_CASES = [
    ("WebQTest-832", "Lou Seal is the mascot for the team that last won the World Series when?"),
    ("WebQTrn-557", "In the film with teh character named Winged Monkey #7, who played Dorothy?"),
    ("WebQTest-743", "Which of JFK's brother held the latest governmental position?"),
    ("WebQTrn-1841", "Which popular sports team in Spain, that won the 2014 Eurocup Finals championship?"),
    ("WebQTest-1785", "What TV series that had 3 episodes was Kim Richards in?"),
    ("WebQTrn-493", "What region is the country whose national anthem is Brabançonne located in?"),
    ("WebQTest-576", "What country in Central America appointed Otto Pérez Molina to a governmental position?"),
    ("WebQTrn-3376", "What educational institution with men's basketball sports team named Temple Owls did Kevin Hart go to school?"),
    ("WebQTest-1528", "In which movies, does Logan Lerman act in, that was production designed by Andrew Menzies?"),
    ("WebQTrn-3083", "Which nation uses The Internationale as the national anthem? Who was the President?"),
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


def main():
    case_domains = load_case_domains()

    print("=" * 80)
    print("Stage 2 Discovery v2 — per-case subgraph domains")
    print("=" * 80)

    for case_id, question in TEST_CASES:
        domains = case_domains.get(case_id, "NOT FOUND")
        print(f"\n{'═' * 60}")
        print(f"[{case_id}] {question}")
        print(f"Domains ({len(domains.split(','))}): {domains[:120]}...")
        print(f"{'─' * 60}")

        try:
            # Stage 1
            s1_msgs = list(STAGE1_FEW_SHOT) + [{"role": "user", "content": f"Question: {question}"}]
            s1 = call_llm(STAGE1_SYSTEM, s1_msgs)
            print(f"Stage 1: {s1}")

            # Stage 2 with real domains
            s2_msgs = list(STAGE2_FEW_SHOT) + [{
                "role": "user",
                "content": f"Available domains: {domains}\n\n{s1}"
            }]
            s2 = call_llm(STAGE2_SYSTEM, s2_msgs)
            print(f"Stage 2:")
            for line in s2.split("\n"):
                if line.strip():
                    print(f"  {line.strip()}")

        except Exception as e:
            print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
