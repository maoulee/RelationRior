#!/usr/bin/env python3
"""Test Agent 2.5 V3 with restructured Agent 1.6 (entity-first → RDF triples)."""

import json
import requests
import sys

sys.path.insert(0, '/zhaoshu/subgraph')
from config.subagent_prompts import AGENT0_SYSTEM, AGENT1_SYSTEM_V2, AGENT2_SYSTEM_V3

LLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "qwen35-9b-local"

AGENT0_FEWSHOT = [
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


def call_llm(system_prompt, messages, temperature=0.3, max_tokens=1024):
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = requests.post(LLM_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def extract_cwq_context(case_id):
    """Extract domains, hint entities, and relations from CWQ data."""
    with open("/zhaoshu/subgraph/data/cwq/cwq_test.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if case_id in d["id"]:
                user_msg = d["messages"][1]["content"]
                lines = user_msg.split("\n")

                domains = ""
                for l in lines:
                    if "Available Domains" in l or ("," in l and len(l.split(",")) > 5):
                        domains = l.replace("Available Domains in Subgraph:", "").strip()
                        break

                entities = []
                in_entities = False
                for l in lines:
                    if "Suggested Start Entities" in l:
                        in_entities = True
                        continue
                    if in_entities:
                        if l.startswith("- "):
                            entities.append(l[2:].strip())
                        elif l.strip() == "" and entities:
                            in_entities = False

                relations_raw = []
                in_relations = False
                for l in lines:
                    if "Suggested Relations" in l:
                        in_relations = True
                        continue
                    if in_relations:
                        if l.startswith("- "):
                            relations_raw.append(l[2:].strip())
                        elif l.strip() == "" and relations_raw:
                            in_relations = False

                domain_groups = {}
                for rel in relations_raw:
                    top = rel.split(".")[0]
                    domain_groups.setdefault(top, []).append(rel)

                return {"domains": domains, "entities": entities, "relations_by_domain": domain_groups}
    return None


def format_entities(entities):
    return "check_entities results:\n" + "\n".join(f'  - "{e}"' for e in entities)


def format_relations(domain_groups):
    lines = ["explore_schema results:"]
    for domain, rels in sorted(domain_groups.items()):
        lines.append(f"  {domain}:")
        for r in rels:
            lines.append(f"    - {r}")
    return "\n".join(lines)


TEST_CASES = [
    ("WebQTest-832", "Lou Seal is the mascot for the team that last won the World Series when?"),
    ("WebQTrn-557", "In the film with teh character named Winged Monkey #7, who played Dorothy?"),
    ("WebQTest-743", "Which of JFK's brother held the latest governmental position?"),
    ("WebQTrn-1841", "Which popular sports team in Spain, that won the 2014 Eurocup Finals championship?"),
    ("WebQTest-576", "What country in Central America appointed Otto Pérez Molina to a governmental position?"),
    ("WebQTrn-3376", "What educational institution with men's basketball sports team named Temple Owls did Kevin Hart go to school?"),
    ("WebQTest-1785", "What TV series that had 3 episodes was Kim Richards in?"),
    ("WebQTrn-493", "What region is the country whose national anthem is Brabançonne located in?"),
]


def run_pipeline(case_id, question):
    ctx = extract_cwq_context(case_id)
    if not ctx:
        return {"error": f"Case {case_id} not found"}

    # Agent 0
    s0 = call_llm(AGENT0_SYSTEM, list(AGENT0_FEWSHOT) + [{"role": "user", "content": f"Question: {question}"}])

    # Agent 1.6
    s16 = call_llm(AGENT1_SYSTEM_V2, [{"role": "user", "content":
        f"## Question\n\n{question}\n\n## Decomposed Question\n\n{s0}\n\n## Available Domains\n\n{ctx['domains']}\n\n"
        f"Analyze each sub-question as a triple. List all entities to verify."
    }])

    # Agent 2.5
    s25 = call_llm(AGENT2_SYSTEM_V3, [{"role": "user", "content":
        f"## Question\n\n{question}\n\n## Decomposed Question\n\n{s0}\n\n"
        f"## Discovery Analysis (triples from Agent 1.6)\n\n{s16}\n\n"
        f"## Verified Entities\n\n{format_entities(ctx['entities'])}\n\n"
        f"## Available Relations\n\n{format_relations(ctx['relations_by_domain'])}\n\n"
        f"Build main plan for [MAIN]. For each [FOLLOW-UP], determine filter_entity and/or filter_relations for Agent 3.5."
    }], max_tokens=1536)

    return {"case_id": case_id, "question": question, "decomposition": s0, "discovery": s16, "plan": s25}


def main():
    print("=" * 80)
    print("Agent 0 → 1.6 (entity-first) → 2.5 V3 Pipeline Test")
    print(f"Model: {MODEL} | Cases: {len(TEST_CASES)}")
    print("=" * 80)

    results = {}
    for case_id, question in TEST_CASES:
        print(f"\n{'═' * 70}")
        print(f"[{case_id}] {question}")
        print(f"{'─' * 70}")
        try:
            r = run_pipeline(case_id, question)
            print(f"\n[Agent 0] {r['decomposition']}")
            print(f"\n[Agent 1.6]")
            for line in r["discovery"].split("\n"):
                if line.strip():
                    print(f"  {line.strip()}")
            print(f"\n[Agent 2.5]")
            for line in r["plan"].split("\n"):
                if line.strip():
                    print(f"  {line.strip()}")
            results[case_id] = r
        except Exception as e:
            print(f"ERROR: {e}")
            results[case_id] = {"case_id": case_id, "question": question, "error": str(e)}

    out = "/zhaoshu/subgraph/scripts/plan_v3_test.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n\nResults saved to {out}")


if __name__ == "__main__":
    main()
