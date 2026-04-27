#!/usr/bin/env python3
"""Diagnostic: GTE recall accuracy for anchor entity and core relation.

For each case:
1. Run Agent 0 decomposition → get sub-questions
2. Get full subgraph (all entities + all relations via direct API)
3. GTE match each sub-question against entities → check anchor rank
4. GTE match each sub-question against relations → check relation rank

Reports: for how many cases does GTE correctly recall the anchor as top-K
         and the correct relation as top-K.
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Config
KG_API_URL = os.getenv("KGQA_KG_API_URL", "http://localhost:8002").rstrip("/")
LLM_API_URL = os.getenv("KGQA_LLM_API_URL", "http://localhost:8000/v1").rstrip("/")
LLM_MODEL = os.getenv("KGQA_MODEL_NAME", "qwen35-9b-local")
GTE_API_URL = os.getenv("KGQA_GTE_API_URL", "http://localhost:8003").rstrip("/")

# Pick cases with known ground truth (case_id → {anchor, relations})
# These are manually verified from previous runs
TEST_CASES = {
    # Case 1: simple single-hop — anchor="San Francisco Giants", relation=sports.sports_team.championships
    "WebQTest-832_c334509bb5e02cacae1ba2e80c176499": {
        "question_hint": "Lou Seal mascot team won World Series in what year?",
        "gt_anchor": "San Francisco Giants",
        "gt_relations": ["sports.sports_team.championships"],
    },
    # Case 2: anchor="Ohio", relation=government.government_position_held.office_holder (or similar gov rel)
    "WebQTest-12_68d745a0657c86906382873e57294d6a": {
        "question_hint": "Governor of Ohio in 2011 before 3-1-1983?",
        "gt_anchor": "Ohio",
        "gt_relations": ["government.government_position_held.office_holder", "government.government_position_held.jurisdiction_of_office"],
    },
    # Case 3: anchor="Nijmegen", relation=aviation.airport.serves
    "WebQTrn-241_dfb6c97ac9bf2f0ac07f27dd80f9edc2": {
        "question_hint": "Airport serving Nijmegen",
        "gt_anchor": "Nijmegen",
        "gt_relations": ["aviation.airport.serves"],
    },
    # Case 4: anchor="2014 World Series", relation=sports.sports_team.championships (inverse)
    "WebQTrn-810_c334509bb5e02cacae1ba2e80c176499": {
        "question_hint": "Lou Seal mascot team won World Series what year?",
        "gt_anchor": "Lou Seal",
        "gt_relations": ["sports.mascot.team", "sports.sports_team.championships"],
    },
    # Case 5: anchor="Battle of Vicksburg", relation=military.military_unit.participated_in
    "WebQTest-1797_68a33792b0a1e18937dcd4b3f50d941e": {
        "question_hint": "Group that fought in Battle of Vicksburg based in Montgomery?",
        "gt_anchor": "Battle of Vicksburg",
        "gt_relations": ["military.military_unit.participated_in"],
    },
    # Case 6: anchor="Angelina Jolie", relation=film.film.directed_by or similar
    "WebQTrn-124_f3990dc9aa470fa81ec4cf2912a9924f": {
        "question_hint": "Movie with character Ajila directed by Angelina Jolie?",
        "gt_anchor": "Angelina Jolie",
        "gt_relations": ["film.film.directed_by", "film.film_character.portrayed_in_films"],
    },
    # Case 7: simple — anchor="Guatemala"
    "WebQTest-576_01e2da60a2779c4ae4b5d1547499a4f8": {
        "question_hint": "Central American country using GTQ?",
        "gt_anchor": "Guatemala",
        "gt_relations": ["finance.currency.countries_used"],
    },
    # Case 8: anchor="Spanish Language", answer=religion practiced in Spain
    "WebQTrn-2069_0fa727f3b282196eb1097410b4be6818": {
        "question_hint": "Language spoken in Mexico?",
        "gt_anchor": "Mexico",
        "gt_relations": ["location.country.languages_spoken"],
    },
}


async def _gte_retrieve(session, query, candidates, top_k=10):
    """GTE retrieve top-k candidates."""
    if not candidates:
        return []
    payload = {"query": query, "candidates": candidates, "top_k": top_k}
    async with session.post(f"{GTE_API_URL}/retrieve", json=payload, timeout=aiohttp.ClientTimeout(total=15)) as resp:
        data = await resp.json()
        return data.get("results", [])


async def _call_backend(session, tool_name, args, sample_id):
    url_map = {
        "check_entities": f"{KG_API_URL}/v2/find_entities",
        "find_entities": f"{KG_API_URL}/v2/find_entities",
        "explore_schema": f"{KG_API_URL}/v2/explore_schema",
    }
    url = url_map.get(tool_name)
    if not url:
        return {"is_success": False, "response_text": "", "found_end_entities": []}
    payload = {**args, "sample_id": sample_id}
    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
        return await resp.json()


def _extract_clean_question(user_message: str) -> str:
    if "Question:" in user_message:
        after = user_message.split("Question:", 1)[1]
        return after.strip().split("\n")[0].strip()
    return user_message.strip()


def _extract_domains(user_message: str) -> List[str]:
    match = re.search(r"Available Domains in Subgraph:\s*\n(.*?)(?:\n\n|\[Retrieval)", user_message, re.S)
    if match:
        return [d.strip() for d in match.group(1).strip().split(",") if d.strip()]
    return []


def _is_relation_name(name: str) -> bool:
    parts = name.split(".")
    return len(parts) >= 3 or bool(re.match(r'^m\.[a-z0-9]+$', name))


async def _run_agent0(session, question: str) -> str:
    """Run Agent 0 decomposition."""
    system = """You break down a complex question into simple atomic sub-questions.
Rules:
- Mark FIRST sub-question with [MAIN]
- Mark subsequent with [HOP] (traversal) or [FILTER] (constraint)
- Keep original entity names exactly
- Separate with |

Output:
<reasoning>
[ENTITY CHECK] named entities: <list>
</reasoning>
[MAIN] <first hop> | [HOP] <next> | [FILTER] <constraint>"""

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Question: {question}"},
        ],
        "max_tokens": 512,
        "temperature": 0.0,
        "top_p": 0.8,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with session.post(f"{LLM_API_URL}/chat/completions", json=payload) as resp:
        data = await resp.json()
        return data["choices"][0]["message"]["content"]


def _parse_subquestions(raw: str) -> List[Dict[str, str]]:
    """Parse [MAIN], [HOP], [FILTER] sub-questions from Agent 0 output."""
    post_reasoning = raw
    reasoning_end = raw.find("</reasoning>")
    if reasoning_end >= 0:
        post_reasoning = raw[reasoning_end + len("</reasoning>"):]

    segments = re.split(r'\s*\|\s*', post_reasoning)
    subqs = []
    for seg in segments:
        seg = seg.strip()
        m = re.match(r'\[(MAIN|HOP|FILTER)\]\s*(.*)', seg, re.S)
        if m:
            subqs.append({"tag": m.group(1), "text": m.group(2).strip()})
    return subqs


async def diagnose_case(
    session: aiohttp.ClientSession,
    case: Dict,
    gt: Dict,
    case_idx: int,
):
    """Run GTE recall diagnostic for one case."""
    case_id = case.get("id", "")
    user_message = next(
        (msg["content"] for msg in case.get("messages", []) if msg.get("role") == "user"),
        "",
    )
    question = _extract_clean_question(user_message)
    domains = _extract_domains(user_message)

    print(f"\n{'='*80}")
    print(f"[Case {case_idx}] {case_id}")
    print(f"  Question: {question[:100]}")
    print(f"  Domains: {len(domains)}")
    print(f"  GT anchor: {gt['gt_anchor']}")
    print(f"  GT relations: {gt['gt_relations']}")

    # Step 1: Agent 0 decomposition
    decomp_raw = await _run_agent0(session, question)
    subqs = _parse_subquestions(decomp_raw)
    print(f"\n  Sub-questions ({len(subqs)}):")
    for sq in subqs:
        print(f"    [{sq['tag']}] {sq['text'][:80]}")

    if not subqs:
        subqs = [{"tag": "MAIN", "text": question}]

    # Step 2: Get full subgraph — entities + relations
    # Entities from check_entities using entity names from decomposition
    entity_names = []
    for line in decomp_raw.split("\n"):
        if "[ENTITY CHECK]" in line:
            for m in re.finditer(r'"([^"]+)"', line):
                entity_names.append(m.group(1))
    if not entity_names:
        # Fallback: extract capitalized words
        entity_names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', question)

    all_entities = []
    entity_set = set()
    for name in entity_names:
        result = await _call_backend(session, "check_entities", {"entity_substring": name}, case_id)
        if result.get("success") and result.get("response_text"):
            for line in result["response_text"].split("\n"):
                line = line.strip()
                if line.startswith("- ") and not line.startswith("- relation:"):
                    ename = line[2:].split("[")[0].strip()
                    if ename and len(ename) > 1 and not _is_relation_name(ename):
                        if ename not in entity_set:
                            all_entities.append(ename)
                            entity_set.add(ename)

    # Relations from ALL domains
    all_relations = []
    relation_set = set()
    for domain in domains:
        result = await _call_backend(session, "explore_schema", {"pattern": domain}, case_id)
        if result.get("success"):
            rels = result.get("found_end_entities", [])
            if not rels and result.get("response_text"):
                for line in result["response_text"].split("\n"):
                    line = line.strip()
                    if line.startswith("- ") and "." in line:
                        rel_name = line[2:].strip()
                        if _is_relation_name(rel_name) and rel_name not in relation_set:
                            rels.append(rel_name)
                            relation_set.add(rel_name)
            for r in rels:
                if r not in relation_set:
                    all_relations.append(r)
                    relation_set.add(r)

    print(f"\n  Full subgraph: {len(all_entities)} entities, {len(all_relations)} relations")

    # Step 3: GTE matching — sub-questions vs entities
    print(f"\n  --- GTE Entity Recall (question → entities) ---")
    # Match original question against entities
    ent_results = await _gte_retrieve(session, question, all_entities, top_k=10)
    ent_ranked = [r["candidate"] for r in ent_results]
    ent_scores = {r["candidate"]: r["score"] for r in ent_results}

    gt_anchor = gt["gt_anchor"]
    anchor_found = False
    for rank, name in enumerate(ent_ranked, 1):
        marker = " ← GT ANCHOR" if name.lower() == gt_anchor.lower() or gt_anchor.lower() in name.lower() else ""
        if marker:
            anchor_found = True
        print(f"    #{rank}: {name} (sim={ent_scores[name]:.4f}){marker}")
    if not anchor_found:
        # Check if GT anchor is even in the entity list
        if gt_anchor in entity_set:
            print(f"    ⚠ GT anchor '{gt_anchor}' in entities but NOT in top-10!")
        else:
            print(f"    ⚠ GT anchor '{gt_anchor}' NOT in entity list at all!")

    # Step 4: GTE matching — sub-questions vs relations
    print(f"\n  --- GTE Relation Recall (per sub-question → relations) ---")
    gt_rels = gt["gt_relations"]
    for sq in subqs:
        rel_results = await _gte_retrieve(session, sq["text"], all_relations, top_k=10)
        print(f"\n  [{sq['tag']}] \"{sq['text'][:60]}\"")
        found_gt = set()
        for rank, r in enumerate(rel_results, 1):
            rel = r["candidate"]
            score = r["score"]
            is_gt = any(rel == g or rel in g or g in rel for g in gt_rels)
            if is_gt:
                found_gt.add(rel)
            marker = " ← GT REL" if is_gt else ""
            print(f"    #{rank}: {rel} (sim={score:.4f}){marker}")

        for g in gt_rels:
            if g not in found_gt:
                # Check if it's in the relation list at all
                if g in relation_set:
                    # Find its actual rank
                    for rank, r in enumerate(rel_results, 1):
                        if r["candidate"] == g:
                            print(f"    ⚠ GT rel '{g}' found but not in top-10 (rank would be >10)")
                            break
                    else:
                        print(f"    ⚠ GT rel '{g}' in relation list but NOT in GTE results at all!")
                else:
                    print(f"    ⚠ GT rel '{g}' NOT in relation list!")

    # Also test: match original question against relations
    print(f"\n  --- GTE Relation Recall (original question → relations) ---")
    q_rel_results = await _gte_retrieve(session, question, all_relations, top_k=10)
    found_gt = set()
    for rank, r in enumerate(q_rel_results, 1):
        rel = r["candidate"]
        score = r["score"]
        is_gt = any(rel == g or rel in g or g in rel for g in gt_rels)
        if is_gt:
            found_gt.add(rel)
        marker = " ← GT REL" if is_gt else ""
        print(f"    #{rank}: {rel} (sim={score:.4f}){marker}")

    return {
        "case_id": case_id,
        "entities": len(all_entities),
        "relations": len(all_relations),
        "gt_anchor": gt_anchor,
        "gt_relations": gt_rels,
    }


async def main():
    # Load cases
    data_path = ROOT / "data/cwq/cwq_test.jsonl"
    cases = {}
    with data_path.open() as f:
        for line in f:
            row = json.loads(line)
            cid = row.get("id", "")
            if cid in TEST_CASES:
                cases[cid] = row

    print(f"Found {len(cases)} test cases out of {len(TEST_CASES)}")

    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        for idx, (cid, gt) in enumerate(TEST_CASES.items(), 1):
            case = cases.get(cid)
            if not case:
                print(f"⚠ Case {cid} not found in data!")
                continue
            await diagnose_case(session, case, gt, idx)


if __name__ == "__main__":
    asyncio.run(main())
