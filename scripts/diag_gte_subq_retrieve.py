#!/usr/bin/env python3
"""
Diagnostic: GTE retrieval using SUB-QUESTIONS (from Agent 0 decomposition).
Tests whether decomposed sub-questions give better recall than the original question.

For each case:
1. Run Agent 0 decomposition → get [MAIN], [HOP], [FILTER] sub-questions
2. Match each sub-question against ALL entities → find anchor
3. Match each sub-question against ALL relations → find core relations
4. Compare with original question matching
"""

import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
GTE_URL = "http://localhost:8003"
LLM_URL = "http://localhost:8000/v1/chat/completions"
LLM_MODEL = "qwen35-9b-local"

# Same GT cases as before
GT_CASES = {
    "WebQTest-832_c334509bb5e02cacae1ba2e80c176499": {
        "question": "Lou Seal is the mascot for the team that last won the World Series when?",
        "gt_anchor": "San Francisco Giants",
        "gt_relations": ["sports.sports_team.championships"],
    },
    "WebQTrn-241_dfb6c97ac9bf2f0ac07f27dd80f9edc2": {
        "question": "What is the name of the airport that serves the city Nijmegen?",
        "gt_anchor": "Nijmegen",
        "gt_relations": ["aviation.airport.serves"],
    },
    "WebQTrn-810_c334509bb5e02cacae1ba2e80c176499": {
        "question": "What year did the team with mascot named Lou Seal win the World Series?",
        "gt_anchor": "Lou Seal",
        "gt_relations": ["sports.mascot.team", "sports.sports_team.championships"],
    },
    "WebQTest-590_6aad73acb74f304bc9acae44314164be": {
        "question": "Which man is the leader of the country that uses Allahu Akbar as its national anthem?",
        "gt_anchor": "Allahu Akbar",
        "gt_relations": ["government.national_anthem_of_a_country.country", "government.government_office_or_title.office_holders"],
    },
    "WebQTrn-662_7a992044f94b39edfc37ac5dcfcb3c26": {
        "question": "What was the name of the team that won the 2008 FIFA Club World Cup Final championship?",
        "gt_anchor": "2008 FIFA Club World Cup Final",
        "gt_relations": ["sports.sports_championship_event.champion"],
    },
    "WebQTest-1797_68a33792b0a1e18937dcd4b3f50d941e": {
        "question": "What group fought in the Battle of Vicksburg that was based in Montgomery?",
        "gt_anchor": "Battle of Vicksburg",
        "gt_relations": ["military.military_unit.participated_in"],
    },
    "WebQTest-12_68d745a0657c86906382873e57294d6a": {
        "question": "Who was the governor of Ohio in 2011 that was in the government prior to 3-1-1983?",
        "gt_anchor": "Ohio",
        "gt_relations": ["government.government_position_held.office_holder"],
    },
    "WebQTrn-124_f3990dc9aa470fa81ec4cf2912a9924f": {
        "question": "Which movie with a character called Ajila was directed by Angelina Jolie?",
        "gt_anchor": "Angelina Jolie",
        "gt_relations": ["film.film.directed_by"],
    },
    "WebQTest-576_01e2da60a2779c4ae4b5d1547499a4f8": {
        "question": "What Central American country uses the Guatemalan quetzal as its national currency?",
        "gt_anchor": "Guatemalan quetzal",
        "gt_relations": ["finance.currency.countries_used"],
    },
    "WebQTrn-2069_0fa727f3b282196eb1097410b4be6818": {
        "question": "What language is spoken in the country where the Basque separatist group ETA was founded?",
        "gt_anchor": "ETA",
        "gt_relations": ["location.location.contains", "location.country.languages_spoken"],
    },
    "WebQTrn-3251_d8cddfe5e947e414b7735780ef1efff8": {
        "question": "What educational institution has a football sports team named Northern Colorado Bears football?",
        "gt_anchor": "Northern Colorado Bears football",
        "gt_relations": ["sports.team_venue_relationship.team", "education.athletics_brand.teams"],
    },
    "WebQTest-743_0a8cdba29cf260283b7c890b3609c0b9": {
        "question": "Which of JFK's brother held the latest governmental position?",
        "gt_anchor": "JFK",
        "gt_relations": ["people.person.sibling"],
    },
}

AGENT0_SYSTEM = r"""You break down a complex question into simple atomic sub-questions.
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


def gte_encode(texts: List[str]) -> np.ndarray:
    resp = requests.post(f"{GTE_URL}/embed", json={"texts": texts, "batch_size": 64}, timeout=60)
    resp.raise_for_status()
    embs = np.array(resp.json()["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / (norms + 1e-8)


def is_cvt(name: str) -> bool:
    return bool(re.match(r'^[mg]\.[a-zA-Z0-9_]+$', name))


def rel_to_readable(rel: str) -> str:
    parts = rel.split(".")
    if len(parts) >= 3:
        return parts[-2].replace("_", " ") + " " + parts[-1].replace("_", " ")
    return parts[-1].replace("_", " ")


def run_agent0(question: str) -> List[Dict[str, str]]:
    """Run Agent 0 decomposition, return list of {tag, text}."""
    resp = requests.post(LLM_URL, json={
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": AGENT0_SYSTEM},
            {"role": "user", "content": f"Question: {question}"},
        ],
        "max_tokens": 512,
        "temperature": 0.0,
        "top_p": 0.8,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=30)
    raw = resp.json()["choices"][0]["message"]["content"]

    # Parse sub-questions
    post_reasoning = raw
    reasoning_end = raw.find("</reasoning>")
    if reasoning_end >= 0:
        post_reasoning = raw[reasoning_end + len("</reasoning>"):]

    subqs = []
    segments = re.split(r'\s*\|\s*', post_reasoning)
    for seg in segments:
        seg = seg.strip()
        m = re.match(r'\[(MAIN|HOP|FILTER)\]\s*(.*)', seg, re.S)
        if m:
            subqs.append({"tag": m.group(1), "text": m.group(2).strip()})
    return subqs


def find_rank(candidates: List[str], target: str) -> Optional[int]:
    """Find rank of target in candidates list (case-insensitive substring match)."""
    target_lower = target.lower()
    for i, c in enumerate(candidates, 1):
        if c.lower() == target_lower or target_lower in c.lower():
            return i
    return None


def find_rel_rank(rel_list: List[str], scores: np.ndarray, ranked_indices: np.ndarray,
                  gt_rels: List[str]) -> Dict[str, int]:
    """Find rank of each GT relation in ranked results."""
    found = {}
    for gr in gt_rels:
        for rank, idx in enumerate(ranked_indices, 1):
            rel = rel_list[idx]
            if rel == gr or rel in gr or gr in rel:
                if gr not in found:
                    found[gr] = rank
                break
    return found


def main():
    # Load subgraphs
    pkl_path = Path("/zhaoshu/SubgraphRAG-main/retrieve/data_files/cwq/processed/test.pkl")
    with pkl_path.open("rb") as f:
        samples = pickle.load(f)
    sample_map = {s["id"]: s for s in samples if "id" in s}

    results = []

    for cid, gt in GT_CASES.items():
        sample = sample_map.get(cid)
        if not sample:
            continue

        question = gt["question"]
        gt_anchor = gt["gt_anchor"]
        gt_rels = gt["gt_relations"]

        # Get all entities and relations
        text_ents = sample.get("text_entity_list", [])
        nontext_ents = sample.get("non_text_entity_list", [])
        all_ents = [e for e in text_ents + nontext_ents if not is_cvt(e) and len(e) > 1]
        all_rels = list(sample.get("relation_list", []))
        rel_readables = [rel_to_readable(r) for r in all_rels]

        # Run Agent 0 decomposition
        subqs = run_agent0(question)
        print(f"\n{'='*80}")
        print(f"Case: {cid[:50]}")
        print(f"  Question: {question[:100]}")
        print(f"  Subgraph: {len(all_ents)} ents, {len(all_rels)} rels")
        print(f"  GT anchor: {gt_anchor}")
        print(f"  GT relations: {gt_rels}")
        print(f"  Sub-questions:")
        for sq in subqs:
            print(f"    [{sq['tag']}] {sq['text'][:80]}")

        if not subqs:
            subqs = [{"tag": "MAIN", "text": question}]

        # Encode all entities and relations once
        all_cands = all_ents + rel_readables
        cand_embs = gte_encode(all_cands)
        ent_embs = cand_embs[:len(all_ents)]
        rel_embs = cand_embs[len(all_ents):]

        # ── Per-query matching ──
        queries = [{"tag": "ORIG", "text": question}] + subqs

        for q in queries:
            q_emb = gte_encode([q["text"]])
            ent_scores = (q_emb @ ent_embs.T)[0]
            rel_scores = (q_emb @ rel_embs.T)[0]

            ent_ranks = np.argsort(ent_scores)[::-1]
            rel_ranks = np.argsort(rel_scores)[::-1]

            anchor_rank = find_rank([all_ents[i] for i in ent_ranks[:20]], gt_anchor)
            gt_rel_found = find_rel_rank(all_rels, rel_scores, rel_ranks, gt_rels)

            ent_top3 = [all_ents[i] for i in ent_ranks[:3]]
            rel_top3 = [(all_rels[i], f"{rel_scores[i]:.3f}") for i in rel_ranks[:3]]

            anchor_str = f"#{anchor_rank}" if anchor_rank else "NOT_FOUND"
            rel_str = ", ".join(f"{r}@#{rank}" for r, rank in gt_rel_found.items()) if gt_rel_found else "NONE"

            print(f"\n  [{q['tag']:6s}] \"{q['text'][:60]}\"")
            print(f"    Anchor: {anchor_str} | Top3 ents: {ent_top3}")
            print(f"    Rel recall: {rel_str}")
            print(f"    Top3 rels: {[(all_rels[i], rel_readables[i], f'{rel_scores[i]:.3f}') for i in rel_ranks[:3]]}")

            results.append({
                "case_id": cid,
                "query_tag": q["tag"],
                "query_text": q["text"],
                "anchor_rank": anchor_rank,
                "gt_rel_found": gt_rel_found,
            })

    # ── Summary ──
    print(f"\n{'='*80}")
    print("SUMMARY: Original Question vs Sub-Questions")
    print(f"{'='*80}")

    for tag in ["ORIG", "MAIN", "HOP", "FILTER"]:
        tag_results = [r for r in results if r["query_tag"] == tag]
        if not tag_results:
            continue

        anchor_ranks = [r["anchor_rank"] for r in tag_results if r["anchor_rank"] is not None]
        anchor_missing = len(tag_results) - len(anchor_ranks)

        total_rels = 0
        found_top3 = found_top5 = found_top10 = 0
        for r in tag_results:
            for gr, rank in r["gt_rel_found"].items():
                total_rels += 1
                if rank <= 3: found_top3 += 1
                if rank <= 5: found_top5 += 1
                if rank <= 10: found_top10 += 1

        print(f"\n  [{tag}] ({len(tag_results)} queries)")
        if anchor_ranks:
            for k in [1, 3, 5, 10]:
                hit = sum(1 for r in anchor_ranks if r <= k)
                print(f"    Anchor Top-{k}: {hit}/{len(tag_results)} ({hit/len(tag_results)*100:.0f}%)")
        if anchor_missing:
            print(f"    Anchor missing: {anchor_missing}")
        if total_rels > 0:
            print(f"    Rel Top-3: {found_top3}/{total_rels} ({found_top3/total_rels*100:.0f}%)")
            print(f"    Rel Top-5: {found_top5}/{total_rels} ({found_top5/total_rels*100:.0f}%)")
            print(f"    Rel Top-10: {found_top10}/{total_rels} ({found_top10/total_rels*100:.0f}%)")


if __name__ == "__main__":
    main()
