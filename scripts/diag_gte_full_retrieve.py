#!/usr/bin/env python3
"""
Diagnostic: GTE retrieval against FULL subgraph (all entities + all relations).

Directly loads subgraph from pkl, no check_entities or explore_schema needed.
Tests whether GTE can recall the correct anchor entity and core relation.

Usage:
    python scripts/diag_gte_full_retrieve.py
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

# ── Ground truth cases ──────────────────────────────────────────────
# Manually verified from previous pipeline runs
GT_CASES = {
    "WebQTest-832_c334509bb5e02cacae1ba2e80c176499": {
        "question": "Lou Seal is the mascot for the team that last won the World Series when?",
        "gt_answers": ["2014 World Series"],
        "gt_anchor": "San Francisco Giants",
        "gt_relations": ["sports.sports_team.championships"],
        "note": "anchor via Lou Seal → mascot.team → SF Giants → championships → 2014 WS",
    },
    "WebQTrn-241_dfb6c97ac9bf2f0ac07f27dd80f9edc2": {
        "question": "What is the name of the airport that serves the city Nijmegen?",
        "gt_answers": ["Germany"],
        "gt_anchor": "Nijmegen",
        "gt_relations": ["aviation.airport.serves"],
        "note": "Nijmegen → airport.serves → Weeze Airport → country → Germany",
    },
    "WebQTrn-810_c334509bb5e02cacae1ba2e80c176499": {
        "question": "What year did the team with mascot named Lou Seal win the World Series?",
        "gt_answers": ["2014 World Series"],
        "gt_anchor": "Lou Seal",
        "gt_relations": ["sports.mascot.team", "sports.sports_team.championships"],
        "note": "multi-hop: Lou Seal → mascot.team → team → championships → year",
    },
    "WebQTest-590_6aad73acb74f304bc9acae44314164be": {
        "question": "Which man is the leader of the country that uses Allahu Akbar as its national anthem?",
        "gt_answers": ["Abdullah al-Thani"],
        "gt_anchor": "Allahu Akbar",
        "gt_relations": ["government.national_anthem_of_a_country.country", "government.government_office_or_title.office_holders"],
        "note": "Allahu Akbar → anthem.country → Libya → office_holders → leader",
    },
    "WebQTrn-662_7a992044f94b39edfc37ac5dcfcb3c26": {
        "question": "What was the name of the team that won the 2008 FIFA Club World Cup Final championship?",
        "gt_answers": ["Newton Heath L&YR F.C."],
        "gt_anchor": "2008 FIFA Club World Cup Final",
        "gt_relations": ["sports.sports_championship_event.champion"],
        "note": "2008 FIFA CWC Final → champion → team",
    },
    "WebQTest-1797_68a33792b0a1e18937dcd4b3f50d941e": {
        "question": "What group fought in the Battle of Vicksburg that was based in Montgomery?",
        "gt_answers": ["Confederate States of America"],
        "gt_anchor": "Battle of Vicksburg",
        "gt_relations": ["military.military_unit.participated_in"],
        "note": "Battle of Vicksburg → military_unit.participated_in → units",
    },
    "WebQTest-12_68d745a0657c86906382873e57294d6a": {
        "question": "Who was the governor of Ohio in 2011 that was in the government prior to 3-1-1983?",
        "gt_answers": ["Return J. Meigs, Jr."],
        "gt_anchor": "Ohio",
        "gt_relations": ["government.government_position_held.office_holder"],
        "note": "Ohio → gov_position.office_holder → governors",
    },
    "WebQTrn-124_f3990dc9aa470fa81ec4cf2912a9924f": {
        "question": "Which movie with a character called Ajila was directed by Angelina Jolie?",
        "gt_answers": ["In the Land of Blood and Honey"],
        "gt_anchor": "Angelina Jolie",
        "gt_relations": ["film.film.directed_by"],
        "note": "Angelina Jolie → film.directed_by → movies",
    },
    "WebQTest-576_01e2da60a2779c4ae4b5d1547499a4f8": {
        "question": "What Central American country uses the Guatemalan quetzal as its national currency?",
        "gt_answers": ["Guatemala"],
        "gt_anchor": "Guatemalan quetzal",
        "gt_relations": ["finance.currency.countries_used"],
        "note": "Guatemalan quetzal → currency.countries_used → Guatemala",
    },
    "WebQTrn-2069_0fa727f3b282196eb1097410b4be6818": {
        "question": "What language is spoken in the country where the Basque separatist group ETA was founded?",
        "gt_answers": ["Spanish Language"],
        "gt_anchor": "ETA",
        "gt_relations": ["location.location.contains", "location.country.languages_spoken"],
        "note": "ETA → location.contains → Spain → languages_spoken → Spanish",
    },
    "WebQTrn-3251_d8cddfe5e947e414b7735780ef1efff8": {
        "question": "What educational institution has a football sports team named Northern Colorado Bears football?",
        "gt_answers": ["University of Northern Colorado"],
        "gt_anchor": "Northern Colorado Bears football",
        "gt_relations": ["sports.team_venue_relationship.team", "education.athletics_brand.teams"],
        "note": "Northern Colorado Bears football → team → university",
    },
    "WebQTest-743_0a8cdba29cf260283b7c890b3609c0b9": {
        "question": "Which of JFK's brother held the latest governmental position?",
        "gt_answers": ["Robert F. Kennedy"],
        "gt_anchor": "JFK",
        "gt_relations": ["people.person.sibling"],
        "note": "JFK → person.sibling → brothers → filter by gov position",
    },
}


def gte_encode(texts: List[str]) -> np.ndarray:
    """Encode texts using GTE-large. Returns (N, dim) normalized embeddings."""
    resp = requests.post(f"{GTE_URL}/embed", json={"texts": texts, "batch_size": 64}, timeout=60)
    resp.raise_for_status()
    embs = np.array(resp.json()["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / (norms + 1e-8)


def is_cvt(name: str) -> bool:
    return bool(re.match(r'^[mg]\.[a-zA-Z0-9_]+$', name))


def _relation_to_readable(rel: str) -> str:
    """Convert domain.type.property_name → 'type property' (last 2 segments)."""
    parts = rel.split(".")
    if len(parts) >= 3:
        return parts[-2].replace("_", " ") + " " + parts[-1].replace("_", " ")
    return parts[-1].replace("_", " ")


def main():
    # 1. Load CWQ test pkl
    pkl_path = Path("/zhaoshu/SubgraphRAG-main/retrieve/data_files/cwq/processed/test.pkl")
    print(f"Loading subgraphs from {pkl_path}...")
    with pkl_path.open("rb") as f:
        samples = pickle.load(f)

    # Build lookup
    sample_map = {s["id"]: s for s in samples if "id" in s}
    print(f"Loaded {len(sample_map)} samples")

    # 2. Load CWQ ground truth for reference
    gt_answers_map = {}
    data_path = ROOT / "data/cwq/cwq_test.jsonl"
    with data_path.open() as f:
        for line in f:
            row = json.loads(line)
            cid = row.get("id", "")
            gt = row.get("ground_truth", {})
            ga = gt.get("global_truth_answers", [])
            oracle_rels = gt.get("oracle_relations", [])
            core_ents = gt.get("core_entities", [])
            core_rels = gt.get("core_relations", [])
            gt_answers_map[cid] = {
                "answers": ga,
                "oracle_relations": oracle_rels,
                "core_entities": core_ents,
                "core_relations": core_rels,
            }

    # 3. Run diagnostic per case
    results = []
    for cid, gt in GT_CASES.items():
        sample = sample_map.get(cid)
        if not sample:
            print(f"\n⚠ {cid}: not found in pkl data!")
            continue

        # Get ALL entities and ALL relations from subgraph
        text_ents = sample.get("text_entity_list", [])
        nontext_ents = sample.get("non_text_entity_list", [])
        all_ents_raw = text_ents + nontext_ents
        # Filter CVT nodes
        all_ents = [e for e in all_ents_raw if not is_cvt(e) and len(e) > 1]
        all_rels = list(sample.get("relation_list", []))

        question = gt["question"]
        gt_anchor = gt["gt_anchor"]
        gt_rels = gt["gt_relations"]

        # Also get oracle/core from data
        data_gt = gt_answers_map.get(cid, {})
        oracle_rels = data_gt.get("oracle_relations", [])
        core_rels = data_gt.get("core_relations", [])
        core_ents = data_gt.get("core_entities", [])

        print(f"\n{'='*80}")
        print(f"Case: {cid}")
        print(f"  Question: {question[:100]}")
        print(f"  Subgraph: {len(all_ents)} entities, {len(all_rels)} relations")
        print(f"  GT anchor: {gt_anchor}")
        print(f"  GT relations: {gt_rels}")
        print(f"  Oracle relations (from data): {oracle_rels}")
        print(f"  Core entities (from data): {core_ents[:8]}")

        # Encode question + all entities + readable relation names
        rel_readables = [_relation_to_readable(r) for r in all_rels]
        all_texts = [question] + all_ents + rel_readables
        embs = gte_encode(all_texts)

        q_emb = embs[0:1]                      # (1, dim)
        ent_embs = embs[1:1+len(all_ents)]      # (E, dim)
        rel_embs = embs[1+len(all_ents):]       # (R, dim)

        # ── Entity ranking ──
        ent_scores = (q_emb @ ent_embs.T)[0]    # (E,)
        ent_ranks = np.argsort(ent_scores)[::-1]

        print(f"\n  --- Entity Recall (top-15) ---")
        anchor_rank = None
        for rank, idx in enumerate(ent_ranks[:15], 1):
            name = all_ents[idx]
            score = ent_scores[idx]
            is_anchor = name.lower() == gt_anchor.lower() or gt_anchor.lower() in name.lower()
            if is_anchor and anchor_rank is None:
                anchor_rank = rank
            marker = " ← GT ANCHOR" if is_anchor else ""
            # Also mark core entities
            is_core = any(name.lower() == ce.lower() for ce in core_ents)
            core_marker = " [CORE]" if is_core and not is_anchor else ""
            print(f"    #{rank:2d}: {name} (sim={score:.4f}){marker}{core_marker}")

        if anchor_rank is None:
            # Check if anchor is in entity list at all
            anchor_in_list = gt_anchor in all_ents or any(gt_anchor.lower() == e.lower() for e in all_ents)
            print(f"    ⚠ GT anchor '{gt_anchor}' {'in entity list but NOT in top-15' if anchor_in_list else 'NOT in entity list!'}")
            if anchor_in_list:
                # Find its actual position
                for rank, idx in enumerate(ent_ranks, 1):
                    if all_ents[idx].lower() == gt_anchor.lower():
                        print(f"    Actual rank: #{rank} (sim={ent_scores[idx]:.4f})")
                        anchor_rank = rank
                        break

        # ── Relation ranking ──
        rel_scores = (q_emb @ rel_embs.T)[0]    # (R,)
        rel_ranks = np.argsort(rel_scores)[::-1]

        print(f"\n  --- Relation Recall (top-15, readable encoding) ---")
        gt_rel_found = {}
        for rank, idx in enumerate(rel_ranks[:15], 1):
            rel = all_rels[idx]
            readable = rel_readables[idx]
            score = rel_scores[idx]
            # Check if it matches any GT relation
            matched_gt = None
            for gr in gt_rels + oracle_rels:
                if rel == gr or rel in gr or gr in rel:
                    matched_gt = gr
                    break
            marker = f" ← GT: {matched_gt}" if matched_gt else ""
            if matched_gt and matched_gt not in gt_rel_found:
                gt_rel_found[matched_gt] = rank
            print(f"    #{rank:2d}: {rel} [{readable}] (sim={score:.4f}){marker}")

        # Check unfound GT relations
        for gr in gt_rels + oracle_rels:
            if gr not in gt_rel_found:
                if gr in all_rels:
                    actual_idx = all_rels.index(gr)
                    actual_rank = np.where(rel_ranks == actual_idx)[0]
                    if len(actual_rank) > 0:
                        print(f"    ⚠ GT rel '{gr}' at rank #{actual_rank[0]+1} (sim={rel_scores[actual_idx]:.4f})")
                    else:
                        print(f"    ⚠ GT rel '{gr}' in list but rank unknown")
                else:
                    print(f"    ⚠ GT rel '{gr}' NOT in subgraph relation list!")

        results.append({
            "case_id": cid,
            "anchor_rank": anchor_rank,
            "gt_rel_found": gt_rel_found,
            "total_entities": len(all_ents),
            "total_relations": len(all_rels),
        })

    # ── Summary ──
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    anchor_ranks = [r["anchor_rank"] for r in results if r["anchor_rank"] is not None]
    anchor_missing = [r["case_id"][:30] for r in results if r["anchor_rank"] is None]

    print(f"\nAnchor Recall ({len(anchor_ranks)}/{len(results)} found):")
    if anchor_ranks:
        for k in [1, 3, 5, 10]:
            hit = sum(1 for r in anchor_ranks if r <= k)
            print(f"  Top-{k}: {hit}/{len(results)} ({hit/len(results)*100:.0f}%)")
    if anchor_missing:
        print(f"  Missing: {anchor_missing}")

    print(f"\nRelation Recall:")
    total_oracle = 0
    found_oracle = 0
    for r in results:
        for rel, rank in r["gt_rel_found"].items():
            total_oracle += 1
            if rank <= 10:
                found_oracle += 1
    # Count unfound
    for r in results:
        case_gt = GT_CASES.get(r["case_id"], {})
        for gr in case_gt.get("gt_relations", []):
            total_oracle += 1  # count even unfound
        # Adjust: only count GT relations that should exist
    # Simpler: just count found vs total GT relations
    total_gt_rels = 0
    found_top10 = 0
    found_top5 = 0
    found_top3 = 0
    for r in results:
        case_gt = GT_CASES.get(r["case_id"], {})
        for gr in case_gt.get("gt_relations", []):
            total_gt_rels += 1
            if gr in r["gt_rel_found"]:
                rank = r["gt_rel_found"][gr]
                if rank <= 10: found_top10 += 1
                if rank <= 5: found_top5 += 1
                if rank <= 3: found_top3 += 1

    print(f"  Total GT relations: {total_gt_rels}")
    print(f"  Top-3: {found_top3}/{total_gt_rels} ({found_top3/total_gt_rels*100:.0f}%)")
    print(f"  Top-5: {found_top5}/{total_gt_rels} ({found_top5/total_gt_rels*100:.0f}%)")
    print(f"  Top-10: {found_top10}/{total_gt_rels} ({found_top10/total_gt_rels*100:.0f}%)")


if __name__ == "__main__":
    main()
