#!/usr/bin/env python3
"""Print top-10 entities and relations per case for manual inspection."""
import json, pickle, re, numpy as np, requests
from pathlib import Path

GTE_URL = "http://localhost:8003"
pkl_path = Path("/zhaoshu/SubgraphRAG-main/retrieve/data_files/cwq/processed/test.pkl")
with pkl_path.open("rb") as f:
    samples = pickle.load(f)
sample_map = {s["id"]: s for s in samples if "id" in s}

data_path = Path("data/cwq/cwq_test.jsonl")
gt_map = {}
with data_path.open() as f:
    for line in f:
        row = json.loads(line)
        gt_map[row["id"]] = {
            "answers": row.get("ground_truth", {}).get("global_truth_answers", []),
            "oracle_rels": row.get("ground_truth", {}).get("oracle_relations", []),
            "core_ents": row.get("ground_truth", {}).get("core_entities", []),
        }

cases = [
    ("WebQTest-832_c334509bb5e02cacae1ba2e80c176499", "Lou Seal is the mascot for the team that last won the World Series when?"),
    ("WebQTrn-241_dfb6c97ac9bf2f0ac07f27dd80f9edc2", "What is the name of the airport that serves the city Nijmegen?"),
    ("WebQTrn-810_c334509bb5e02cacae1ba2e80c176499", "What year did the team with mascot named Lou Seal win the World Series?"),
    ("WebQTest-590_6aad73acb74f304bc9acae44314164be", "Which man is the leader of the country that uses Allahu Akbar as its national anthem?"),
    ("WebQTrn-662_7a992044f94b39edfc37ac5dcfcb3c26", "What was the name of the team that won the 2008 FIFA Club World Cup Final championship?"),
    ("WebQTest-1797_68a33792b0a1e18937dcd4b3f50d941e", "What group fought in the Battle of Vicksburg that was based in Montgomery?"),
    ("WebQTest-12_68d745a0657c86906382873e57294d6a", "Who was the governor of Ohio in 2011 that was in the government prior to 3-1-1983?"),
    ("WebQTrn-124_f3990dc9aa470fa81ec4cf2912a9924f", "Which movie with a character called Ajila was directed by Angelina Jolie?"),
    ("WebQTest-576_01e2da60a2779c4ae4b5d1547499a4f8", "What Central American country uses the Guatemalan quetzal as its national currency?"),
    ("WebQTrn-2069_0fa727f3b282196eb1097410b4be6818", "What language is spoken in the country where the Basque separatist group ETA was founded?"),
    ("WebQTrn-3251_d8cddfe5e947e414b7735780ef1efff8", "What educational institution has a football sports team named Northern Colorado Bears football?"),
    ("WebQTest-743_0a8cdba29cf260283b7c890b3609c0b9", "Which of JFK's brother held the latest governmental position?"),
]

def is_cvt(n):
    return bool(re.match(r"^[mg]\.[a-zA-Z0-9_]+$", n))

def rel_readable(r):
    p = r.split(".")
    if len(p) >= 3:
        return p[-2].replace("_", " ") + " " + p[-1].replace("_", " ")
    return p[-1].replace("_", " ")

for cid, question in cases:
    sample = sample_map.get(cid)
    if not sample:
        continue
    gt = gt_map.get(cid, {})
    answers = gt.get("answers", [])
    core_ents = gt.get("core_ents", [])
    oracle_rels = gt.get("oracle_rels", [])

    text_ents = sample.get("text_entity_list", [])
    nontext_ents = sample.get("non_text_entity_list", [])
    all_ents = [e for e in text_ents + nontext_ents if not is_cvt(e) and len(e) > 1]
    all_rels = list(sample.get("relation_list", []))
    rel_texts = [rel_readable(r) for r in all_rels]

    all_texts = [question] + all_ents + rel_texts
    resp = requests.post(f"{GTE_URL}/embed", json={"texts": all_texts, "batch_size": 64}, timeout=60)
    embs = np.array(resp.json()["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / (norms + 1e-8)

    q_emb = embs[0:1]
    ent_embs = embs[1:1 + len(all_ents)]
    rel_embs = embs[1 + len(all_ents):]

    ent_scores = (q_emb @ ent_embs.T)[0]
    ent_ranks = np.argsort(ent_scores)[::-1]

    rel_scores = (q_emb @ rel_embs.T)[0]
    rel_ranks = np.argsort(rel_scores)[::-1]

    print(f"\n{'='*80}")
    print(f"Q: {question[:90]}")
    print(f"GT answers: {answers}")
    print(f"Core ents: {core_ents[:6]}")
    print(f"Oracle rels: {oracle_rels[:4]}")
    print(f"Subgraph: {len(all_ents)} ents, {len(all_rels)} rels")

    print(f"\n  Top-10 Entities:")
    for rank, idx in enumerate(ent_ranks[:10], 1):
        name = all_ents[idx]
        score = ent_scores[idx]
        is_core = any(name.lower() == c.lower() for c in core_ents)
        is_answer = any(name.lower() == a.lower() for a in answers)
        mark = ""
        if is_answer:
            mark = " <-- ANSWER"
        elif is_core:
            mark = " [core]"
        print(f"    #{rank:2d}: {name[:60]:60s} ({score:.4f}){mark}")

    print(f"\n  Top-10 Relations:")
    for rank, idx in enumerate(rel_ranks[:10], 1):
        rel = all_rels[idx]
        readable = rel_texts[idx]
        score = rel_scores[idx]
        is_oracle = rel in oracle_rels
        mark = " <-- ORACLE" if is_oracle else ""
        print(f"    #{rank:2d}: {readable[:45]:45s} ({score:.4f}) [{rel}]{mark}")
