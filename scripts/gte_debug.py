#!/usr/bin/env python3
"""GTE Debug Tool — Interactive entity/relation retrieval tester for CWQ cases.

Usage:
    python scripts/gte_debug.py
    python scripts/gte_debug.py -c 0          # Load case by pilot index
    python scripts/gte_debug.py -c WebQTest-590_...

Commands:
    case <id|idx>        Load case by CWQ ID or pilot index (0-49)
    info                 Show current case info
    q                    Show question and GT answers

    ents [text]          GTE entity search (default: question text)
    rels [text]          GTE relation search (default: question text)

    sparql               Show original SPARQL query
    trace                Show pipeline's GTE retrieval trace
    path                 Find paths from anchor to GT entities (≤3 hops)
    check <rel>          Check if relation exists in subgraph (fuzzy)
    compare              Compare SPARQL relations vs pipeline relations

    show rels            List all relations in subgraph
    show ents [N]        List first N entities (default 30)
    show nbr <entity>    Show all triples connected to entity
    show triple <h> <r> <t>   Search triples (* = wildcard)

    help                 Show this help
    quit / q             Exit
"""

import argparse, asyncio, aiohttp, json, pickle, re, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
GTE_URL = "http://localhost:8003/retrieve"


# ── Data ──

def load_pkl():
    pkl_dir = ROOT / "data/cwq_processed"
    out = {}
    for split in ["test", "train", "val"]:
        p = pkl_dir / f"{split}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                for d in pickle.load(f):
                    key = d.get("id", d.get("question_id", ""))
                    out[key] = d
                    prefix = key.split("_")[0]
                    out.setdefault(prefix, d)
    return out

def load_pilot():
    p = ROOT / "reports/stage_pipeline_test/cwq_50_stage_v7/pilot.json"
    return json.loads(p.read_text()) if p.exists() else []

def load_results():
    p = ROOT / "reports/stage_pipeline_test/chain_decompose_test/results.json"
    return json.loads(p.read_text()) if p.exists() else []

def load_sparql():
    p = ROOT / "data/cwq_sparql/test.json"
    if p.exists():
        return {d["ID"]: d for d in json.loads(p.read_text())}
    return {}

def norm(t):
    return re.sub(r"[^a-z0-9]", "", t.lower().strip())

def is_cvt(name):
    return bool(re.match(r"^[mg]\.\d+", name))

def rel_short(rel):
    return rel.split(".")[-1].replace("_", " ")


# ── GTE Call ──

async def gte(session, query, candidates, candidate_texts=None, top_k=20):
    payload = {
        "query": query,
        "candidates": candidates,
        "candidate_texts": candidate_texts,
        "top_k": top_k,
    }
    async with session.post(GTE_URL, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
        if resp.status != 200:
            print(f"  GTE error: HTTP {resp.status}")
            return []
        data = await resp.json()
        return data.get("results", [])


# ── Path Finding ──

def find_paths(anchor, targets, ents, rels, h_ids, r_ids, t_ids, max_hops=3):
    fwd = defaultdict(list)
    rev = defaultdict(list)
    for j in range(len(h_ids)):
        fwd[h_ids[j]].append((t_ids[j], r_ids[j]))
        rev[t_ids[j]].append((h_ids[j], r_ids[j]))

    results = []
    queue = [(anchor, [], {anchor})]
    for _ in range(max_hops):
        nxt = []
        for node, path, vis in queue:
            for nb, ri in fwd.get(node, []):
                if nb in vis: continue
                p = path + [(ri, "->", nb)]
                if nb in targets: results.append(p)
                nxt.append((nb, p, vis | {nb}))
            for nb, ri in rev.get(node, []):
                if nb in vis: continue
                p = path + [(ri, "<-", nb)]
                if nb in targets: results.append(p)
                nxt.append((nb, p, vis | {nb}))
        queue = nxt
    return results


# ── Main ──

async def main(args):
    print("Loading data...")
    pkl_map = load_pkl()
    pilot = load_pilot()
    results = load_results()
    sparql_map = load_sparql()
    result_map = {r.get("case_id", ""): r for r in results}
    print(f"  pkl: {len(pkl_map)} cases | pilot: {len(pilot)} | results: {len(results)} | sparql: {len(sparql_map)}")

    # State
    case_id = None
    sample = None
    result = None
    sparql = None
    ents = rels = h_ids = r_ids = t_ids = []
    rel_texts = []  # short human-readable for GTE

    def load_case(cid):
        nonlocal case_id, sample, result, sparql, ents, rels, h_ids, r_ids, t_ids, rel_texts
        if cid.isdigit():
            idx = int(cid)
            if 0 <= idx < len(pilot):
                cid = pilot[idx]["case_id"]
            elif 0 <= idx < len(results):
                cid = results[idx].get("case_id", cid)

        case_id = cid
        sample = pkl_map.get(cid)
        result = result_map.get(cid)
        sparql = sparql_map.get(cid)

        if not sample:
            sample = pkl_map.get(cid.split("_")[0])

        if sample:
            ents = list(sample.get("text_entity_list", [])) + list(sample.get("non_text_entity_list", []))
            rels = list(sample.get("relation_list", []))
            h_ids = list(sample.get("h_id_list", []))
            r_ids = list(sample.get("r_id_list", []))
            t_ids = list(sample.get("t_id_list", []))
            rel_texts = [f"{r} ; {rel_short(r)}" for r in rels]
        else:
            ents, rels, h_ids, r_ids, t_ids, rel_texts = [], [], [], [], [], []
        return sample is not None

    def show_info():
        if not sample:
            print("No case loaded.")
            return
        q = sample.get("question", "?")
        gt = result.get("gt_answers", []) if result else []
        gt_hit = result.get("gt_hit", "?") if result else "?"
        llm_hit = result.get("llm_hit", "?") if result else "?"
        anchor = result.get("anchor_name", "?") if result else "?"
        print(f"Case: {case_id}")
        print(f"Q: {q}")
        print(f"GT answers: {gt}")
        print(f"GT hit: {gt_hit} | LLM hit: {llm_hit}")
        print(f"Anchor: {anchor}")
        print(f"Subgraph: {len(ents)} ents, {len(rels)} rels, {len(h_ids)} triples")
        if sparql:
            print(f"SPARQL: available")

    # Pre-load
    if args.case:
        if load_case(args.case):
            show_info()
        else:
            print(f"Case '{args.case}' not found!")

    async with aiohttp.ClientSession() as session:
        print("\nGTE Debug Tool. Type 'help' for commands.\n")
        while True:
            try:
                raw = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not raw:
                continue

            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("quit", "exit"):
                break

            elif cmd == "help":
                print(__doc__)

            elif cmd == "case":
                if not arg:
                    print("Usage: case <id|idx>")
                    continue
                if load_case(arg):
                    show_info()
                else:
                    print(f"Case '{arg}' not found!")

            elif cmd in ("info", "i"):
                show_info()

            elif cmd in ("q", "question"):
                if sample:
                    print(f"Q: {sample.get('question', '?')}")
                    if result:
                        print(f"GT: {result.get('gt_answers', [])}")
                        print(f"GT hit: {result.get('gt_hit', '?')} | LLM hit: {result.get('llm_hit', '?')}")
                else:
                    print("No case loaded.")

            # ── GTE Entity Search ──
            elif cmd == "ents":
                if not ents:
                    print("No case loaded.")
                    continue
                query = arg or (sample.get("question", "") if sample else "")
                if not query:
                    print("Usage: ents [query_text]")
                    continue
                clean = [e for e in ents if e and len(e) > 1 and not is_cvt(e)]
                print(f"Querying {len(clean)} entities: '{query[:60]}'")
                rows = await gte(session, query, clean, top_k=30)
                if rows:
                    for i, r in enumerate(rows[:30]):
                        c = r.get("candidate", "?")
                        s = r.get("score", 0)
                        mark = ""
                        if result:
                            if c == result.get("anchor_name"): mark += " ← ANCHOR"
                            if c in result.get("gt_answers", []): mark += " ← GT"
                        print(f"  {i+1}. {c} (gte={s:.4f}){mark}")
                else:
                    print("  No results")

            # ── GTE Relation Search ──
            elif cmd == "rels":
                if not rels:
                    print("No case loaded.")
                    continue
                query = arg or (sample.get("question", "") if sample else "")
                if not query:
                    print("Usage: rels [query_text]")
                    continue
                print(f"Querying {len(rels)} relations: '{query[:60]}'")
                rows = await gte(session, query, rels, candidate_texts=rel_texts, top_k=30)
                if rows:
                    for i, r in enumerate(rows[:30]):
                        c = r.get("candidate", "?")
                        s = r.get("score", 0)
                        mark = ""
                        if result:
                            for si, srids in enumerate(result.get("step_relations", [])):
                                if any(rid < len(rels) and rels[rid] == c for rid in srids):
                                    mark += f" ← PIPE step{si+1}"
                        print(f"  {i+1}. {c} (gte={s:.4f}){mark}")
                else:
                    print("  No results")

            # ── SPARQL ──
            elif cmd == "sparql":
                if not sparql:
                    print("No SPARQL data.")
                    continue
                print(f"Question: {sparql.get('question','')}")
                print(f"Machine Q: {sparql.get('machine_question','')}")
                print(f"Type: {sparql.get('compositionality_type','')}")
                print(f"\n{sparql.get('sparql','')}")

            # ── Pipeline Trace ──
            elif cmd == "trace":
                if not result:
                    print("No pipeline result.")
                    continue
                print("=== Pipeline GTE Trace ===\n")
                for d in result.get("entity_retrieval_details", []):
                    if "ner" in d.get("role", ""):
                        print(f"NER (role={d.get('role','')}):")
                        for name, score in d.get("ner_top_ents", []):
                            sel = " ← SELECTED" if name == d.get("selected") else ""
                            print(f"  {name} (gte={score:.4f}){sel}")

                rrd = result.get("relation_retrieval_details", [])
                for si, step_rr in enumerate(rrd):
                    print(f"\nStep {si+1} GTE Retrieval:")
                    if isinstance(step_rr, dict):
                        gte_idx = set(step_rr.get("gte_indices", []))
                        for qi, q in enumerate(step_rr.get("queries", [])):
                            print(f"  Query {qi+1}: '{q.get('query','')[:50]}'")
                            for item in q.get("top_k", [])[:10]:
                                name = item.get("candidate", "?")
                                score = item.get("score", 0)
                                idx = item.get("rel_idx", "?")
                                kept = " [KEPT]" if idx in gte_idx else ""
                                print(f"    {item.get('rank','?')}. {name} (score={score:.4f}, idx={idx}){kept}")

                print(f"\nFinal step_relations:")
                for si, srids in enumerate(result.get("step_relations", [])):
                    names = [rels[rid] if rid < len(rels) else f"?{rid}" for rid in srids]
                    print(f"  Step {si+1}: {names}")

            # ── Path Finding ──
            elif cmd == "path":
                if not result or not sample:
                    print("Load case first.")
                    continue
                aidx = result.get("anchor_idx", -1)
                gt = result.get("gt_answers", [])
                if aidx < 0:
                    print("No anchor.")
                    continue

                gt_idx = set()
                for g in gt:
                    gn = norm(g)
                    for idx, e in enumerate(ents):
                        if norm(e) == gn or (len(gn) >= 4 and gn in norm(e)):
                            gt_idx.add(idx)
                            break

                a_name = ents[aidx] if aidx < len(ents) else "?"
                gt_names = [ents[i] for i in gt_idx if i < len(ents)]
                print(f"Anchor: {a_name} (idx={aidx})")
                print(f"GT in subgraph: {gt_names} (indices={gt_idx})")

                if gt_idx:
                    paths = find_paths(aidx, gt_idx, ents, rels, h_ids, r_ids, t_ids)
                    print(f"\nPaths found: {len(paths)}")
                    seen = set()
                    for p in paths[:20]:
                        pat = tuple((ri, d) for ri, d, _ in p)
                        if pat in seen: continue
                        seen.add(pat)
                        s = a_name
                        for ri, direction, nid in p:
                            rn = rels[ri] if ri < len(rels) else f"?{ri}"
                            nn = ents[nid] if nid < len(ents) else f"?{nid}"
                            arrow = f"  --[{rn}]--> " if direction == "->" else f"  <--[{rn}]-- "
                            s += f"\n{arrow}{nn}"
                        print(f"\n{s}")
                    if len(paths) > 20:
                        print(f"\n... {len(paths)-20} more paths")
                else:
                    print("GT entity not found in subgraph.")

            # ── Check Relation ──
            elif cmd == "check":
                if not arg:
                    print("Usage: check <relation_name>")
                    continue
                if not rels:
                    print("No case loaded.")
                    continue
                qn = norm(arg)
                found = []
                for ri, r in enumerate(rels):
                    rn = norm(r)
                    if qn in rn or rn in qn:
                        found.append((ri, r))
                if found:
                    print(f"Found {len(found)} matching relations:")
                    for ri, r in found:
                        cnt = sum(1 for x in r_ids if x == ri)
                        mark = ""
                        if result:
                            for si, srids in enumerate(result.get("step_relations", [])):
                                if ri in srids: mark += f" ← PIPE step{si+1}"
                        print(f"  idx={ri}: {r} ({cnt} triples){mark}")
                else:
                    print(f"'{arg}' NOT found in subgraph")

            # ── Show ──
            elif cmd == "show":
                sp = arg.split(maxsplit=1)
                if not sp:
                    print("Usage: show rels|ents|nbr|triple")
                    continue

                if sp[0] == "rels":
                    print(f"All {len(rels)} relations:")
                    for ri, r in enumerate(rels):
                        cnt = sum(1 for x in r_ids if x == ri)
                        print(f"  {ri}: {r} ({cnt} triples)")

                elif sp[0] == "ents":
                    n = int(sp[1]) if len(sp) > 1 else 30
                    clean = [(i, e) for i, e in enumerate(ents) if not is_cvt(e)]
                    print(f"Entities ({min(n, len(clean))}/{len(clean)} non-CVT):")
                    for idx, name in clean[:n]:
                        deg = sum(1 for h, t in zip(h_ids, t_ids) if h == idx or t == idx)
                        print(f"  {idx}: {name} (deg={deg})")

                elif sp[0] == "nbr":
                    if len(sp) < 2:
                        print("Usage: show nbr <entity>")
                        continue
                    tn = norm(sp[1])
                    found = None
                    for idx, e in enumerate(ents):
                        if norm(e) == tn or (len(tn) >= 3 and tn in norm(e)):
                            found = idx
                            break
                    if found is None:
                        print(f"'{sp[1]}' not found")
                        continue
                    ename = ents[found]
                    print(f"Neighbors of {ename} (idx={found}):")
                    for j in range(len(h_ids)):
                        if h_ids[j] == found:
                            rn = rels[r_ids[j]] if r_ids[j] < len(rels) else "?"
                            tn = ents[t_ids[j]] if t_ids[j] < len(ents) else "?"
                            print(f"  --[{rn}]--> {tn}")
                        if t_ids[j] == found:
                            rn = rels[r_ids[j]] if r_ids[j] < len(rels) else "?"
                            hn = ents[h_ids[j]] if h_ids[j] < len(ents) else "?"
                            print(f"  <--[{rn}]-- {hn}")

                elif sp[0] == "triple":
                    tp = arg.split()
                    if len(tp) < 2:
                        print("Usage: show triple <head> <rel> <tail> (* = wildcard)")
                        continue
                    h_p = tp[1] if len(tp) > 1 else "*"
                    r_p = tp[2] if len(tp) > 2 else "*"
                    t_p = tp[3] if len(tp) > 3 else "*"
                    cnt = 0
                    for j in range(len(h_ids)):
                        hn = ents[h_ids[j]] if h_ids[j] < len(ents) else "?"
                        rn = rels[r_ids[j]] if r_ids[j] < len(rels) else "?"
                        tn = ents[t_ids[j]] if t_ids[j] < len(ents) else "?"
                        if (h_p == "*" or norm(h_p) in norm(hn)) and \
                           (r_p == "*" or norm(r_p) in norm(rn)) and \
                           (t_p == "*" or norm(t_p) in norm(tn)):
                            print(f"  ({hn}, {rn}, {tn})")
                            cnt += 1
                            if cnt >= 50:
                                print(f"  ... (showing first 50)")
                                break
                    print(f"  Found {cnt} matches")

                else:
                    print("Usage: show rels|ents|nbr|triple")

            # ── Compare ──
            elif cmd == "compare":
                if not sparql or not result:
                    print("Load case with SPARQL data first.")
                    continue
                st = sparql.get("sparql", "")
                s_rels = [r for r in re.findall(r"ns:([a-zA-Z][a-zA-Z0-9_.]+)", st) if "." in r and not r.startswith("m.")]
                pipe_rids = set()
                for srids in result.get("step_relations", []):
                    pipe_rids.update(srids)
                pipe_rels = {rels[rid] for rid in pipe_rids if rid < len(rels)}
                gte_all = set()
                for step_rr in result.get("relation_retrieval_details", []):
                    if isinstance(step_rr, dict):
                        for q in step_rr.get("queries", []):
                            for item in q.get("top_k", []):
                                gte_all.add(item.get("candidate", ""))

                print(f"SPARQL relations ({len(s_rels)}):")
                for sr in s_rels:
                    srn = norm(sr)
                    in_pipe = in_gte = False
                    sub_match = None
                    for ri, r in enumerate(rels):
                        rn = norm(r)
                        if srn in rn or rn in srn:
                            sub_match = f"{r} (idx={ri})"
                            in_pipe = ri in pipe_rids
                            in_gte = r in gte_all
                            break
                    if sub_match:
                        st = "✓ PIPELINE" if in_pipe else ("△ GTE but not selected" if in_gte else "✗ In subgraph, NOT retrieved")
                    else:
                        st = "✗ NOT in subgraph"
                    print(f"  {sr}: {st}")
                    if sub_match:
                        print(f"    → {sub_match}")

            else:
                print(f"Unknown: {cmd}. Type 'help'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GTE Debug Tool")
    parser.add_argument("-c", "--case", help="Pre-load case by ID or pilot index")
    asyncio.run(main(parser.parse_args()))
