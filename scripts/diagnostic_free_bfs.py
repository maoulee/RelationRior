#!/usr/bin/env python3
"""Diagnostic: Free 3-hop traversal, check if last-hop relation is from L2."""
import json, pickle
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
CWQ_TEST = Path("/zhaoshu/SubgraphRAG-main/retrieve/data_files/cwq/processed/test.pkl")
CWQ_TRAIN = Path("/zhaoshu/SubgraphRAG-main/retrieve/data_files/cwq/processed/train.pkl")
V14 = ROOT / "reports/stage_pipeline_test/answer_template_v14/results.json"

def is_cvt(name):
    if not name: return False
    n = name.strip()
    return n.startswith('m.') or n.startswith('g.')

def load_all():
    with open(CWQ_TEST, "rb") as f: test = pickle.load(f)
    with open(CWQ_TRAIN, "rb") as f: train = pickle.load(f)
    cwq = {s['id']: s for s in test + train}
    with open(V14) as f: v14 = json.load(f)
    return cwq, v14

def free_bfs(start_idx, fwd, rev, ents, rels, max_depth=3):
    """Free BFS: any relation allowed, CVT nodes auto-traverse.
    Returns list of (depth, path) for all reached non-start nodes.
    path = [(src_idx, rel_idx, tgt_idx), ...]
    """
    visited = {start_idx}
    frontier = [(start_idx, [])]
    results = []
    for depth in range(1, max_depth + 1):
        next_frontier = []
        for node, path in frontier:
            # Get ALL edges (fwd + rev)
            edges = []
            for (tgt, r) in fwd.get(node, []):
                edges.append((tgt, r))
            for (tgt, r) in rev.get(node, []):
                edges.append((tgt, r))

            for tgt, r in edges:
                if tgt in visited:
                    continue
                visited.add(tgt)
                new_path = path + [(node, r, tgt)]
                next_frontier.append((tgt, new_path))
                results.append((depth, tgt, new_path))
        frontier = next_frontier
    return results

def fmt_path(path, ents, rels):
    """Format path for display."""
    parts = []
    for src, r, tgt in path:
        s = ents[src][:25] if src < len(ents) else str(src)
        rn = rels[r] if r < len(rels) else f"r{r}"
        t = ents[tgt][:25] if tgt < len(ents) else str(tgt)
        parts.append(f"{s} --[{rn}]--> {t}")
    return " | ".join(parts)

def main():
    cwq, v14 = load_all()
    weak = [c for c in v14 if not c.get('gt_hit', True) or not c.get('llm_hit', True)]
    print(f"=== Free 3-hop traversal ({len(weak)} weak cases) ===\n")

    for case in weak:
        qid = case['case_id']
        sample = cwq.get(qid)
        if not sample: print(f"{qid}: not found\n"); continue

        ents = sample.get('text_entity_list', []) + sample.get('non_text_entity_list', [])
        rels = list(sample.get('relation_list', []))
        h_ids, r_ids, t_ids = sample['h_id_list'], sample['r_id_list'], sample['t_id_list']

        fwd = defaultdict(list)
        rev = defaultdict(list)
        for i in range(len(h_ids)):
            fwd[h_ids[i]].append((t_ids[i], r_ids[i]))
            rev[t_ids[i]].append((h_ids[i], r_ids[i]))

        anchor_idx = case['anchor_idx']
        anchor_name = case['anchor_name']
        sr = case['step_relations']
        gt = case['gt_answers']
        gt_hit = case.get('gt_hit', False)
        llm_hit = case.get('llm_hit', False)

        # L2 rel indices
        l2_rels = set(sr[1]) if len(sr) > 1 else set()
        all_layer_rels = set()
        for s in sr: all_layer_rels.update(s)

        # GT entity indices
        gt_indices = set()
        for g in (gt if isinstance(gt, list) else [gt]):
            for i, e in enumerate(ents):
                if e and str(g).lower() == str(e).lower():
                    gt_indices.add(i)

        print(f"{'='*70}")
        print(f"Case: {qid}")
        print(f"  Q: {case['question']}")
        print(f"  GT: {gt} (idx={gt_indices}) | GT_HIT={gt_hit} LLM_HIT={llm_hit}")
        print(f"  Anchor: {anchor_name} idx={anchor_idx}")
        for li, s in enumerate(sr):
            print(f"  L{li+1} rels: {[f'{r}:{rels[r]}' for r in s]}")

        # === Test 1: Free BFS from anchor, 3 hops ===
        print(f"\n  --- Free 3-hop from anchor={anchor_name} ---")
        results = free_bfs(anchor_idx, fwd, rev, ents, rels, max_depth=3)
        gt_paths = [(d, n, p) for d, n, p in results if n in gt_indices]
        if gt_paths:
            for d, n, p in gt_paths[:5]:
                last_rel = p[-1][1]
                last_rel_name = rels[last_rel] if last_rel < len(rels) else f"r{last_rel}"
                in_l2 = "✓ L2" if last_rel in l2_rels else "✗ not L2"
                in_any = "L1" if last_rel in (set(sr[0]) if len(sr)>0 else set()) else ("L2" if last_rel in l2_rels else "OTHER")
                print(f"    ✓ d={d}: {fmt_path(p, ents, rels)}")
                print(f"      last_rel={last_rel_name} [{in_any}] {in_l2}")
        else:
            print(f"    ✗ GT not reachable in 3 hops")

        # === Test 2: Free BFS from anchor, 4 hops ===
        print(f"\n  --- Free 4-hop from anchor={anchor_name} ---")
        results4 = free_bfs(anchor_idx, fwd, rev, ents, rels, max_depth=4)
        gt_paths4 = [(d, n, p) for d, n, p in results4 if n in gt_indices]
        if gt_paths4:
            for d, n, p in gt_paths4[:3]:
                last_rel = p[-1][1]
                last_rel_name = rels[last_rel] if last_rel < len(rels) else f"r{last_rel}"
                in_any = "L1" if last_rel in (set(sr[0]) if len(sr)>0 else set()) else ("L2" if last_rel in l2_rels else "OTHER")
                print(f"    ✓ d={d}: {fmt_path(p, ents, rels)}")
                print(f"      last_rel={last_rel_name} [{in_any}]")
        else:
            print(f"    ✗ GT not reachable in 4 hops")

        # === Test 3: For Case 590 - correct anchor (anthem) ===
        if '590' in qid:
            anthem_idx = None
            for i, e in enumerate(ents):
                if e and 'libya, libya, libya' in str(e).lower():
                    anthem_idx = i; break
            if anthem_idx is not None:
                print(f"\n  --- Free 3-hop from CORRECT anchor=Libya anthem (idx={anthem_idx}) ---")
                results_an = free_bfs(anthem_idx, fwd, rev, ents, rels, max_depth=3)
                gt_an = [(d, n, p) for d, n, p in results_an if n in gt_indices]
                if gt_an:
                    for d, n, p in gt_an[:5]:
                        last_rel = p[-1][1]
                        last_rel_name = rels[last_rel] if last_rel < len(rels) else f"r{last_rel}"
                        in_any = "L1" if last_rel in (set(sr[0]) if len(sr)>0 else set()) else ("L2" if last_rel in l2_rels else "OTHER")
                        print(f"    ✓ d={d}: {fmt_path(p, ents, rels)}")
                        print(f"      last_rel={last_rel_name} [{in_any}]")
                else:
                    print(f"    ✗ GT not reachable in 3 hops from anthem")

        # === Test 4: For Case 241 - reversed anchor (Nijmegen) ===
        bps = case.get('breakpoints', {})
        if bps:
            for step_key, bp_name in bps.items():
                bp_idx = None
                for i, e in enumerate(ents):
                    if e and str(bp_name).lower() in str(e).lower():
                        bp_idx = i; break
                if bp_idx is not None:
                    print(f"\n  --- Free 3-hop from endpoint={bp_name} (idx={bp_idx}) ---")
                    results_bp = free_bfs(bp_idx, fwd, rev, ents, rels, max_depth=3)
                    gt_bp = [(d, n, p) for d, n, p in results_bp if n in gt_indices]
                    if gt_bp:
                        for d, n, p in gt_bp[:5]:
                            print(f"    ✓ d={d}: {fmt_path(p, ents, rels)}")
                    else:
                        print(f"    ✗ GT not reachable in 3 hops from {bp_name}")

        print()
