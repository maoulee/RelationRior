#!/usr/bin/env python3
"""Diagnostic: L2-skip with proper CVT traversal for all weak cases."""
import json, pickle, sys
from pathlib import Path
from collections import defaultdict, deque

ROOT = Path(__file__).resolve().parents[1]
CWQ_TEST = Path("/zhaoshu/SubgraphRAG-main/retrieve/data_files/cwq/processed/test.pkl")
CWQ_TRAIN = Path("/zhaoshu/SubgraphRAG-main/retrieve/data_files/cwq/processed/train.pkl")
V14 = ROOT / "reports/stage_pipeline_test/answer_template_v14/results.json"

def is_cvt(name):
    """Heuristic CVT detection: short m. or g. IDs or generic names."""
    if not name: return False
    n = name.strip()
    if n.startswith('m.') or n.startswith('g.'): return True
    if len(n) < 3: return False
    return False

def load_all():
    with open(CWQ_TEST, "rb") as f: test = pickle.load(f)
    with open(CWQ_TRAIN, "rb") as f: train = pickle.load(f)
    cwq = {s['id']: s for s in test + train}
    with open(V14) as f: v14 = json.load(f)
    return cwq, v14

def bfs_with_cvt(start_idx, rel_set, fwd, rev, ents, rels, max_depth=3):
    """BFS that auto-traverses CVT intermediaries. Returns list of (depth, path) to any node."""
    visited = {start_idx}
    frontier = [(start_idx, [])]
    all_reach = []
    for depth in range(1, max_depth + 1):
        next_frontier = []
        for node, path in frontier:
            # Expand via fwd/rev with rel_set, OR auto-expand CVT nodes (any relation)
            edges = []
            for (neighbor, r) in fwd.get(node, []):
                edges.append((neighbor, r, 'fwd'))
            for (neighbor, r) in rev.get(node, []):
                edges.append((neighbor, r, 'rev'))

            for neighbor, r, direction in edges:
                if neighbor in visited:
                    continue
                # Allow if rel in set OR if current node is CVT (free traversal)
                current_is_cvt = is_cvt(ents[node]) if node < len(ents) else False
                rel_allowed = r in rel_set

                if rel_allowed or current_is_cvt:
                    visited.add(neighbor)
                    rel_name = rels[r] if r < len(rels) else f"rel{r}"
                    if direction == 'rev':
                        rel_name = f"←{rel_name}"
                    new_path = path + [(node, rel_name, neighbor)]
                    next_frontier.append((neighbor, new_path))
                    all_reach.append((depth, neighbor, new_path))
        frontier = next_frontier
    return all_reach

def main():
    cwq, v14 = load_all()

    # Weak cases
    weak = [c for c in v14 if not c.get('gt_hit', True) or not c.get('llm_hit', True)]
    print(f"=== L2-Skip Diagnostic with CVT ({len(weak)} weak cases) ===\n")

    for case in weak:
        qid = case['case_id']
        sample = cwq.get(qid)
        if not sample:
            print(f"{qid}: not found\n"); continue

        ents = sample.get('text_entity_list', []) + sample.get('non_text_entity_list', [])
        rels = list(sample.get('relation_list', []))
        h_ids, r_ids, t_ids = sample['h_id_list'], sample['r_id_list'], sample['t_id_list']

        # Build adjacency
        fwd = defaultdict(list)
        rev = defaultdict(list)
        for i in range(len(h_ids)):
            fwd[h_ids[i]].append((t_ids[i], r_ids[i]))
            rev[t_ids[i]].append((h_ids[i], r_ids[i]))

        anchor_idx = case['anchor_idx']
        anchor_name = case['anchor_name']
        sr = case['step_relations']
        bps = case.get('breakpoints', {})
        gt = case['gt_answers']
        gt_hit = case.get('gt_hit', False)
        llm_hit = case.get('llm_hit', False)

        # Find GT indices
        gt_indices = set()
        for g in (gt if isinstance(gt, list) else [gt]):
            for i, e in enumerate(ents):
                if e and str(g).lower() == str(e).lower():
                    gt_indices.add(i)

        # Find nearby_airports rel
        nearby_idx = None
        for i, r in enumerate(rels):
            if 'nearby_airports' in r:
                nearby_idx = i

        print(f"{'='*70}")
        print(f"Case: {qid}")
        print(f"  Q: {case['question']}")
        print(f"  GT: {gt} (indices: {gt_indices}) | GT_HIT={gt_hit} LLM_HIT={llm_hit}")
        print(f"  Anchor: {anchor_name} idx={anchor_idx}")
        for li, s in enumerate(sr):
            print(f"  L{li+1} rels: {[f'{r}:{rels[r]}' for r in s]}")
        if nearby_idx is not None:
            print(f"  + nearby_airports idx={nearby_idx}")

        # Collect L2 rels
        l2_rels = set(sr[1]) if len(sr) > 1 else set()
        if nearby_idx is not None:
            l2_plus = l2_rels | {nearby_idx}
        else:
            l2_plus = l2_rels

        # All rels from all layers
        all_rels = set()
        for s in sr:
            all_rels.update(s)

        def find_gt_paths(start_idx, rel_set, label, max_d=3):
            print(f"\n  --- {label} ---")
            results = bfs_with_cvt(start_idx, rel_set, fwd, rev, ents, rels, max_depth=max_d)
            gt_paths = [(d, n, p) for d, n, p in results if n in gt_indices]
            if gt_paths:
                for d, n, p in gt_paths[:5]:
                    path_str = " → ".join([
                        f"{ents[s][:30] if s < len(ents) else s}[{r}]" for s, r, t in p
                    ] + [ents[n][:30] if n < len(ents) else n])
                    print(f"    ✓ d={d}: {path_str}")
            else:
                # Show what IS reachable
                reachable = [(d, n, p) for d, n, p in results if not is_cvt(ents[n] if n < len(ents) else "")]
                print(f"    ✗ No GT path. {len(reachable)} non-CVT nodes reachable:")
                for d, n, p in reachable[:8]:
                    print(f"      d={d}: ... → {ents[n][:50] if n < len(ents) else n}")

        # Test A: L2 only with CVT traversal
        find_gt_paths(anchor_idx, l2_rels, f"A: L2 only, anchor={anchor_name}")

        # Test B: L2 + nearby_airports
        if l2_plus != l2_rels:
            find_gt_paths(anchor_idx, l2_plus, f"B: L2+nearby_airports, anchor={anchor_name}")

        # Test C: All layers combined (what v14 should have done)
        find_gt_paths(anchor_idx, all_rels, f"C: ALL layers, anchor={anchor_name}")

        # Test D: Reversed anchor — use endpoint/breakpoint
        if bps:
            for step_key, bp_name in bps.items():
                bp_idx = None
                for i, e in enumerate(ents):
                    if e and str(bp_name).lower() in str(e).lower():
                        bp_idx = i; break
                if bp_idx is not None:
                    find_gt_paths(bp_idx, all_rels | l2_plus, f"D: Reversed anchor={bp_name} (idx={bp_idx})")

        # Test E: For Case 590 - try correct anchor (anthem)
        if '590' in qid:
            anthem_idx = None
            for i, e in enumerate(ents):
                if e and 'libya, libya, libya' in str(e).lower():
                    anthem_idx = i; break
            if anthem_idx is not None:
                find_gt_paths(anthem_idx, all_rels, f"E: Correct anchor=Libya anthem (idx={anthem_idx})")

        # Test E: For Case 962 - try with L1+L2 combined (should be 3hop)
        if '962' in qid:
            find_gt_paths(anchor_idx, all_rels, f"E: All rels, max_depth=4, anchor={anchor_name}", max_d=4)

        print()


if __name__ == "__main__":
    main()
