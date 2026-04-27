#!/usr/bin/env python3
"""Diagnostic: skip L1 relations, use only L2 (+nearby_airports) for all weak cases."""
import json, pickle, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
CWQ_TEST = Path("/zhaoshu/SubgraphRAG-main/retrieve/data_files/cwq/processed/test.pkl")
CWQ_TRAIN = Path("/zhaoshu/SubgraphRAG-main/retrieve/data_files/cwq/processed/train.pkl")
V14_RESULTS = ROOT / "reports/stage_pipeline_test/answer_template_v14/results.json"

def rel_to_text(r):
    return r.replace(".", " ").replace("_", " ")

def load_case(case_id):
    """Load CWQ sample and v14 result for a case."""
    for pkl in [CWQ_TEST, CWQ_TRAIN]:
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        for s in data:
            cid = s.get("case_id", "")
            if cid == case_id or cid.startswith(case_id.split("_")[0]):
                # Check exact match
                pass
        # Try by index
    # Load v14 results to get case info
    with open(V14_RESULTS) as f:
        v14 = json.load(f)
    return v14

def main():
    # Load data
    with open(CWQ_TEST, "rb") as f:
        test_data = pickle.load(f)
    with open(CWQ_TRAIN, "rb") as f:
        train_data = pickle.load(f)
    with open(V14_RESULTS) as f:
        v14 = json.load(f)

    # Build index by 'id' field
    all_cwq = {}
    for s in test_data + train_data:
        cid = s.get("id", "")
        if cid:
            all_cwq[cid] = s

    # Weak cases: GT_MISS or LLM_MISS
    weak_cases = []
    for case in v14:
        if not case.get("gt_hit", True) or not case.get("llm_hit", True):
            weak_cases.append(case)

    print(f"=== L2-Skip Diagnostic: {len(weak_cases)} weak cases ===\n")

    for case in weak_cases:
        qid = case["case_id"]
        gt = case["gt_answers"]
        anchor_name = case.get("anchor_name", "?")
        anchor_idx = case.get("anchor_idx", None)
        sr = case.get("step_relations", [])
        bps = case.get("breakpoints", {})
        n_paths = case.get("num_paths", 0)
        gt_hit = case.get("gt_hit", False)
        llm_hit = case.get("llm_hit", False)

        # Find CWQ sample by exact case_id match
        sample = all_cwq.get(qid)
        if sample is None:
            print(f"{qid}: CWQ sample not found\n")
            continue

        ents = sample.get("text_entity_list", []) + sample.get("non_text_entity_list", [])
        rels = list(sample.get("relation_list", []))
        h_ids = sample.get("h_id_list", [])
        r_ids = sample.get("r_id_list", [])
        t_ids = sample.get("t_id_list", [])

        # Build rel_idx -> name map
        rel_name_map = {i: r for i, r in enumerate(rels)}

        # Find nearby_airports rel idx
        nearby_airport_idx = None
        for i, r in enumerate(rels):
            if "nearby_airports" in r:
                nearby_airport_idx = i
                break

        # Find GT entity indices
        gt_indices = set()
        for g in (gt if isinstance(gt, list) else [gt]):
            g_str = str(g)
            for i, e in enumerate(ents):
                if e and g_str.lower() in str(e).lower():
                    gt_indices.add(i)

        print(f"{'='*70}")
        print(f"Case: {qid}")
        print(f"  Question: {case['question']}")
        print(f"  GT: {gt} | GT_HIT={gt_hit} | LLM_HIT={llm_hit} | paths={n_paths}")
        print(f"  Anchor: {anchor_name} (idx={anchor_idx})")
        print(f"  L1 rels: {[f'{r}:{rels[r]}' for r in sr[0]]}" if len(sr) > 0 else "  L1: none")
        print(f"  L2 rels: {[f'{r}:{rels[r]}' for r in sr[1]]}" if len(sr) > 1 else "  L2: none")
        if nearby_airport_idx is not None:
            print(f"  + nearby_airports idx={nearby_airport_idx}")
        print(f"  GT entity indices: {gt_indices}")

        # Build adjacency for traversal
        # Forward: h -> (t, r)
        fwd = defaultdict(list)
        rev = defaultdict(list)
        for i in range(len(h_ids)):
            fwd[h_ids[i]].append((t_ids[i], r_ids[i]))
            rev[t_ids[i]].append((h_ids[i], r_ids[i]))

        # Test 1: Skip L1, use L2 only
        l2_rels = set(sr[1]) if len(sr) > 1 else set()
        if nearby_airport_idx is not None:
            l2_rels_plus = l2_rels | {nearby_airport_idx}
        else:
            l2_rels_plus = l2_rels

        def bfs_reachable(start, rel_set, max_depth=3):
            """BFS from start using only rel_set relations, return (depth, path) to GT nodes."""
            visited = {start}
            frontier = [(start, [])]
            results = []
            for depth in range(1, max_depth + 1):
                next_frontier = []
                for node, path in frontier:
                    for (neighbor, r) in fwd.get(node, []):
                        if r in rel_set and neighbor not in visited:
                            visited.add(neighbor)
                            new_path = path + [(node, rels[r], neighbor)]
                            if neighbor in gt_indices:
                                results.append((depth, new_path))
                            next_frontier.append((neighbor, new_path))
                    for (neighbor, r) in rev.get(node, []):
                        if r in rel_set and neighbor not in visited:
                            visited.add(neighbor)
                            new_path = path + [(node, f"←{rels[r]}", neighbor)]
                            if neighbor in gt_indices:
                                results.append((depth, new_path))
                            next_frontier.append((neighbor, new_path))
                frontier = next_frontier
            return results

        def bfs_to_target(start, target_idx, rel_set, max_depth=3):
            """BFS from start to specific target."""
            visited = {start}
            frontier = [(start, [])]
            for depth in range(1, max_depth + 1):
                next_frontier = []
                for node, path in frontier:
                    for (neighbor, r) in fwd.get(node, []):
                        if r in rel_set and neighbor not in visited:
                            visited.add(neighbor)
                            new_path = path + [(node, rels[r], neighbor)]
                            if neighbor == target_idx:
                                return depth, new_path
                            next_frontier.append((neighbor, new_path))
                    for (neighbor, r) in rev.get(node, []):
                        if r in rel_set and neighbor not in visited:
                            visited.add(neighbor)
                            new_path = path + [(node, f"←{rels[r]}", neighbor)]
                            if neighbor == target_idx:
                                return depth, new_path
                            next_frontier.append((neighbor, new_path))
                frontier = next_frontier
            return None, []

        # Get endpoint/breakpoint indices
        bp_indices = {}
        for step_key, bp_name in bps.items():
            for i, e in enumerate(ents):
                if e and bp_name.lower() in str(e).lower():
                    bp_indices[step_key] = i
                    break

        print(f"\n  --- Test A: L2 only (skip L1), anchor={anchor_name} ---")
        results_a = bfs_reachable(anchor_idx, l2_rels, max_depth=3)
        if results_a:
            for d, p in results_a[:3]:
                path_str = " → ".join([f"{ents[n] if n < len(ents) else n}[{rel}]" for n, rel, m in p])
                print(f"    d={d}: {path_str} → GT")
        else:
            print(f"    No GT path found within 3hop")

        print(f"\n  --- Test B: L2 + nearby_airports, anchor={anchor_name} ---")
        results_b = bfs_reachable(anchor_idx, l2_rels_plus, max_depth=3)
        if results_b:
            for d, p in results_b[:3]:
                path_str = " → ".join([f"{ents[n] if n < len(ents) else n}[{rel}]" for n, rel, m in p])
                print(f"    d={d}: {path_str} → GT")
        else:
            print(f"    No GT path found within 3hop")

        # Test C: reversed anchor — use breakpoint/endpoint as anchor
        if bp_indices:
            for step_key, bp_idx in bp_indices.items():
                bp_name = bps[step_key]
                print(f"\n  --- Test C: Reversed anchor={bp_name} (idx={bp_idx}), L2 rels ---")
                # Use all relations from all layers for reversed
                all_rels = set()
                for s in sr:
                    all_rels.update(s)
                results_c = bfs_reachable(bp_idx, all_rels, max_depth=3)
                if results_c:
                    for d, p in results_c[:3]:
                        path_str = " → ".join([f"{ents[n] if n < len(ents) else n}[{rel}]" for n, rel, m in p])
                        print(f"    d={d}: {path_str} → GT")
                else:
                    print(f"    No GT path found within 3hop")

                # Also try with anchor_idx as target
                print(f"\n  --- Test D: Reversed anchor={bp_name}, targeting {anchor_name} ---")
                d, p = bfs_to_target(bp_idx, anchor_idx, all_rels | l2_rels_plus, max_depth=3)
                if p:
                    path_str = " → ".join([f"{ents[n] if n < len(ents) else n}[{rel}]" for n, rel, m in p])
                    print(f"    d={d}: {path_str} → {anchor_name}")

        # Test E: All spatial relations from anchor (for location cases)
        spatial_keywords = ["border", "adjoin", "contain", "nearby", "airport", "country", "administrative"]
        spatial_rels = set()
        for i, r in enumerate(rels):
            if any(kw in r.lower() for kw in spatial_keywords):
                spatial_rels.add(i)

        if spatial_rels:
            print(f"\n  --- Test E: All spatial rels ({len(spatial_rels)} rels), anchor={anchor_name} ---")
            results_e = bfs_reachable(anchor_idx, spatial_rels, max_depth=3)
            if results_e:
                for d, p in results_e[:3]:
                    path_str = " → ".join([f"{ents[n] if n < len(ents) else n}[{rel}]" for n, rel, m in p])
                    print(f"    d={d}: {path_str} → GT")
            else:
                print(f"    No GT path found within 3hop")

        print()


if __name__ == "__main__":
    main()
