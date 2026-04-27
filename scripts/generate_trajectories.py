#!/usr/bin/env python3
"""Generate trajectory MD files for failing cases from stage pipeline results."""

import argparse, json, os
from pathlib import Path


def generate_trajectory(r, case_num, pilot_row=None):
    """Generate trajectory markdown for a single case result."""
    L = []
    W = L.append

    case_id = r.get('case_id', '?')
    question = r.get('question', '?')
    gt = r.get('gt_answers', [])
    gt_hit = r.get('gt_hit', False)
    llm_hit = r.get('llm_hit', False)
    llm_answer = r.get('llm_answer', None)
    status = "HIT" if gt_hit else "MISS"
    llm_status = "HIT" if llm_hit else "MISS"

    W(f"## CASE {case_num}: {case_id}")
    W(f"**Question**: {question}")
    W(f"**GT Answers**: {gt}")
    if pilot_row:
        a_ent = pilot_row.get('a_entity', [])
        if a_ent is not None:
            W(f"**GT Entities (a_entity)**: {a_ent[:10]}{'...' if a_ent and len(a_ent) > 10 else ''}")
    W(f"**Result**: GT={status} | LLM={llm_status} | LLM_Answer={llm_answer}")

    times = r.get('stage_times', {})
    if times:
        parts = [f"{k}:{v:.1f}s" if isinstance(v, (int, float)) else f"{k}:{v}" for k, v in times.items()]
        W(f"**Times**: {' | '.join(parts)}")
    W("")

    error = r.get('error')
    if error:
        W(f"**ERROR**: {error}")
        W("")

    # ── Stage 0: NER ──
    W("### Stage 0: NER Entity Resolution")
    erd = r.get('entity_retrieval_details', [])
    for detail in erd:
        if isinstance(detail, dict):
            role = detail.get('role', '')
            if 'ner' in role:
                top_ents = detail.get('ner_top_ents', [])
                selected = detail.get('selected', '?')
                sel_idx = detail.get('selected_idx', '?')
                if top_ents:
                    W("  Top entities (GTE scored):")
                    for name, score in top_ents[:8]:
                        marker = " <-- SELECTED" if name == selected else ""
                        W(f"    - {name} (gte={score:.4f}){marker}")
                W(f"  **Selected**: {selected} (idx={sel_idx})")
    anchor_idx = r.get('anchor_idx', '?')
    anchor_name = r.get('anchor_name', '?')
    W(f"  **Anchor**: {anchor_name} (idx={anchor_idx})")
    W("")

    # ── Stage 1: Decomposition ──
    W("### Stage 1: Decomposition")
    decomp = r.get('decomposition', '')
    if decomp:
        for line in decomp.strip().split('\n'):
            W(f"  | {line}")
    steps = r.get('steps_parsed', [])
    if steps:
        W("")
        W("  Steps Parsed:")
        for i, step in enumerate(steps):
            sq = step.get('sub_question', step.get('question', '?'))
            rq = step.get('relation_query', '?')
            W(f"    **Step {i+1}**: {sq}")
            W(f"      relation_query: {rq}")
    W("")

    # ── Stage 2: Entity Resolution ──
    W("### Stage 2: Entity Resolution")
    breakpoints = r.get('breakpoints', {})
    if breakpoints:
        for bp_name, bp_val in breakpoints.items():
            W(f"  Breakpoint '{bp_name}': {bp_val}")
    else:
        W("  (NER mode)")
    W("")

    # ── Stage 3: GTE Relation Retrieval ──
    W("### Stage 3: GTE Relation Retrieval")
    rrd = r.get('relation_retrieval_details', [])
    for si, step_rr in enumerate(rrd):
        W(f"  **Step {si+1}**:")
        if isinstance(step_rr, dict):
            queries = step_rr.get('queries', [])
            gte_indices = step_rr.get('gte_indices', [])
            for qi, q in enumerate(queries):
                query_text = q.get('query', '')
                label = f"GTE query {qi+1}" + (f" ({query_text})" if query_text else "")
                top_k = q.get('top_k', [])
                if top_k:
                    W(f"    {label}:")
                    for item in top_k[:8]:
                        name = item.get('candidate', item.get('rel_text', '?'))
                        score = item.get('score', 0)
                        rel_idx = item.get('rel_idx', '?')
                        kept = rel_idx in gte_indices if gte_indices else False
                        marker = " [KEPT]" if kept else ""
                        W(f"      {item.get('rank', '?')}. {name} (score={score:.4f}, idx={rel_idx}){marker}")
            if gte_indices:
                W(f"    GTE indices for LLM: {gte_indices}")
    W("")

    # ── Stage 4: Relation Pruning ──
    W("### Stage 4: Relation Pruning (LLM)")
    prune = r.get('prune_debug', {})
    if prune:
        resp = prune.get('response', '')
        if resp:
            for line in resp.strip().split('\n'):
                W(f"  | {line}")
            W("")
        parsed = prune.get('parsed_yaml', '')
        if parsed:
            W(f"  Parsed: {parsed}")
    step_rels = r.get('step_relations', [])
    if step_rels:
        W("  Final Step Relations:")
        for si, rels in enumerate(step_rels):
            if isinstance(rels, list) and rels:
                W(f"    Step {si+1}: {rels}")
            elif isinstance(rels, list):
                W(f"    Step {si+1}: [] (EMPTY)")
    W("")

    # ── Stage 5: Graph Traversal ──
    W("### Stage 5: Graph Traversal")
    num_paths = r.get('num_paths', 0)
    max_depth = r.get('max_depth', 0)
    W(f"  Paths: {num_paths} | Max depth: {max_depth}")
    W(f"  **GT_HIT: {gt_hit}**")

    answer_cands = r.get('answer_candidates', [])
    if answer_cands:
        W(f"  Answer Candidates ({len(answer_cands)}): {answer_cands[:20]}{'...' if len(answer_cands) > 20 else ''}")
    else:
        W(f"  Answer Candidates: NONE")

    logical_paths = r.get('logical_paths', [])
    if logical_paths:
        W("")
        W("  Logical Paths:")
        for i, lp in enumerate(logical_paths[:10]):
            W(f"    {i+1}. {lp}")

    patterns = r.get('pattern_details', [])
    if patterns:
        W("")
        W(f"  Patterns ({len(patterns)}):")
        for i, p in enumerate(patterns[:5]):
            W(f"    {i+1}. {p}")
    W("")

    # ── Stage 7: Path Selection ──
    W("### Stage 7: Path Selection")
    sel = r.get('selected_paths', [])
    W(f"  Selected path indices: {sel}")
    W("")

    # ── Stage 8: Answer Reasoning ──
    W("### Stage 8: Answer Reasoning")
    reasoning = r.get('llm_reasoning_full', '')
    if reasoning:
        for line in reasoning.strip().split('\n'):
            W(f"  | {line}")
    W("")
    W("---\n")
    return '\n'.join(L)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='reports/stage_pipeline_test/chain_decompose_test/results.json')
    parser.add_argument('--pilot', default='reports/stage_pipeline_test/cwq_50_stage_v7/pilot.json')
    parser.add_argument('--output-dir', default='reports/stage_pipeline_test/cwq_50_stage_v7/trajectories')
    parser.add_argument('--filter', choices=['failed', 'all'], default='failed')
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    # Load pilot for a_entity info
    pilot_map = {}
    if os.path.exists(args.pilot):
        with open(args.pilot) as f:
            pilot = json.load(f)
        if isinstance(pilot, dict) and 'cases' in pilot:
            for c in pilot['cases']:
                pilot_map[c.get('case_id', '')] = c
        elif isinstance(pilot, list):
            for c in pilot:
                pilot_map[c.get('case_id', '')] = c

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases_to_gen = []
    for i, r in enumerate(data):
        if args.filter == 'failed' and r.get('gt_hit', True):
            continue
        cases_to_gen.append((i, r))

    # Generate individual files
    combined_lines = [f"# Failing Case Trajectories ({len(cases_to_gen)} cases)\n\n---\n\n"]
    for idx, (case_num, r) in enumerate(cases_to_gen):
        case_id = r.get('case_id', '?')
        pilot_row = pilot_map.get(case_id)
        traj = generate_trajectory(r, case_num, pilot_row)

        # Individual file
        safe_id = case_id.split('_')[0] if '_' in case_id else case_id
        fpath = output_dir / f"case{case_num}_{safe_id}.md"
        fpath.write_text(traj, encoding='utf-8')

        combined_lines.append(traj)
        print(f"  [{idx+1}/{len(cases_to_gen)}] Case {case_num}: {r.get('question','')[:60]}")

    # Combined file
    combined_path = output_dir / "all_failures.md"
    combined_path.write_text('\n'.join(combined_lines), encoding='utf-8')

    print(f"\nGenerated {len(cases_to_gen)} trajectories in {output_dir}/")
    print(f"  Combined: {combined_path}")


if __name__ == '__main__':
    raise SystemExit(main())
