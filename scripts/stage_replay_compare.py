#!/usr/bin/env python3
"""
Stage-by-stage replay: feed 27B's actual LLM input prompts to 9B,
compare outputs field-by-field to find divergence.

Usage:
  python scripts/stage_replay_compare.py \
    --golden reports/entity_v3_p0fix/results.json \
    --stages 1a,1b,4,8 \
    [--limit 50]
"""
import argparse, asyncio, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_chain_decompose import batch_call_llm, extract_xml_tag


def load_cases(path):
    with open(path) as f:
        return {c["case_id"]: c for c in json.load(f)}


def extract_analysis(text):
    """Parse Stage 1a output: anchor, endpoints, answer_type, rewritten."""
    anchor = ""
    endpoints = ""
    answer_type = ""
    rewritten = ""
    for line in (text or "").split("\n"):
        line = line.strip()
        if line.startswith("Anchor:"):
            anchor = line.split("Anchor:", 1)[1].strip()
        elif line.startswith("Endpoints:"):
            endpoints = line.split("Endpoints:", 1)[1].strip()
        elif line.startswith("Answer type:"):
            answer_type = line.split("Answer type:", 1)[1].strip()
        elif line.startswith("Rewritten:"):
            rewritten = line.split("Rewritten:", 1)[1].strip()
    return anchor, endpoints, answer_type, rewritten


async def replay_stage_1a(session, case_ids, golden):
    """Stage 1a: Entity Analysis — compare anchor/endpoints/answer_type."""
    print("\n" + "=" * 70)
    print("STAGE 1a: ENTITY ANALYSIS REPLAY (50 cases)")
    print("=" * 70)

    prompts, cids = [], []
    for cid in case_ids:
        g = golden[cid]
        p = g.get("stage_1a_prompt", "")
        if p:
            prompts.append([{"role": "user", "content": p}])
            cids.append(cid)

    print(f"  Sending {len(prompts)} prompts to 9B...")
    responses = await batch_call_llm(session, prompts, max_tokens=400)

    anchor_diff = 0
    endpoint_diff = 0
    atype_diff = 0
    total = 0

    for cid, raw in zip(cids, responses):
        g = golden[cid]
        g_anchor = g.get("stage_1a_anchor", "")
        g_endpoints = g.get("stage_1a_endpoints", "")
        g_atype = g.get("stage_1a_answer_type", "")
        g_rewritten = g.get("stage_1a_rewritten", "")

        new_anchor, new_endpoints, new_atype, new_rewritten = extract_analysis(raw)

        a_same = g_anchor.lower().strip() == new_anchor.lower().strip()
        e_same = g_endpoints.lower().strip() == new_endpoints.lower().strip()
        t_same = g_atype.lower().strip() == new_atype.lower().strip()

        if not a_same:
            anchor_diff += 1
        if not e_same:
            endpoint_diff += 1
        if not t_same:
            atype_diff += 1

        total += 1
        if not (a_same and e_same and t_same):
            print(f"  DIFF: {g['question'][:55]}")
            if not a_same:
                print(f"    Anchor:   27B={g_anchor:30s} 9B={new_anchor}")
            if not e_same:
                print(f"    Endpoint: 27B={g_endpoints:30s} 9B={new_endpoints}")
            if not t_same:
                print(f"    Atype:    27B={g_atype:30s} 9B={new_atype}")

    same = total - max(anchor_diff, endpoint_diff)
    print(f"\n  Summary ({total} cases):")
    print(f"    Anchor identical:    {total - anchor_diff}/{total}")
    print(f"    Endpoints identical: {total - endpoint_diff}/{total}")
    print(f"    Answer type ident:   {total - atype_diff}/{total}")


async def replay_stage_1b(session, case_ids, golden):
    """Stage 1b: Chain Decomposition — compare chain structure."""
    print("\n" + "=" * 70)
    print("STAGE 1b: CHAIN DECOMPOSITION REPLAY (50 cases)")
    print("=" * 70)

    prompts, cids = [], []
    for cid in case_ids:
        g = golden[cid]
        p = g.get("decomposition_prompt", "")
        if p:
            prompts.append([{"role": "user", "content": p}])
            cids.append(cid)

    print(f"  Sending {len(prompts)} prompts to 9B...")
    responses = await batch_call_llm(session, prompts, max_tokens=600)

    chain_diff = 0
    reasoning_diff = 0
    step_diff = 0

    for cid, raw in zip(cids, responses):
        g = golden[cid]
        golden_decomp = g.get("decomposition", "")
        new_decomp = raw or ""

        g_chain = (extract_xml_tag(golden_decomp, "chain") or "").strip()
        n_chain = (extract_xml_tag(new_decomp, "chain") or "").strip()
        g_reasoning = (extract_xml_tag(golden_decomp, "reasoning") or "").strip()
        n_reasoning = (extract_xml_tag(new_decomp, "reasoning") or "").strip()

        chain_match = g_chain == n_chain
        reasoning_match = g_reasoning == n_reasoning

        if not chain_match:
            chain_diff += 1
        if not reasoning_match:
            reasoning_diff += 1

        if not chain_match:
            print(f"  CHAIN DIFF: {g['question'][:55]}")
            print(f"    27B: {g_chain[:100]}")
            print(f"    9B:  {n_chain[:100]}")

    total = len(cids)
    print(f"\n  Summary ({total} cases):")
    print(f"    Chain identical:    {total - chain_diff}/{total} ({chain_diff} different)")
    print(f"    Reasoning identical: {total - reasoning_diff}/{total}")


async def replay_stage_4(session, case_ids, golden):
    """Stage 4: Relation Pruning — compare selected relations."""
    print("\n" + "=" * 70)
    print("STAGE 4: RELATION PRUNING REPLAY (50 cases)")
    print("=" * 70)

    prompts, cids = [], []
    for cid in case_ids:
        g = golden[cid]
        debug = g.get("prune_debug", {})
        if isinstance(debug, dict):
            p = debug.get("prompt", "")
        else:
            p = ""
        if p:
            prompts.append([{"role": "user", "content": p}])
            cids.append(cid)

    if not prompts:
        print("  No pruning prompts found. Skipping.")
        return

    print(f"  Sending {len(prompts)} prompts to 9B...")
    responses = await batch_call_llm(session, prompts, max_tokens=600)

    rel_diff = 0
    g_rel_sets = {}
    n_rel_sets = {}
    for cid, raw in zip(cids, responses):
        g = golden[cid]
        g_rels = g.get("step_relations", [])
        n_rels_raw = raw or ""

        # Extract relation names from golden (list of sets of strings)
        g_all_rels = set()
        if g_rels:
            for layer in g_rels:
                if isinstance(layer, (list, set)):
                    g_all_rels.update(str(r).lower().strip() for r in layer)
        g_rel_sets[cid] = g_all_rels

        # Extract relations from 9B output (look for relation patterns)
        n_all_rels = set(re.findall(r'(\w+)\s*(?:relation|edge|link)', n_rels_raw.lower()))
        if not n_all_rels:
            # Fallback: extract words that look like Freebase relations (contain dots)
            n_all_rels = set(re.findall(r'[\w.]+\.[\w.]+', n_rels_raw.lower()))
        n_rel_sets[cid] = n_all_rels

        if g_all_rels and n_all_rels:
            overlap = len(g_all_rels & n_all_rels) / max(len(g_all_rels), 1)
            if overlap < 0.5:
                rel_diff += 1
                print(f"  REL DIFF: {g['question'][:55]}")
                print(f"    27B rels ({len(g_all_rels)}): {list(g_all_rels)[:5]}")
                print(f"    9B  rels ({len(n_all_rels)}): {list(n_all_rels)[:5]}")

    total = len(cids)
    print(f"\n  Summary ({total} cases):")
    print(f"    Relations similar (≥50% overlap): {total - rel_diff}/{total}")
    print(f"    Relations divergent (<50% overlap): {rel_diff}/{total}")


async def replay_stage_8(session, case_ids, golden):
    """Stage 8: Answer Reasoning — compare final answers."""
    print("\n" + "=" * 70)
    print("STAGE 8: ANSWER REASONING REPLAY (50 cases)")
    print("=" * 70)

    prompts, cids = [], []
    for cid in case_ids:
        g = golden[cid]
        p = g.get("llm_reasoning_prompt", "")
        if p:
            prompts.append([{"role": "user", "content": p}])
            cids.append(cid)

    print(f"  Sending {len(prompts)} prompts to 9B...")
    responses = await batch_call_llm(session, prompts, max_tokens=1000)

    answer_diff = 0
    gt_hit_27b = 0
    gt_hit_9b = 0

    for cid, raw in zip(cids, responses):
        g = golden[cid]
        golden_answer = g.get("llm_answer", "")
        new_answer = extract_xml_tag(raw or "", "answer") or ""

        # Extract boxed content
        golden_boxed = [b.strip() for b in re.findall(r"\\boxed\{([^}]+)\}", golden_answer)]
        new_boxed = [b.strip() for b in re.findall(r"\\boxed\{([^}]+)\}", new_answer)]

        gt = g.get("gt_answers", [])
        golden_hit = any(g_.lower() in golden_answer.lower() for g_ in gt) if gt else False
        new_hit = any(g_.lower() in new_answer.lower() for g_ in gt) if gt else False

        if golden_hit:
            gt_hit_27b += 1
        if new_hit:
            gt_hit_9b += 1

        # Compare answer strings
        answer_same = set(golden_boxed) == set(new_boxed)
        if not answer_same:
            answer_diff += 1

        if not answer_same:
            status = "OK(9B)" if new_hit and not golden_hit else ("MISS" if golden_hit and not new_hit else "DIFF")
            print(f"  [{status}] {g['question'][:50]}")
            print(f"    27B: {golden_boxed}")
            print(f"    9B:  {new_boxed}")

    total = len(cids)
    print(f"\n  Summary ({total} cases):")
    print(f"    Answer identical:  {total - answer_diff}/{total}")
    print(f"    GT hit 27B: {gt_hit_27b}/{total}")
    print(f"    GT hit 9B:  {gt_hit_9b}/{total}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", required=True)
    parser.add_argument("--stages", default="1a,1b,4,8")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    golden = load_cases(args.golden)
    case_ids = list(golden.keys())[:args.limit]
    stages = [s.strip() for s in args.stages.split(",")]

    print(f"Replaying {len(case_ids)} cases on stages: {stages}")
    print(f"Golden: {args.golden}")

    import aiohttp
    async with aiohttp.ClientSession() as session:
        for stage in stages:
            if stage == "1a":
                await replay_stage_1a(session, case_ids, golden)
            elif stage == "1b":
                await replay_stage_1b(session, case_ids, golden)
            elif stage == "4":
                await replay_stage_4(session, case_ids, golden)
            elif stage == "8":
                await replay_stage_8(session, case_ids, golden)

    print("\n" + "=" * 70)
    print("DONE — compare each stage's identical/different ratio above")
    print("The stage with the MOST diffs is the bottleneck for 9B.")


if __name__ == "__main__":
    asyncio.run(main())
