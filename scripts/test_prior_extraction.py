#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List

import aiohttp


ROOT = Path(__file__).resolve().parents[1]
LLM_API_URL = "http://localhost:8000/v1/chat/completions"
LLM_MODEL = "qwen35-9b-local"
GTE_API_URL = "http://localhost:8003"
DEFAULT_PILOT_RESULTS = ROOT / "reports" / "stage_pipeline_test" / "find_check_plan_pilot_10cases" / "results.json"
DEFAULT_CWQ_PKL = Path("/zhaoshu/SubgraphRAG-main/retrieve/data_files/cwq/processed/test.pkl")
DEFAULT_OUTPUT = ROOT / "reports" / "stage_pipeline_test" / "prior_extraction_pilot_10cases"


SYSTEM_PROMPT = """You analyze a question and identify key entities and relation intents.

Goal:
- extract entity priors: short noun/entity phrases from the question
- extract relation priors: short predicate/verb phrases describing relation intents

Rules:
- entity_priors: short noun/entity phrases (explicit names, quoted titles, bridge entities if obvious)
- relation_priors: short predicate/verb phrases (answer-bearing, check/filter)
- preserve explicit entities from the question
- do NOT compose paths, output actions, or choose anchor
- output JSON only

Output JSON schema:
{
  "entity_priors": ["entity1", "entity2", ...],
  "relation_priors": ["relation intent 1", "relation intent 2", ...]
}

Examples:

Question: "Lou Seal is the mascot for the team that last won the World Series when?"
Output: {"entity_priors": ["Lou Seal", "World Series", "team with mascot Lou Seal"], "relation_priors": ["mascot of team", "team championships", "latest championship year"]}

Question: "Which man is the leader of the country that uses Libya, Libya, Libya as its national anthem?"
Output: {"entity_priors": ["Libya, Libya, Libya", "country using this anthem", "leader of that country"], "relation_priors": ["anthem to country", "country to leader", "office holder"]}

Question: "In which countries do the people speak Portuguese, where the child labor percentage was once 1.8?"
Output: {"entity_priors": ["Portuguese", "countries speaking Portuguese", "countries with child labor percentage 1.8"], "relation_priors": ["language spoken in country", "country child labor percentage", "filter by 1.8"]}
"""


def is_cvt_like(name: str) -> bool:
    return bool(re.match(r"^[mg]\.[A-Za-z0-9_]+$", name))


def rel_to_text(rel: str) -> str:
    parts = rel.split(".")
    if len(parts) >= 2:
        return " ".join(p.replace("_", " ") for p in parts[-2:])
    return rel.replace("_", " ")


async def call_llm(session: aiohttp.ClientSession, question: str) -> Dict[str, Any]:
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"},
        ],
        "max_tokens": 300,
        "temperature": 0.0,
        "top_p": 0.8,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with session.post(LLM_API_URL, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
        data = await resp.json()
    raw = data["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"raw_output": raw, "entity_priors": [], "relation_priors": []}
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"raw_output": raw, "entity_priors": [], "relation_priors": []}
    parsed["raw_output"] = raw
    parsed["entity_priors"] = parsed.get("entity_priors", [])[:5]
    parsed["relation_priors"] = parsed.get("relation_priors", [])[:5]
    return parsed


async def gte_retrieve(
    session: aiohttp.ClientSession,
    query: str,
    candidates: List[str],
    candidate_texts: List[str] | None = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    payload = {
        "query": query,
        "candidates": candidates,
        "candidate_texts": candidate_texts,
        "top_k": top_k,
    }
    async with session.post(
        f"{GTE_API_URL}/retrieve",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=30),
    ) as resp:
        data = await resp.json()
    return data.get("results", [])


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def candidate_hit(cands: List[str], targets: List[str]) -> bool:
    norm_cands = [normalize(c) for c in cands]
    for t in targets:
        nt = normalize(t)
        for c in norm_cands:
            if c == nt or nt in c or c in nt:
                return True
    return False


async def run_case(
    session: aiohttp.ClientSession,
    sample: Dict[str, Any],
    pilot_row: Dict[str, Any],
) -> Dict[str, Any]:
    question = pilot_row["question"]
    llm = await call_llm(session, question)

    # Prepare entity candidates
    all_entities = []
    seen = set()
    for e in sample.get("text_entity_list", []) + sample.get("non_text_entity_list", []):
        if not e or is_cvt_like(e) or len(e) <= 1:
            continue
        if e not in seen:
            all_entities.append(e)
            seen.add(e)

    # Prepare relation candidates
    all_relations = list(sample.get("relation_list", []))
    relation_texts = [f"{r} ; {rel_to_text(r)}" for r in all_relations]

    # GTE expansion for entity priors
    entity_results = []
    for prior in llm.get("entity_priors", []):
        rows = await gte_retrieve(session, prior, all_entities, top_k=10)
        entity_results.append({
            "prior": prior,
            "top_candidates": [r["candidate"] for r in rows],
        })

    # GTE expansion for relation priors
    relation_results = []
    for prior in llm.get("relation_priors", []):
        rows = await gte_retrieve(session, prior, all_relations, candidate_texts=relation_texts, top_k=10)
        relation_results.append({
            "prior": prior,
            "top_candidates": [r["candidate"] for r in rows],
        })

    # Extract expected values from pilot row
    gt_answers = pilot_row.get("gt", [])
    anchor = pilot_row.get("anchor")
    proposed_related = pilot_row.get("parsed", {}).get("proposed_related", [])
    proposed_maybe_related = pilot_row.get("parsed", {}).get("proposed_maybe_related", [])

    # Hit checking
    anchor_hit = any(candidate_hit(item["top_candidates"], [anchor]) for item in entity_results) if anchor else False
    answer_hit = any(candidate_hit(item["top_candidates"], gt_answers) for item in entity_results)
    answer_relation_hit = any(candidate_hit(item["top_candidates"], proposed_related) for item in relation_results)
    filter_relation_hit = any(candidate_hit(item["top_candidates"], proposed_maybe_related) for item in relation_results)

    return {
        "case_id": pilot_row["case_id"],
        "question": question,
        "gt_answers": gt_answers,
        "expected_anchor": anchor,
        "expected_answer_relations": proposed_related,
        "expected_filter_relations": proposed_maybe_related,
        "llm_raw_output": llm.get("raw_output", ""),
        "entity_priors": llm.get("entity_priors", []),
        "relation_priors": llm.get("relation_priors", []),
        "entity_results": entity_results,
        "relation_results": relation_results,
        "anchor_hit": anchor_hit,
        "answer_hit": answer_hit,
        "answer_relation_hit": answer_relation_hit,
        "filter_relation_hit": filter_relation_hit,
    }


def write_report(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = []

    # Aggregate statistics
    cases = len(rows)
    anchor_hits = sum(1 for r in rows if r["anchor_hit"])
    answer_hits = sum(1 for r in rows if r["answer_hit"])
    answer_relation_hits = sum(1 for r in rows if r["answer_relation_hit"])
    filter_relation_hits = sum(1 for r in rows if r["filter_relation_hit"])

    lines.append("# Entity/Relation Prior Extraction Test")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total cases: {cases}")
    lines.append(f"- Anchor recall: {anchor_hits}/{cases} ({100*anchor_hits/cases if cases else 0:.1f}%)")
    lines.append(f"- Answer recall: {answer_hits}/{cases} ({100*answer_hits/cases if cases else 0:.1f}%)")
    lines.append(f"- Answer relation recall: {answer_relation_hits}/{cases} ({100*answer_relation_hits/cases if cases else 0:.1f}%)")
    lines.append(f"- Filter relation recall: {filter_relation_hits}/{cases} ({100*filter_relation_hits/cases if cases else 0:.1f}%)")
    lines.append("")

    # Per-case details
    lines.append("## Per-Case Results")
    lines.append("")

    for row in rows:
        lines.append(f"### {row['case_id']}")
        lines.append(f"**Question:** {row['question']}")
        lines.append("")
        lines.append(f"- GT answers: {row['gt_answers']}")
        lines.append(f"- Expected anchor: {row['expected_anchor']}")
        lines.append(f"- Expected answer relations: {row['expected_answer_relations']}")
        lines.append(f"- Expected filter relations: {row['expected_filter_relations']}")
        lines.append("")
        lines.append("**Hit results:**")
        lines.append(f"- Anchor hit: {row['anchor_hit']}")
        lines.append(f"- Answer hit: {row['answer_hit']}")
        lines.append(f"- Answer relation hit: {row['answer_relation_hit']}")
        lines.append(f"- Filter relation hit: {row['filter_relation_hit']}")
        lines.append("")

        lines.append("**Entity priors:**")
        for item in row["entity_results"]:
            lines.append(f"- `{item['prior']}`")
            lines.append(f"  - Top: {item['top_candidates'][:5]}")
        lines.append("")

        lines.append("**Relation priors:**")
        for item in row["relation_results"]:
            lines.append(f"- `{item['prior']}`")
            lines.append(f"  - Top: {item['top_candidates'][:5]}")
        lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


async def amain(args: argparse.Namespace) -> None:
    pilot_rows = json.loads(Path(args.pilot_results).read_text())
    with Path(args.cwq_pkl).open("rb") as f:
        samples = pickle.load(f)
    sample_map = {s["id"]: s for s in samples if "id" in s}

    rows = []
    async with aiohttp.ClientSession() as session:
        for pilot_row in pilot_rows[: args.limit]:
            sample = sample_map[pilot_row["case_id"]]
            rows.append(await run_case(session, sample, pilot_row))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write results.json
    (out_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write summary.json
    summary = {
        "cases": len(rows),
        "anchor_recall": sum(1 for r in rows if r["anchor_hit"]),
        "answer_recall": sum(1 for r in rows if r["answer_hit"]),
        "answer_relation_recall": sum(1 for r in rows if r["answer_relation_hit"]),
        "filter_relation_recall": sum(1 for r in rows if r["filter_relation_hit"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write report.md
    write_report(out_dir, rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-results", default=str(DEFAULT_PILOT_RESULTS))
    parser.add_argument("--cwq-pkl", default=str(DEFAULT_CWQ_PKL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
