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
DEFAULT_OUTPUT = ROOT / "reports" / "stage_pipeline_test" / "llm_query_gte_10cases"


SYSTEM_PROMPT = """You analyze a question and propose short retrieval commands.

Goal:
- suggest a few entity-retrieval commands that may surface key anchor or bridge entities
- suggest a few relation-retrieval commands that may surface the answer-bearing relation

Rules:
- commands must be short natural phrases
- prefer the form:
  - find entities about ...
  - find relations about ...
- include bridge entities if they are likely useful
- include answer-type hints if useful
- do not explain, output JSON only

Output JSON schema:
{
  "entity_queries": ["find entities about ...", "..."],
  "relation_queries": ["find relations about ...", "..."]
}
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
        return {"raw_output": raw, "entity_queries": [], "relation_queries": []}
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"raw_output": raw, "entity_queries": [], "relation_queries": []}
    parsed["raw_output"] = raw
    parsed["entity_queries"] = parsed.get("entity_queries", [])[:4]
    parsed["relation_queries"] = parsed.get("relation_queries", [])[:4]
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

    all_entities = []
    seen = set()
    for e in sample.get("text_entity_list", []) + sample.get("non_text_entity_list", []):
        if not e or is_cvt_like(e) or len(e) <= 1:
            continue
        if e not in seen:
            all_entities.append(e)
            seen.add(e)

    all_relations = list(sample.get("relation_list", []))
    relation_texts = [f"{r} ; {rel_to_text(r)}" for r in all_relations]

    entity_results = []
    for q in llm.get("entity_queries", []):
        rows = await gte_retrieve(session, q, all_entities, top_k=10)
        entity_results.append({
            "query": q,
            "top_candidates": [r["candidate"] for r in rows],
        })

    relation_results = []
    for q in llm.get("relation_queries", []):
        rows = await gte_retrieve(session, q, all_relations, candidate_texts=relation_texts, top_k=10)
        relation_results.append({
            "query": q,
            "top_candidates": [r["candidate"] for r in rows],
        })

    gt_answers = pilot_row.get("gt", [])
    anchor = pilot_row.get("anchor")
    proposed_related = pilot_row.get("parsed", {}).get("proposed_related", [])

    entity_hit_answer = any(candidate_hit(item["top_candidates"], gt_answers) for item in entity_results)
    entity_hit_anchor = any(candidate_hit(item["top_candidates"], [anchor]) for item in entity_results) if anchor else False
    relation_hit_proposed = any(candidate_hit(item["top_candidates"], proposed_related) for item in relation_results)

    return {
        "case_id": pilot_row["case_id"],
        "question": question,
        "gt_answers": gt_answers,
        "expected_anchor": anchor,
        "expected_relations_proxy": proposed_related,
        "llm_raw_output": llm.get("raw_output", ""),
        "entity_queries": llm.get("entity_queries", []),
        "relation_queries": llm.get("relation_queries", []),
        "entity_results": entity_results,
        "relation_results": relation_results,
        "entity_hit_answer": entity_hit_answer,
        "entity_hit_anchor": entity_hit_anchor,
        "relation_hit_proxy": relation_hit_proposed,
    }


def write_report(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    answer_hits = sum(1 for r in rows if r["entity_hit_answer"])
    anchor_hits = sum(1 for r in rows if r["entity_hit_anchor"])
    relation_hits = sum(1 for r in rows if r["relation_hit_proxy"])

    lines.append("# LLM Query -> GTE Diagnostic")
    lines.append("")
    lines.append(f"- cases: {len(rows)}")
    lines.append(f"- answer entity recalled by any entity query: {answer_hits}/{len(rows)}")
    lines.append(f"- anchor recalled by any entity query: {anchor_hits}/{len(rows)}")
    lines.append(f"- proxy relation recalled by any relation query: {relation_hits}/{len(rows)}")
    lines.append("")

    for row in rows:
        lines.append(f"## {row['case_id']}")
        lines.append(f"- question: {row['question']}")
        lines.append(f"- gt_answers: {row['gt_answers']}")
        lines.append(f"- expected_anchor: {row['expected_anchor']}")
        lines.append(f"- expected_relations_proxy: {row['expected_relations_proxy']}")
        lines.append(f"- entity_hit_answer: {row['entity_hit_answer']}")
        lines.append(f"- entity_hit_anchor: {row['entity_hit_anchor']}")
        lines.append(f"- relation_hit_proxy: {row['relation_hit_proxy']}")
        lines.append("- entity queries:")
        for item in row["entity_results"]:
            lines.append(f"  - {item['query']}")
            lines.append(f"    - top: {item['top_candidates'][:5]}")
        lines.append("- relation queries:")
        for item in row["relation_results"]:
            lines.append(f"  - {item['query']}")
            lines.append(f"    - top: {item['top_candidates'][:5]}")
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
    (out_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "cases": len(rows),
        "answer_entity_recall_any_query": sum(1 for r in rows if r["entity_hit_answer"]),
        "anchor_recall_any_query": sum(1 for r in rows if r["entity_hit_anchor"]),
        "relation_proxy_recall_any_query": sum(1 for r in rows if r["relation_hit_proxy"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
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
