#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Set

import aiohttp


ROOT = Path(__file__).resolve().parents[1]
LLM_API_URL = "http://localhost:8000/v1/chat/completions"
LLM_MODEL = "qwen35-9b-local"
GTE_API_URL = "http://localhost:8003"
DEFAULT_PILOT_RESULTS = ROOT / "reports" / "stage_pipeline_test" / "find_check_plan_pilot_10cases" / "results.json"
DEFAULT_CWQ_PKL = Path("/zhaoshu/SubgraphRAG-main/retrieve/data_files/cwq/processed/test.pkl")
DEFAULT_OUTPUT = ROOT / "reports" / "stage_pipeline_test" / "entity_route_decompose_10cases_v22"


def is_cvt_like(name: str) -> bool:
    return bool(re.match(r"^[mg]\.[A-Za-z0-9_]+$", name))


async def call_chat(session: aiohttp.ClientSession, system: str, user: str, max_tokens: int = 700) -> str:
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 0.8,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with session.post(LLM_API_URL, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
        data = await resp.json()
    return data["choices"][0]["message"]["content"]


async def gte_retrieve(
    session: aiohttp.ClientSession,
    query: str,
    candidates: List[str],
    candidate_texts: List[str] | None = None,
    top_k: int = 12,
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
        timeout=aiohttp.ClientTimeout(total=60),
    ) as resp:
        data = await resp.json()
    return data.get("results", [])


ENDPOINT_DETECT_PROMPT = """DEPRECATED — not used."""


DECOMP_UNIFIED_PROMPT = """Decompose the question into short simple phrases that describe each relation lookup needed.

Each phrase should be a simple noun-phrase describing one graph relation to look up. Do NOT invent answers. Only describe what to search for.

Example:
Question: Which river flows through the capital of Australia?
Decomposition:
- "capital of Australia"
- "river flowing through that city"

Example:
Question: What genre of music is the artist who sang Imagine known for?
Decomposition:
- "artist who sang Imagine"
- "music genre of that artist"

Rules:
- Keep each phrase short (3-6 words).
- Each phrase describes ONE relation lookup.
- Only use entities mentioned in the question or KG candidates.
- If multiple starting points exist, list phrases for each direction.

Output:
<analysis>
- Named entities in question: ...
- Starting points from KG: ...
</analysis>

Direction A phrases (from [entity]):
- ...

Direction B phrases (from [entity], if exists):
- ...

Recommended: [A or B or only]
Reason: [one sentence]
"""


DECOMP_SINGLE_PROMPT = """You decompose a question into graph exploration steps.

Rules:
- Do not answer the question.
- Do not write WH-questions.
- Write command-style steps only: Find ... / Check ...
- Use the chosen anchor.
- Use only the provided reasoning entities and entities explicitly stated in the question.
- Do not invent new anchor candidates, bridge entities, or pseudo-entities.
- Build the answer path first, then add checks/filters later.
- Preserve any explicit named bridge entity or entity-grounded description.
- Numbers and scalar values are checks, not anchors.
- Keep the analysis short.
- Do not imitate benchmark cases.

Safe synthetic example:
Question: Which city in the north contains a museum that exhibits Artifact X?
Good decomposition:
- Find museums exhibiting Artifact X
- Find cities containing those museums
- Check whether those cities are in the north
Why:
- It preserves the bridge entity \"museum\"
- It finds the answer path before applying the filter

Output format:
<analysis>
- Answer type: ...
- Chosen anchor: ...
- Named bridge entity or description: ... or none
- Main path before constraints: ...
</analysis>

Anchor: [entity]
1. Find ...
2. Find ...
3. Check ...
"""


DECOMP_DUAL_PROMPT = """Decompose a question into graph exploration steps from BOTH endpoints.

The question has two identifiable endpoints. Decompose from each direction, then pick the better one.

Rules:
- Do not answer the question.
- Write command-style steps only: Find ... / Check ...
- Use only entities explicitly stated in the question.
- Do not invent entities.
- Build the answer path first, then add checks/filters.
- Keep analysis short.

Option A: explore from start entity.
Option B: explore from end entity.

Prefer the direction with: less first-step branching, shorter path to answer, preserving bridge entities.

Output format:
<analysis>
- Answer type: ...
- Bridge entity or description: ... or none
- Path A vs B: ...
</analysis>

Option A (from {start}):
1. Find ...
2. Find ...
3. Check ...

Option B (from {end}):
1. Find ...
2. Find ...
3. Check ...

Selection: [A or B]
Reason: [one short sentence]
"""


def extract_json_block(raw: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {}


def extract_xml_tag(raw: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", raw, re.S | re.I)
    return m.group(1).strip() if m else ""


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9%.' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_scalar_like(text: str) -> bool:
    t = text.strip().lower()
    if re.fullmatch(r"\d+(\.\d+)?%?", t):
        return True
    if re.fullmatch(r"\d{4}", t):
        return True
    return False


def literal_entity_candidates(question: str, retrieved_entities: List[str]) -> List[str]:
    qnorm = normalize_text(question)
    chosen: List[str] = []
    for ent in retrieved_entities:
        enorm = normalize_text(ent)
        if not enorm or is_scalar_like(enorm):
            continue
        if enorm in qnorm:
            chosen.append(ent)
    return chosen


def relation_domain(rel: str) -> str:
    parts = rel.split(".")
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return rel


def build_entity_contexts(
    sample: Dict[str, Any],
    target_entities: List[str],
    preferred_domains: Set[str],
    max_per_entity: int = 3,
) -> Dict[str, List[str]]:
    entity_list = sample.get("text_entity_list", []) + sample.get("non_text_entity_list", [])
    rel_list = sample.get("relation_list", [])
    name_to_ids: Dict[str, List[int]] = {}
    for i, name in enumerate(entity_list):
        name_to_ids.setdefault(name, []).append(i)

    contexts: Dict[str, List[str]] = {ent: [] for ent in target_entities}
    fallback: Dict[str, List[str]] = {ent: [] for ent in target_entities}

    for h, r, t in zip(sample.get("h_id_list", []), sample.get("r_id_list", []), sample.get("t_id_list", [])):
        if h >= len(entity_list) or t >= len(entity_list) or r >= len(rel_list):
            continue
        h_name = entity_list[h]
        t_name = entity_list[t]
        rel = rel_list[r]
        dom = relation_domain(rel)

        for ent in target_entities:
            ent_ids = name_to_ids.get(ent, [])
            if h in ent_ids:
                item = f"out: {rel} -> {t_name}"
                if dom in preferred_domains and item not in contexts[ent] and len(contexts[ent]) < max_per_entity:
                    contexts[ent].append(item)
                elif item not in fallback[ent]:
                    fallback[ent].append(item)
            if t in ent_ids:
                item = f"in: {h_name} <- {rel}"
                if dom in preferred_domains and item not in contexts[ent] and len(contexts[ent]) < max_per_entity:
                    contexts[ent].append(item)
                elif item not in fallback[ent]:
                    fallback[ent].append(item)

    for ent in target_entities:
        if len(contexts[ent]) < max_per_entity:
            for item in fallback[ent]:
                if item not in contexts[ent]:
                    contexts[ent].append(item)
                if len(contexts[ent]) >= max_per_entity:
                    break
    return contexts


async def run_case(session: aiohttp.ClientSession, sample: Dict[str, Any], pilot_row: Dict[str, Any]) -> Dict[str, Any]:
    question = pilot_row["question"]
    all_entities = []
    seen = set()
    for e in sample.get("text_entity_list", []) + sample.get("non_text_entity_list", []):
        if not e or is_cvt_like(e) or len(e) <= 1:
            continue
        if e not in seen:
            all_entities.append(e)
            seen.add(e)

    gte_rows = await gte_retrieve(session, question, all_entities, top_k=12)
    retrieved_entities = [r["candidate"] for r in gte_rows if r.get("candidate")]
    rel_rows = await gte_retrieve(session, question, sample.get("relation_list", []), top_k=5)
    top_relations = [r["candidate"] for r in rel_rows if r.get("candidate")]
    top_relation_domains = {relation_domain(r) for r in top_relations}

    # Step 1: literal candidates (code-level hard filter)
    literal_candidate_entities = literal_entity_candidates(question, retrieved_entities)

    # Build full candidate list for model: all GTE results with graph contexts
    entity_contexts = build_entity_contexts(sample, retrieved_entities[:12], top_relation_domains, max_per_entity=3)

    decomp_raw = ""
    if retrieved_entities:
        context_lines: List[str] = []
        for ent in retrieved_entities[:12]:
            ctx = entity_contexts.get(ent, [])
            context_lines.append(f"- {ent}")
            for item in ctx:
                context_lines.append(f"  {item}")
        decomp_user = (
            f"Question: {question}\n\n"
            "KG candidates:\n" +
            "\n".join(f"- {e}" for e in retrieved_entities[:12]) +
            "\n\nGraph contexts:\n" +
            ("\n".join(context_lines) if context_lines else "- none")
        )
        decomp_raw = await call_chat(session, DECOMP_UNIFIED_PROMPT, decomp_user, max_tokens=900)

    # Extract analysis and recommended direction
    analysis = extract_xml_tag(decomp_raw, "analysis")
    recommended = ""
    rec_m = re.search(r"Recommended:\s*(.+)", decomp_raw)
    if rec_m:
        recommended = rec_m.group(1).strip()

    return {
        "case_id": pilot_row["case_id"],
        "question": question,
        "gt_answers": pilot_row.get("gt", []),
        "gte_entities": retrieved_entities,
        "gte_relations": top_relations,
        "literal_candidate_entities": literal_candidate_entities,
        "entity_contexts": entity_contexts,
        "decomposition_raw": decomp_raw,
        "analysis": analysis,
        "recommended": recommended,
    }


def render_report(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    lines: List[str] = []
    lines.append("# Unified Decomposition Diagnostic")
    lines.append("")
    lines.append(f"- cases: {len(rows)}")
    lines.append("")
    for row in rows:
        lines.append(f"## {row['case_id']}")
        lines.append(f"- question: {row['question']}")
        lines.append(f"- gt_answers: {row['gt_answers']}")
        lines.append(f"- gte_entities: {row['gte_entities']}")
        lines.append(f"- gte_relations: {row.get('gte_relations')}")
        lines.append(f"- literal_candidate_entities: {row.get('literal_candidate_entities')}")
        lines.append(f"- recommended: {row.get('recommended')}")
        if row.get("entity_contexts"):
            lines.append("")
            lines.append("### Entity Contexts")
            lines.append("```text")
            for ent, ctx in row.get("entity_contexts", {}).items():
                if not ctx:
                    continue
                lines.append(ent)
                for item in ctx:
                    lines.append(f"- {item}")
                lines.append("")
            lines.append("```")
        lines.append("")
        lines.append("### Decomposition")
        lines.append("```text")
        lines.append((row.get("decomposition_raw") or "").strip())
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


async def amain(args: argparse.Namespace) -> None:
    pilot_rows = json.loads(Path(args.pilot_results).read_text())
    with Path(args.cwq_pkl).open("rb") as f:
        samples = pickle.load(f)
    sample_map = {s["id"]: s for s in samples if "id" in s}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    async with aiohttp.ClientSession() as session:
        for pilot_row in pilot_rows[: args.limit]:
            sample = sample_map[pilot_row["case_id"]]
            rows.append(await run_case(session, sample, pilot_row))

    (out_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    render_report(rows, out_dir)
    print(out_dir / "report.md")


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
