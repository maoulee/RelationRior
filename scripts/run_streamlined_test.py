#!/usr/bin/env python3
"""
Streamlined Prompt Pipeline Test — CWQ Dataset

Tests the per-stage short prompt system against CWQ graph backend.
Pipeline: BASE_SYSTEM (<300 chars) + per-stage prompt (<400 chars each).

Usage:
    python scripts/run_streamlined_test.py \
        --data-path data/cwq/cwq_test.jsonl \
        --limit-cases 20 \
        --max-concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.streamlined_prompts import (
    BASE_SYSTEM,
    STAGE1_DISCOVERY,
    STAGE2_PLANNING,
    STAGE3_EXECUTION,
    STAGE4_COLLECTION,
    STAGE5_REASONING,
    BACKTRACK_HINT,
    format_backend_feedback,
    format_candidates_message,
    build_stage5_with_candidates,
)

# Backend config
KG_API_URL = os.getenv("KGQA_KG_API_URL", "http://localhost:8002").rstrip("/")
LLM_API_URL = os.getenv("KGQA_LLM_API_URL", "http://localhost:8000/v1").rstrip("/")
LLM_MODEL = os.getenv("KGQA_MODEL_NAME", "qwen35-9b-local")

TOOL_ENDPOINTS = {
    "check_entities": f"{KG_API_URL}/v2/find_entities",
    "find_entities": f"{KG_API_URL}/v2/find_entities",
    "explore_schema": f"{KG_API_URL}/v2/explore_schema",
    "plan_subquestion": f"{KG_API_URL}/v2/plan_subquestion",
    "plan": f"{KG_API_URL}/v2/plan_subquestion",
    "match_pattern": f"{KG_API_URL}/v2/match_pattern",
    "action": f"{KG_API_URL}/v2/match_pattern",
    "get_neighbors": f"{KG_API_URL}/v2/get_neighbors",
    "filter": f"{KG_API_URL}/v2/filter",
}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default


def _normalize_text(value: str) -> str:
    if not isinstance(value, str):
        value = str(value)
    value = value.lower().strip()
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = "".join(ch for ch in value if ch.isalnum() or ch in " .-_")
    return " ".join(value.split())


def _entity_overlap(pred: List[str], gt: List[str]) -> Tuple[float, float, float]:
    if not pred or not gt:
        return (0.0, 0.0, 0.0)
    pred_set = {_normalize_text(x) for x in pred}
    gt_set = {_normalize_text(x) for x in gt}
    pred_set.discard("")
    gt_set.discard("")
    if not pred_set or not gt_set:
        return (0.0, 0.0, 0.0)
    common = pred_set & gt_set
    precision = len(common) / len(pred_set) if pred_set else 0.0
    recall = len(common) / len(gt_set) if gt_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return (precision, recall, f1)


def _extract_clean_question(user_message: str) -> str:
    if "Question:" in user_message:
        after_question = user_message.split("Question:", 1)[1]
        question = after_question.strip().split("\n")[0].strip()
        return question
    return user_message.strip()


def _extract_domains(user_message: str) -> List[str]:
    """Extract domain list from the user message header."""
    match = re.search(r"Available Domains in Subgraph:\s*\n(.*?)(?:\n\n|\[Retrieval)", user_message, re.S)
    if match:
        domains_text = match.group(1).strip()
        return [d.strip() for d in domains_text.split(",") if d.strip()]
    return []


# ── Stage management ──────────────────────────────────────────────

STAGE_PROMPTS = {
    1: STAGE1_DISCOVERY,
    2: STAGE2_PLANNING,
    3: STAGE3_EXECUTION,
    4: STAGE4_COLLECTION,
    5: STAGE5_REASONING,
}

STAGE_TOOLS = {
    1: {"check_entities", "find_entities", "explore_schema"},
    2: {"plan_subquestion", "plan"},
    3: {"match_pattern", "action"},
    4: set(),  # no tool calls, just extraction
    5: set(),  # no tool calls, just answer
}


def _detect_stage_from_actions(tool_calls: List[Dict]) -> int:
    """Determine which stage the model's tool calls belong to."""
    if not tool_calls:
        return 5  # assume final answer
    tools = {tc.get("tool_name", "") for tc in tool_calls}
    for stage, allowed in STAGE_TOOLS.items():
        if tools & allowed:
            return stage
    return 3  # default to execution


def _is_relation_name(name: str) -> bool:
    """Check if a string looks like a relation (domain.type.rel format) rather than an entity."""
    # Relations have 2+ dots like "film.performance.film" or "government.politician.party"
    parts = name.split(".")
    if len(parts) >= 3:
        return True
    # Also filter m.id patterns like "m.02wrxkz"
    if re.match(r'^m\.[a-z0-9]+$', name):
        return True
    return False


def _filter_answer_entities(entities: List[str]) -> List[str]:
    """Filter out non-entity predictions (relations, None, template text)."""
    filtered = []
    for e in entities:
        e = e.strip()
        if not e or e.lower() in {"none", "entity name", "n/a", "unknown"}:
            continue
        if _is_relation_name(e):
            continue
        filtered.append(e)
    return filtered


def _extract_entity_from_sentence(text: str, candidates: List[str]) -> List[str]:
    """Fallback: extract entity names from a sentence by matching candidates."""
    text_lower = text.lower()
    found = []
    for cand in candidates:
        if cand.lower() in text_lower:
            found.append(cand)
    return found


def _parse_model_output(text: str, candidates: Optional[List[str]] = None) -> Dict[str, Any]:
    """Parse model output into checkpoint + actions/answer."""
    result: Dict[str, Any] = {
        "checkpoint": "",
        "tool_calls": [],
        "answer": [],
        "candidates": [],
    }

    # Extract checkpoint
    cp_match = re.search(r"<checkpoint>(.*?)</checkpoint>", text, re.S)
    if cp_match:
        result["checkpoint"] = cp_match.group(1).strip()

    # Extract answer from <answer> tag
    ans_match = re.search(r"<answer>(.*?)</answer>", text, re.S)
    if ans_match:
        ans_text = ans_match.group(1).strip()
        # Extract \boxed{} entities
        boxed = re.findall(r"\\boxed\{([^}]+)\}", ans_text)
        if boxed:
            result["answer"] = _filter_answer_entities(boxed)
        else:
            # No boxed — try candidate matching fallback
            if candidates:
                matched = _extract_entity_from_sentence(ans_text, candidates)
                if matched:
                    result["answer"] = matched
                else:
                    result["answer"] = _filter_answer_entities([ans_text.strip()])
            else:
                result["answer"] = _filter_answer_entities([ans_text.strip()])

    # Fallback: if no <answer> tag but text looks like a final answer (no tool calls)
    if not result["answer"]:
        act_match_check = re.search(r"<act>", text, re.S)
        if not act_match_check:
            # No action block — might be a sentence answer
            # Try candidate matching against full text
            if candidates:
                matched = _extract_entity_from_sentence(text, candidates)
                if matched:
                    result["answer"] = matched

    # Extract tool calls from <act> block
    act_match = re.search(r"<act>(.*?)</act>", text, re.S)
    if act_match:
        act_text = act_match.group(1)
        for qm in re.finditer(r"<query>(.*?)</query>", act_text, re.S):
            query_text = qm.group(1).strip()
            tc = _parse_single_tool_call(query_text)
            if tc:
                result["tool_calls"].append(tc)

    return result


def _parse_single_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse a single tool call like: tool_name(arg1="val1", arg2="val2")"""
    # Match function call pattern
    match = re.match(r"(\w+)\s*\((.*)\)\s*$", text.strip(), re.S)
    if not match:
        return None
    tool_name = match.group(1)
    args_text = match.group(2)

    args: Dict[str, Any] = {}
    # Parse keyword arguments
    for kv_match in re.finditer(r'(\w+)\s*=\s*(\[.*?\]|"[^"]*"|\'[^\']*\'|\S+)', args_text, re.S):
        key = kv_match.group(1)
        val_str = kv_match.group(2).strip()
        # Parse list
        if val_str.startswith("[") and val_str.endswith("]"):
            items = re.findall(r'"([^"]*)"|\'([^\']*)\'|(\w+)', val_str)
            args[key] = [i[0] or i[1] or i[2] for i in items if any(i)]
        elif val_str.startswith('"') and val_str.endswith('"'):
            args[key] = val_str[1:-1]
        elif val_str.startswith("'") and val_str.endswith("'"):
            args[key] = val_str[1:-1]
        else:
            args[key] = val_str

    return {"tool_name": tool_name, "arguments": args}


# ── Backend calls ──────────────────────────────────────────────────

async def _call_backend(
    session: aiohttp.ClientSession,
    tool_name: str,
    args: Dict[str, Any],
    sample_id: str,
) -> Dict[str, Any]:
    """Call a single graph backend tool."""
    url = TOOL_ENDPOINTS.get(tool_name)
    if not url:
        return {
            "tool_name": tool_name,
            "is_success": False,
            "status": "UNKNOWN_TOOL",
            "response_text": f"Unknown tool: {tool_name}",
            "found_end_entities": [],
            "action_hints": [],
        }

    payload = {**args, "sample_id": sample_id}
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            data = await resp.json()
            return {
                "tool_name": tool_name,
                "is_success": data.get("success", False),
                "status": data.get("status", ""),
                "response_text": data.get("response_text", ""),
                "found_end_entities": data.get("found_end_entities", []),
                "action_hints": data.get("action_hints", []),
                "structured_data": data.get("structured_data"),
            }
    except Exception as exc:
        return {
            "tool_name": tool_name,
            "is_success": False,
            "status": "BACKEND_ERROR",
            "response_text": str(exc),
            "found_end_entities": [],
            "action_hints": [],
        }


async def _call_llm(messages: List[Dict[str, str]]) -> str:
    """Call local LLM with thinking disabled."""
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": _env_int("KGQA_LLM_MAX_TOKENS", 1024),
        "temperature": _env_float("KGQA_LLM_TEMPERATURE", 0.3),
        "top_p": _env_float("KGQA_LLM_TOP_P", 0.8),
        "top_k": _env_int("KGQA_LLM_TOP_K", 20),
        "repetition_penalty": _env_float("KGQA_LLM_REPETITION_PENALTY", 1.0),
        "presence_penalty": _env_float("KGQA_LLM_PRESENCE_PENALTY", 0.0),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    timeout = aiohttp.ClientTimeout(total=_env_float("KGQA_LLM_TIMEOUT_SEC", 120.0))

    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        for attempt in range(3):
            try:
                async with session.post(f"{LLM_API_URL}/chat/completions", headers=headers, json=payload) as resp:
                    if resp.status in {429, 500, 502, 503, 504}:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    resp.raise_for_status()
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt >= 2:
                    raise
                await asyncio.sleep(2 ** attempt)
    raise RuntimeError("LLM call failed after retries")


# ── Pipeline simulation ────────────────────────────────────────────

async def _run_streamlined_case(
    *,
    case: Dict[str, Any],
    max_turns: int = 10,
    semaphore: asyncio.Semaphore,
    case_index: int,
    total_cases: int,
) -> Dict[str, Any]:
    """Run a single test case through the streamlined 5-stage pipeline."""
    async with semaphore:
        case_id = case.get("id", "unknown")
        user_message = next(
            (msg["content"] for msg in case.get("messages", []) if msg.get("role") == "user"),
            "",
        )
        gt = case.get("ground_truth", {})
        gt_answers = gt.get("global_truth_answers", []) or case.get("solution", [])

        # Extract question and domains
        question = _extract_clean_question(user_message)
        domains = _extract_domains(user_message)

        # Build initial user message for the pipeline
        domains_str = ", ".join(domains[:30]) if domains else "N/A"
        initial_user = (
            f"Available Domains: {domains_str}\n\n"
            f"Question: {question}"
        )

        # Conversation history
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": BASE_SYSTEM},
            {"role": "user", "content": initial_user},
        ]

        # Track state
        current_stage = 1
        all_tool_calls: List[Dict] = []
        all_backend_results: List[Dict] = []
        accumulated_candidates: List[str] = []  # all candidates found so far
        predicted: List[str] = []
        error_text = ""
        stage_prompt = STAGE1_DISCOVERY

        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as backend_session:
            turn = 0
            while turn < max_turns:
                turn += 1

                # Inject stage prompt as the last user instruction
                turn_messages = list(messages)
                # Add stage hint
                if turn == 1:
                    # First turn: stage 1 prompt is part of the user message
                    turn_messages[-1]["content"] += f"\n\n{STAGE1_DISCOVERY}"
                else:
                    # Subsequent turns: add stage prompt to user message
                    pass

                # Call LLM
                try:
                    raw_response = await _call_llm(turn_messages)
                except Exception as exc:
                    error_text = f"LLM error: {exc}"
                    break

                # Add assistant response
                messages.append({"role": "assistant", "content": raw_response})

                # Parse output
                parsed = _parse_model_output(raw_response, candidates=accumulated_candidates)

                # If answer found, we're done
                if parsed["answer"]:
                    predicted = parsed["answer"]
                    break

                # Execute tool calls
                tool_calls = parsed["tool_calls"]
                if not tool_calls:
                    # No tool calls and no answer — try to advance stage or backtrack
                    if current_stage < 5:
                        current_stage += 1
                        stage_prompt = STAGE_PROMPTS.get(current_stage, "")
                        messages.append({
                            "role": "user",
                            "content": f"Previous output had no actions. {stage_prompt}",
                        })
                    else:
                        messages.append({
                            "role": "user",
                            "content": BACKTRACK_HINT,
                        })
                        current_stage = 2
                    continue

                # Execute each tool call
                backend_text_parts = []
                new_candidates = set()
                for tc in tool_calls:
                    tool_name = tc["tool_name"]
                    args = tc["arguments"]
                    result = await _call_backend(backend_session, tool_name, args, case_id)
                    all_tool_calls.append(tc)
                    all_backend_results.append(result)

                    # Use compressed feedback for match_pattern, keep full text for others
                    if result["is_success"] and result["response_text"]:
                        if tool_name in ("match_pattern", "action"):
                            formatted = format_backend_feedback(
                                tool_name, result["response_text"],
                                found_entities=result.get("found_end_entities", []),
                            )
                            backend_text_parts.append(formatted)
                        else:
                            backend_text_parts.append(
                                f"[{tool_name}] {result['response_text'][:1500]}"
                            )

                    # Collect candidates
                    if result.get("found_end_entities"):
                        new_candidates.update(result["found_end_entities"])

                    # Detect stage from tools used
                    used_stage = _detect_stage_from_actions([tc])
                    if used_stage > current_stage:
                        current_stage = used_stage

                # Build feedback from backend results
                feedback = "\n\n".join(backend_text_parts) if backend_text_parts else "No results returned."

                # Accumulate candidates across turns
                for c in new_candidates:
                    if c not in accumulated_candidates:
                        accumulated_candidates.append(c)

                # Advance stage and inject next stage prompt
                candidates_msg = format_candidates_message(sorted(new_candidates))

                if current_stage <= 1 and any(
                    tc["tool_name"] in {"check_entities", "find_entities", "explore_schema"}
                    for tc in tool_calls
                ):
                    current_stage = 2
                    next_prompt = STAGE2_PLANNING
                elif current_stage <= 2 and any(
                    tc["tool_name"] in {"plan_subquestion", "plan"}
                    for tc in tool_calls
                ):
                    current_stage = 3
                    next_prompt = STAGE3_EXECUTION
                elif current_stage <= 3 and any(
                    tc["tool_name"] in {"match_pattern", "action"}
                    for tc in tool_calls
                ):
                    current_stage = 4
                    next_prompt = STAGE4_COLLECTION
                elif current_stage <= 4 and accumulated_candidates:
                    current_stage = 5
                    # Dynamic Stage 5 with only LATEST match_pattern leaf entities
                    latest_candidates = list(new_candidates) if new_candidates else accumulated_candidates[-10:]
                    next_prompt = build_stage5_with_candidates(latest_candidates, question)
                else:
                    next_prompt = BACKTRACK_HINT
                    current_stage = min(current_stage + 1, 5)

                # Inject backend results + next stage prompt
                user_msg = (
                    f"[BACKEND RESULTS]\n{feedback}\n\n"
                    f"{candidates_msg}\n\n"
                    f"{next_prompt}"
                )
                messages.append({"role": "user", "content": user_msg})

        # Calculate F1
        _, _, f1 = _entity_overlap(predicted, gt_answers)
        hit_at_1 = 1.0 if (predicted and _normalize_text(predicted[0]) in {_normalize_text(g) for g in gt_answers}) else 0.0

        print(f"[{case_index}/{total_cases}] {case_id}: F1={f1:.2f} | Turns={turn} | Stage={current_stage} | Pred={predicted[:2]} | GT={gt_answers[:2]}")

        return {
            "case_id": case_id,
            "question": question,
            "f1": f1,
            "hit_at_1": hit_at_1,
            "predicted": predicted,
            "ground_truth": gt_answers,
            "turns": turn,
            "final_stage": current_stage,
            "error": error_text,
            "tool_calls": [{"tool_name": tc["tool_name"], "arguments": tc["arguments"]} for tc in all_tool_calls],
        }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Streamlined prompt pipeline test on CWQ")
    parser.add_argument("--data-path", default="data/cwq/cwq_test.jsonl")
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--case-id", action="append", default=[], help="Run specific case IDs")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--label", default=None)
    parser.add_argument("--output-dir", default="reports/streamlined_test")
    args = parser.parse_args()

    if args.label is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.label = f"{timestamp}_streamlined"

    # Load cases
    data_path = Path(args.data_path)
    cases: List[Dict] = []
    wanted = set(args.case_id) if args.case_id else None
    with data_path.open() as f:
        for line in f:
            row = json.loads(line)
            if wanted and str(row.get("id")) not in wanted:
                continue
            if args.limit_cases and len(cases) >= args.limit_cases:
                break
            cases.append(row)
    print(f"Loaded {len(cases)} cases from {data_path}")

    # Run
    semaphore = asyncio.Semaphore(args.max_concurrency)
    start = asyncio.get_event_loop().time()

    tasks = [
        _run_streamlined_case(
            case=case,
            max_turns=args.max_turns,
            semaphore=semaphore,
            case_index=i + 1,
            total_cases=len(cases),
        )
        for i, case in enumerate(cases)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    clean = []
    failed = 0
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            failed += 1
            case = cases[i]
            clean.append({
                "case_id": case.get("id", f"case_{i}"),
                "question": "",
                "f1": 0.0,
                "hit_at_1": 0.0,
                "predicted": [],
                "ground_truth": case.get("solution", []),
                "turns": 0,
                "final_stage": 0,
                "error": str(r),
                "tool_calls": [],
            })
        else:
            clean.append(r)

    duration = asyncio.get_event_loop().time() - start

    # Report
    avg_f1 = mean(r["f1"] for r in clean) if clean else 0.0
    hit_at_1 = mean(r["hit_at_1"] for r in clean) if clean else 0.0
    avg_turns = mean(r["turns"] for r in clean) if clean else 0.0
    exact = mean(1.0 if r["f1"] >= 0.95 else 0.0 for r in clean) if clean else 0.0

    output_dir = Path(args.output_dir) / args.label
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(clean, ensure_ascii=False, indent=2))

    # Generate markdown report
    lines = [
        "# Streamlined Prompt Pipeline Report",
        "",
        f"- Label: `{args.label}`",
        f"- Data: `{args.data_path}`",
        f"- Max Turns: `{args.max_turns}`",
        f"- Timestamp: `{datetime.now().isoformat()}`",
        f"- Duration: `{duration:.1f}s`",
        "",
        "## Prompt Sizes",
        f"- BASE_SYSTEM: `{len(BASE_SYSTEM)}` chars",
        f"- STAGE1_DISCOVERY: `{len(STAGE1_DISCOVERY)}` chars",
        f"- STAGE2_PLANNING: `{len(STAGE2_PLANNING)}` chars",
        f"- STAGE3_EXECUTION: `{len(STAGE3_EXECUTION)}` chars",
        f"- STAGE4_COLLECTION: `{len(STAGE4_COLLECTION)}` chars",
        f"- STAGE5_REASONING: `{len(STAGE5_REASONING)}` chars",
        "",
        "## Overall Metrics",
        "",
        f"- Cases: `{len(clean)}` (failed: {failed})",
        f"- **Avg F1: `{avg_f1:.4f}`**",
        f"- **Hit@1: `{hit_at_1:.4f}`**",
        f"- **Exact Match: `{exact:.4f}`**",
        f"- Avg Turns: `{avg_turns:.2f}`",
        "",
        "## Per-Case Results",
        "",
        "| Case ID | F1 | Hit@1 | Turns | Stage | Predicted | GT |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for item in sorted(clean, key=lambda x: x["f1"], reverse=True):
        pred = ", ".join(item["predicted"][:3]) if item["predicted"] else "-"
        gt = ", ".join(item["ground_truth"][:3]) if item["ground_truth"] else "-"
        lines.append(
            f"| {item['case_id']} | {item['f1']:.2f} | {item['hit_at_1']:.2f} | "
            f"{item['turns']} | {item['final_stage']} | {pred} | {gt} |"
        )

    # Failed cases
    fail_cases = [r for r in clean if r["f1"] < 0.5]
    if fail_cases:
        lines.extend([
            "",
            "## Failed Cases (F1 < 0.5)",
            "",
            "| Case ID | F1 | Turns | Question | Predicted | GT |",
            "|---|---:|---:|---|---|---|",
        ])
        for item in sorted(fail_cases, key=lambda x: x["f1"]):
            pred = ", ".join(item["predicted"][:2]) if item["predicted"] else "-"
            gt = ", ".join(item["ground_truth"][:2]) if item["ground_truth"] else "-"
            lines.append(
                f"| {item['case_id']} | {item['f1']:.2f} | {item['turns']} | "
                f"{item['question']} | {pred} | {gt} |"
            )

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines))

    print(f"\n=== Streamlined Pipeline Results ===")
    print(f"Avg F1: {avg_f1:.4f}")
    print(f"Hit@1: {hit_at_1:.4f}")
    print(f"Exact Match: {exact:.4f}")
    print(f"Avg Turns: {avg_turns:.2f}")
    print(f"Report: {report_path}")
    print(f"JSON: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
