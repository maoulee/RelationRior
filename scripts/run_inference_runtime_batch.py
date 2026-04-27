from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subgraph_kgqa.inference import GraphBackendClient, InferenceRuntime
from subgraph_kgqa.inference import consistent_call
from subgraph_kgqa.rl.plugin import calculate_f1, parse_prompt_context
from sys_prompt import get_prompt_variant_followup_hint, get_system_prompt


DEFAULT_VARIANTS = [
    "original",
    "action_id_experiment",
    "checklist_action_id_experiment",
]


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


def _load_cases(data_path: Path, limit: int) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with data_path.open() as handle:
        for line in handle:
            if len(cases) >= limit:
                break
            cases.append(json.loads(line))
    if not cases:
        raise ValueError(f"No cases found in {data_path}")
    return cases


async def _call_llm(messages: List[Dict[str, str]]) -> str:
    api_url = os.getenv("KGQA_LLM_API_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    api_key = os.getenv("KGQA_LLM_API_KEY", "EMPTY")
    model_name = os.getenv("KGQA_MODEL_NAME", "qwen35-9b-local")
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": _env_int("KGQA_LLM_MAX_TOKENS", 2048),
        "temperature": _env_float("KGQA_LLM_TEMPERATURE", 0.7),
        "top_p": _env_float("KGQA_LLM_TOP_P", 0.8),
        "presence_penalty": _env_float("KGQA_LLM_PRESENCE_PENALTY", 1.5),
        "repetition_penalty": _env_float("KGQA_LLM_REPETITION_PENALTY", 1.0),
    }
    top_k = _env_int("KGQA_LLM_TOP_K", 20)
    min_p = _env_float("KGQA_LLM_MIN_P", 0.0)
    enable_thinking = os.getenv("KGQA_ENABLE_THINKING", "0").strip().lower() in {"1", "true", "yes"}
    if top_k > 0:
        payload["top_k"] = top_k
    if min_p > 0:
        payload["min_p"] = min_p
    if not enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    timeout = aiohttp.ClientTimeout(total=_env_float("KGQA_LLM_TIMEOUT_SEC", 180.0))
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        async with session.post(f"{api_url}/chat/completions", headers=headers, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            return data["choices"][0]["message"]["content"]


def _endpoints_from_base(base_url: str) -> Dict[str, str]:
    base = base_url.rstrip("/")
    return {
        "find_entities": f"{base}/v2/find_entities",
        "check_entities": f"{base}/v2/find_entities",
        "explore_schema": f"{base}/v2/explore_schema",
        "plan_subquestion": f"{base}/v2/plan_subquestion",
        "plan": f"{base}/v2/plan_subquestion",
        "find_logical_path_with_relation": f"{base}/v2/find_logical_path_with_relation",
        "match_pattern": f"{base}/v2/match_pattern",
        "action": f"{base}/v2/match_pattern",
        "get_neighbors": f"{base}/v2/get_neighbors",
        "filter": f"{base}/v2/filter",
    }


@contextmanager
def _variant_mode_env(variant: str):
    tracked = {
        "KGQA_ACTION_ID_MODE": os.environ.get("KGQA_ACTION_ID_MODE"),
        "KGQA_CHECKLIST_MODE": os.environ.get("KGQA_CHECKLIST_MODE"),
        "KGQA_COMPACT_RELATION_MODE": os.environ.get("KGQA_COMPACT_RELATION_MODE"),
    }
    try:
        for name in tracked:
            os.environ.pop(name, None)
        if variant in {
            "action_id_experiment",
            "compact_relation_experiment",
            "compact_relation_action_id_experiment",
            "checklist_action_id_experiment",
            "protocol_guard_action_id_experiment",
            "workflow_free_action_id_experiment",
        }:
            os.environ["KGQA_ACTION_ID_MODE"] = "1"
        if variant in {"checklist_action_id_experiment", "protocol_guard_action_id_experiment"}:
            os.environ["KGQA_CHECKLIST_MODE"] = "1"
        if variant in {"compact_relation_experiment", "compact_relation_action_id_experiment"}:
            os.environ["KGQA_COMPACT_RELATION_MODE"] = "1"
        yield
    finally:
        for name, value in tracked.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _backend_for_variant(variant: str, original_url: str, experiment_url: str) -> str:
    return original_url if variant == "original" else experiment_url


def _dynamic_turn_cap(base_max_turns: int) -> int:
    if os.getenv("KGQA_EXTEND_TURNS_ON_FRONTEND_ERROR", "1").strip().lower() not in {"1", "true", "yes", "on"}:
        return base_max_turns
    try:
        configured_cap = int(os.getenv("KGQA_MAX_TURNS_WITH_FRONTEND_REPAIR", "16"))
    except Exception:
        configured_cap = 16
    return max(base_max_turns, configured_cap)


async def _run_case(
    *,
    case: Dict[str, Any],
    variant: str,
    backend_base_url: str,
    max_turns: int,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    async with semaphore:
        question = next((msg["content"] for msg in case.get("messages", []) if msg.get("role") == "user"), "")
        gt = case.get("ground_truth", {})
        raw_core_entities = gt.get("core_entities", [])
        raw_core_relations = gt.get("core_relations", [])
        prompt_context = parse_prompt_context(question)

        runtime = InferenceRuntime(
            backend_client=GraphBackendClient(endpoints=_endpoints_from_base(backend_base_url)),
            system_prompt=get_system_prompt(variant),
            followup_hint=get_prompt_variant_followup_hint(variant),
        )
        conversation, state = runtime.create_session(
            question=question,
            core_entities=raw_core_entities,
            core_relations=prompt_context["core_relations"] | set(raw_core_relations),
            available_domains=prompt_context["available_domains"],
        )

        turns = []
        predicted: List[str] = []
        error_text = ""
        consistency_meta: Dict[str, Any] = {}
        timeout = aiohttp.ClientTimeout(total=180.0)
        try:
            async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as backend_session:
                current_turn_budget = max_turns
                max_turn_cap = _dynamic_turn_cap(max_turns)
                turn_idx = 0
                while turn_idx < current_turn_budget:
                    prepared_messages = runtime.prepare_messages(conversation, state)
                    raw_response, _turn_consistency = await consistent_call(
                        _call_llm,
                        prepared_messages,
                        turn_number=turn_idx + 1,
                    )
                    if _turn_consistency:
                        consistency_meta = _turn_consistency
                    turn = await runtime.apply_model_response(
                        session=backend_session,
                        sample_id=case.get("id", "unknown"),
                        conversation=conversation,
                        state=state,
                        raw_response=raw_response,
                    )
                    turns.append(turn)
                    if turn.frontend_errors and current_turn_budget < max_turn_cap:
                        current_turn_budget = min(max_turn_cap, current_turn_budget + 1)
                    if turn.parsed_output.get("final_answer") and not turn.frontend_errors:
                        predicted = turn.parsed_output["final_answer"]
                        break
                    turn_idx += 1
        except Exception as exc:
            error_text = str(exc)

        gt_answers = gt.get("global_truth_answers", []) or case.get("solution", [])
        f1 = calculate_f1(predicted, gt_answers)
        hit_at_1 = 1.0 if (predicted and predicted[0] in gt_answers) else 0.0
        frontend_errors = sum(len(turn.frontend_errors) for turn in turns)
        return {
            "case_id": case.get("id", "unknown"),
            "f1": f1,
            "hit_at_1": hit_at_1,
            "predicted": predicted,
            "ground_truth": gt_answers,
            "turns": len(turns),
            "frontend_errors": frontend_errors,
            "repair_mode": turns[-1].state_snapshot.get("repair_mode") if turns else None,
            "error": error_text,
            "consistency": consistency_meta,
        }


def _render_variant_report(
    *,
    variant: str,
    backend_base_url: str,
    results: List[Dict[str, Any]],
) -> str:
    avg_f1 = mean(item["f1"] for item in results) if results else 0.0
    hit_at_1 = mean(item.get("hit_at_1", 0.0) for item in results) if results else 0.0
    avg_turns = mean(item["turns"] for item in results) if results else 0.0
    total_frontend_errors = sum(item["frontend_errors"] for item in results)

    lines = [
        "# Inference Runtime Batch Report",
        "",
        f"- Variant: `{variant}`",
        f"- Backend: `{backend_base_url}`",
        f"- Cases: `{len(results)}`",
        f"- Avg F1: `{avg_f1:.4f}`",
        f"- Hit@1 (first pred in GT): `{hit_at_1:.4f}`",
        f"- Avg Turns: `{avg_turns:.2f}`",
        f"- Frontend Errors: `{total_frontend_errors}`",
        "",
        "| Case ID | F1 | Turns | Frontend Errors | Repair Mode | Predicted | Error |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for item in results:
        predicted = ", ".join(item["predicted"]) if item["predicted"] else "-"
        error = item["error"].replace("|", "/") if item["error"] else "-"
        lines.append(
            f"| {item['case_id']} | {item['f1']:.2f} | {item['turns']} | {item['frontend_errors']} | "
            f"{item['repair_mode'] or '-'} | {predicted} | {error} |"
        )
    return "\n".join(lines)


def _render_summary(label: str, variant_results: Dict[str, List[Dict[str, Any]]]) -> str:
    lines = [
        "# Inference Runtime Batch Summary",
        "",
        f"- Label: `{label}`",
        "",
        "| Variant | Cases | Avg F1 | Hit@1 | Avg Turns | Frontend Errors |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, results in variant_results.items():
        avg_f1 = mean(item["f1"] for item in results) if results else 0.0
        hit_at_1 = mean(item.get("hit_at_1", 0.0) for item in results) if results else 0.0
        avg_turns = mean(item["turns"] for item in results) if results else 0.0
        total_frontend_errors = sum(item["frontend_errors"] for item in results)
        lines.append(
            f"| {variant} | {len(results)} | {avg_f1:.4f} | {hit_at_1:.4f} | {avg_turns:.2f} | {total_frontend_errors} |"
        )
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-run inference-only runtime experiments.")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--limit-cases", type=int, default=30)
    parser.add_argument("--data-path", default="data/webqsp/webqsp_train.jsonl")
    parser.add_argument("--max-turns", type=int, default=_env_int("KGQA_MAX_TURNS", 8))
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--label", default="batch30")
    parser.add_argument("--original-kg-api-url", default="http://127.0.0.1:8014")
    parser.add_argument("--experiment-kg-api-url", default="http://127.0.0.1:8013")
    args = parser.parse_args()

    cases = _load_cases(Path(args.data_path), args.limit_cases)
    out_dir = Path("reports") / "inference_runtime_batch" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    variant_results: Dict[str, List[Dict[str, Any]]] = {}
    for variant in args.variants:
        backend_base_url = _backend_for_variant(
            variant,
            original_url=args.original_kg_api_url,
            experiment_url=args.experiment_kg_api_url,
        )
        print(f"[variant] {variant} -> {backend_base_url}")
        with _variant_mode_env(variant):
            semaphore = asyncio.Semaphore(args.max_concurrency)
            tasks = [
                _run_case(
                    case=case,
                    variant=variant,
                    backend_base_url=backend_base_url,
                    max_turns=args.max_turns,
                    semaphore=semaphore,
                )
                for case in cases
            ]
            results = await asyncio.gather(*tasks)
        variant_results[variant] = results
        report_text = _render_variant_report(
            variant=variant,
            backend_base_url=backend_base_url,
            results=results,
        )
        report_path = out_dir / f"{variant}.md"
        report_path.write_text(report_text)
        print(report_path)

    summary_path = out_dir / "summary.md"
    summary_path.write_text(_render_summary(args.label, variant_results))
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
