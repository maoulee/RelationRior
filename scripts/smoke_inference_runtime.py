from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import aiohttp

from subgraph_kgqa.inference import GraphBackendClient, InferenceRuntime
from subgraph_kgqa.inference import consistent_call
from subgraph_kgqa.rl.plugin import calculate_f1, parse_prompt_context
from subgraph_kgqa.skill_mining.retriever import build_retrieved_skill_bundle
from sys_prompt import get_prompt_variant_followup_hint, get_system_prompt


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


def _dynamic_turn_cap(base_max_turns: int) -> int:
    if os.getenv("KGQA_EXTEND_TURNS_ON_FRONTEND_ERROR", "1").strip().lower() not in {"1", "true", "yes", "on"}:
        return base_max_turns
    return max(base_max_turns, _env_int("KGQA_MAX_TURNS_WITH_FRONTEND_REPAIR", 16))


def _load_case(data_path: Path, case_id: str | None) -> Dict[str, Any]:
    with data_path.open() as handle:
        for line in handle:
            case = json.loads(line)
            if case_id is None or case.get("id") == case_id:
                return case
    raise ValueError(f"Unable to find case_id={case_id!r} in {data_path}")


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


def _render_report(
    *,
    case: Dict[str, Any],
    turns: List[Any],
    predicted: List[str],
    f1: float,
    variant: str,
    skill_bundle: Dict[str, Any] | None = None,
    consistency_meta: Dict[str, Any] | None = None,
) -> str:
    gt = case.get("ground_truth", {}).get("global_truth_answers", []) or case.get("solution", [])
    lines = [
        "# Inference Runtime Smoke Test",
        "",
        f"- Variant: `{variant}`",
        f"- Case ID: `{case.get('id', 'unknown')}`",
        f"- F1: `{f1:.4f}`",
        f"- Ground Truth: `{gt}`",
        f"- Predicted: `{predicted}`",
        "",
    ]
    if skill_bundle:
        lines.extend(
            [
                "## Retrieved Skills",
                "",
                f"- Shortlisted Case IDs: `{skill_bundle.get('shortlisted_case_ids', [])}`",
                f"- Selected Skill IDs: `{skill_bundle.get('selected_case_ids', [])}`",
                f"- Retrieved Case IDs: `{skill_bundle.get('retrieved_case_ids', [])}`",
                f"- Retrieval Note: `{skill_bundle.get('retrieval_note', '')}`",
                "",
                "### Relation-Stage Hint",
                "```text",
                skill_bundle.get("relation_stage_hint", ""),
                "```",
                "",
                "### Reasoning-Stage Hint",
                "```text",
                skill_bundle.get("reasoning_stage_hint", ""),
                "```",
                "",
            ]
        )
    if turns:
        lines.append(f"- Hint History: `{turns[-1].state_snapshot.get('repair_mode')}` / `{len(turns)}` turns")
        if consistency_meta:
            lines.append(f"- Consistency: agreed={consistency_meta.get('consistency_agreed_initially')} sig={consistency_meta.get('consistency_chosen_signature', '')[:60]}")
        lines.append("")

    for index, turn in enumerate(turns, 1):
        lines.extend(
            [
                f"## Turn {index}",
                "",
                f"- Frontend Errors: `{len(turn.frontend_errors)}`",
                f"- Executed Queries: `{[q.get('tool_name') for q in turn.executed_queries]}`",
                f"- Repair Mode: `{turn.state_snapshot.get('repair_mode')}`",
                f"- Candidates: `{turn.state_snapshot.get('retrieved_candidates')}`",
                "",
                "### Raw Response",
                "```text",
                turn.raw_response,
                "```",
                "",
                "### Feedback",
                "```text",
                turn.feedback,
                "```",
                "",
            ]
        )
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single-case smoke test through the new inference runtime.")
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--variant", default=os.getenv("KGQA_PROMPT_VARIANT", "original"))
    parser.add_argument("--data-path", default="data/webqsp/webqsp_train.jsonl")
    parser.add_argument("--max-turns", type=int, default=_env_int("KGQA_MAX_TURNS", 8))
    parser.add_argument("--inject-similar-skills", action="store_true")
    parser.add_argument("--skills-root", default="skills")
    parser.add_argument("--skill-top-k", type=int, default=3)
    args = parser.parse_args()

    variant_name = (args.variant or "original").strip().lower()
    if variant_name in {
        "action_id_experiment",
        "compact_relation_experiment",
        "compact_relation_action_id_experiment",
        "checklist_action_id_experiment",
        "protocol_guard_action_id_experiment",
        "workflow_free_action_id_experiment",
    }:
        os.environ.setdefault("KGQA_ACTION_ID_MODE", "1")
    if variant_name in {"checklist_action_id_experiment", "protocol_guard_action_id_experiment"}:
        os.environ.setdefault("KGQA_CHECKLIST_MODE", "1")
    if variant_name in {"compact_relation_experiment", "compact_relation_action_id_experiment"}:
        os.environ.setdefault("KGQA_COMPACT_RELATION_MODE", "1")

    case = _load_case(Path(args.data_path), args.case_id)
    question = next((msg["content"] for msg in case.get("messages", []) if msg.get("role") == "user"), "")
    gt = case.get("ground_truth", {})
    raw_core_entities = gt.get("core_entities", [])
    raw_core_relations = gt.get("core_relations", [])
    prompt_context = parse_prompt_context(question)
    stage_skill_hints: Dict[str, str] = {}
    skill_bundle_payload: Dict[str, Any] | None = None
    if args.inject_similar_skills:
        bundle = await build_retrieved_skill_bundle(
            question_text=question,
            data_path=Path(args.data_path),
            skills_root=Path(args.skills_root),
            exclude_case_id=case.get("id"),
            top_k=args.skill_top_k,
            use_llm=True,
        )
        stage_skill_hints = {
            "stage:4": bundle.reasoning_stage_hint,
        }
        skill_bundle_payload = {
            "shortlisted_case_ids": bundle.shortlisted_case_ids,
            "selected_case_ids": bundle.selected_case_ids,
            "retrieved_case_ids": bundle.retrieved_case_ids,
            "retrieval_note": bundle.retrieval_note,
            "relation_stage_hint": bundle.relation_stage_hint,
            "reasoning_stage_hint": bundle.reasoning_stage_hint,
        }

    runtime = InferenceRuntime(
        backend_client=GraphBackendClient(),
        system_prompt=get_system_prompt(args.variant),
        followup_hint=get_prompt_variant_followup_hint(args.variant),
        stage_skill_hints=stage_skill_hints,
        retrieved_skill_cards=bundle.retrieved_cards if args.inject_similar_skills else None,
        skill_target_question=bundle.target_question if args.inject_similar_skills else "",
    )
    conversation, state = runtime.create_session(
        question=question,
        core_entities=raw_core_entities,
        core_relations=prompt_context["core_relations"] | set(raw_core_relations),
        available_domains=prompt_context["available_domains"],
    )
    turns = []
    predicted: List[str] = []

    consistency_meta: Dict[str, Any] = {}
    timeout = aiohttp.ClientTimeout(total=180.0)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as backend_session:
        current_turn_budget = args.max_turns
        max_turn_cap = _dynamic_turn_cap(args.max_turns)
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

    gt_answers = gt.get("global_truth_answers", []) or case.get("solution", [])
    f1 = calculate_f1(predicted, gt_answers)
    report = _render_report(
        case=case,
        turns=turns,
        predicted=predicted,
        f1=f1,
        variant=args.variant,
        skill_bundle=skill_bundle_payload,
        consistency_meta=consistency_meta,
    )
    suffix = "_skills" if args.inject_similar_skills else ""
    report_path = Path("reports") / f"inference_runtime_smoke_{case.get('id', 'unknown')}_{args.variant}{suffix}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(report_path)
    print(f"F1={f1:.4f}")
    print(f"Predicted={predicted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
