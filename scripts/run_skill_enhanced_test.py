#!/usr/bin/env python3
"""
Skill-Enhanced Batch Test Runner

Runs batch evaluation on WebQSP test set with retrieved training skills as hints.

Usage:
    python scripts/run_skill_enhanced_test.py \\
        --data-path data/webqsp/webqsp_test.jsonl \\
        --limit-cases 100 \\
        --max-concurrency 32 \\
        --skill-top-k 3 \\
        --label test_with_skills
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
from subgraph_kgqa.skill_mining.intent_analyzer import (
    IntentSignature,
    analyze_question_intent,
    is_intent_analysis_enabled,
    resolve_skill_conflicts,
)
from subgraph_kgqa.skill_mining.retriever import build_classified_discovery_hint, build_retrieved_skill_bundle, prewarm_precomputed_embedding_indexes
from subgraph_kgqa.skill_mining.skill_aggregator import (
    aggregate_skills,
    format_aggregated_skill_for_prompt,
    format_aggregated_stage_hints,
    is_skill_aggregation_enabled,
)
from subgraph_kgqa.inference.audit_agent import (
    audit_final_answer,
    format_audit_feedback_for_rereasoning,
    is_audit_agent_enabled,
)
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
    configured_cap = _env_int("KGQA_MAX_TURNS_WITH_FRONTEND_REPAIR", 16)
    return max(base_max_turns, configured_cap)


def _load_cases(data_path: Path, limit: int | None, case_ids: Iterable[str] | None = None) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    wanted = set(case_ids or [])
    with data_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if wanted and str(row.get("id")) not in wanted:
                continue
            if limit is not None and len(cases) >= limit:
                break
            cases.append(row)
    if not cases:
        raise ValueError(f"No cases found in {data_path}")
    return cases


def _extract_clean_question(user_message: str) -> str:
    """
    Extract the clean question text from a user message that may contain
    system prompts and context.

    Test set messages have format like:
        Available Domains...
        [Retrieval Context]...
        Question:
        where is jamarcus russell from
        [PHASE 1]...

    We need to extract just the question line.
    """
    if "Question:" in user_message:
        # Split by "Question:" and take the part after it
        after_question = user_message.split("Question:", 1)[1]
        # Take only the first line (the actual question)
        question = after_question.strip().split("\n")[0].strip()
        return question
    # Fallback: return as-is
    return user_message.strip()


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


def _extract_entities_from_check_response(text: str) -> List[str]:
    entities: List[str] = []
    for match in re.finditer(r"-\s+([^\[\n]+?)(?:\s*\[Context:|\s*$)", text):
        name = match.group(1).strip()
        if name and len(name) > 1:
            entities.append(name)
    return entities[:20]


def _extract_entities_from_action_response(text: str) -> List[str]:
    entities: List[str] = []
    for pattern in [
        r'(?:Leaf|Target|CVT-Expanded)\s+(?:Entities?)\s*\([^)]*\):\s*\n\s*\[([^\]]+)\]',
        r'Candidates?:\s*\[([^\]]+)\]',
    ]:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            list_str = match.group(1)
            for item in re.findall(r"'([^']*)'|\"([^\"]*)\"|,\s*([^,\[\]]+)", list_str):
                for value in item:
                    if value and value.strip():
                        entities.append(value.strip())
    if not entities:
        for match in re.finditer(r"^\s+-\s+([A-Z][^\n]{2,80})", text, re.MULTILINE):
            entities.append(match.group(1).strip())
    return entities[:50]


def _extract_entities_from_filter_response(text: str) -> List[str]:
    entities: List[str] = []
    for match in re.finditer(r"✓\s+([^:\n]+):", text):
        name = match.group(1).strip()
        if name:
            entities.append(name)
    return entities[:50]


def _extract_stage_scores(
    *,
    trajectory: List[Dict[str, Any]],
    core_entities: List[str],
    core_relations: List[str],
    truth_answers: List[str],
) -> Dict[str, Any]:
    stages: Dict[str, Any] = {
        "entity_linking": {"entities_found": []},
        "plan": {"planned_relations": []},
        "action": {"candidates_retrieved": []},
        "filter": {"candidates_after_filter": []},
        "final_reasoning": {"answer": []},
    }

    for step in trajectory:
        parsed = step.get("parsed_output") or {}
        queries = parsed.get("queries") or []
        candidates = parsed.get("candidates") or []
        final_answer = parsed.get("final_answer") or []

        if candidates:
            stages["action"]["candidates_retrieved"] = candidates
        if final_answer:
            stages["final_reasoning"]["answer"] = final_answer

        for query in queries:
            tool = query.get("tool_name", "")
            args = query.get("arguments", {})
            if tool == "plan":
                related = args.get("related", [])
                maybe_related = args.get("maybe_related", [])
                constraint_relations = args.get("constraint_relations", [])
                stages["plan"]["planned_relations"] = related + maybe_related + constraint_relations
            elif tool == "filter":
                constraint_relations = args.get("constraint_relations", [])
                if constraint_relations:
                    stages["filter"]["filter_relations"] = constraint_relations

        for backend_result in step.get("backend_results") or []:
            tool_name = backend_result.get("tool_name", "")
            response_text = backend_result.get("response_text", "")
            if tool_name == "check_entities" and response_text:
                found = _extract_entities_from_check_response(response_text)
                if found:
                    stages["entity_linking"]["entities_found"] = found
            elif tool_name in ("action", "match_pattern") and response_text:
                found = _extract_entities_from_action_response(response_text)
                if found:
                    stages["action"]["candidates_retrieved"] = found
            elif tool_name == "filter" and response_text:
                found = _extract_entities_from_filter_response(response_text)
                if found:
                    stages["filter"]["candidates_after_filter"] = found

    if not stages["filter"]["candidates_after_filter"] and stages["action"]["candidates_retrieved"]:
        stages["filter"]["candidates_after_filter"] = stages["action"]["candidates_retrieved"]

    entity_p, entity_r, entity_f1 = _entity_overlap(stages["entity_linking"]["entities_found"], core_entities)

    plan_pred = stages["plan"]["planned_relations"]
    plan_precision = 0.0
    plan_recall = 0.0
    if core_relations and plan_pred:
        matched_gt = sum(1 for rel in core_relations if any(rel in pred or pred in rel for pred in plan_pred))
        matched_pred = sum(1 for pred in plan_pred if any(rel in pred or pred in rel for rel in core_relations))
        plan_recall = matched_gt / len(core_relations)
        plan_precision = matched_pred / len(plan_pred) if plan_pred else 0.0
    plan_f1 = (
        2 * plan_precision * plan_recall / (plan_precision + plan_recall)
        if (plan_precision + plan_recall) > 0
        else 0.0
    )

    action_p, action_r, action_f1 = _entity_overlap(stages["action"]["candidates_retrieved"], truth_answers)
    filter_p, filter_r, filter_f1 = _entity_overlap(stages["filter"]["candidates_after_filter"], truth_answers)
    reason_p, reason_r, reason_f1 = _entity_overlap(stages["final_reasoning"]["answer"], truth_answers)

    return {
        "entity_linking": {"precision": entity_p, "recall": entity_r, "f1": entity_f1},
        "plan": {"precision": plan_precision, "recall": plan_recall, "f1": plan_f1},
        "action": {"precision": action_p, "recall": action_r, "f1": action_f1},
        "filter": {"precision": filter_p, "recall": filter_r, "f1": filter_f1},
        "final_reasoning": {"precision": reason_p, "recall": reason_r, "f1": reason_f1},
    }


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
    top_k = _env_int("KGQA_LLM_TOP_K", _env_int("KGQA_TOP_K", 20))
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
    max_attempts = _env_int("KGQA_LLM_RETRY_ATTEMPTS", 6)
    base_sleep = _env_float("KGQA_LLM_RETRY_BASE_SEC", 2.0)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                async with session.post(f"{api_url}/chat/completions", headers=headers, json=payload) as response:
                    if response.status in {429, 500, 502, 503, 504}:
                        body = await response.text()
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=body[:500],
                            headers=response.headers,
                        )
                    response.raise_for_status()
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
            except (aiohttp.ClientResponseError, aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
                last_error = exc
                should_retry = True
                if isinstance(exc, aiohttp.ClientResponseError) and exc.status not in {429, 500, 502, 503, 504}:
                    should_retry = False
                if not should_retry or attempt >= max_attempts:
                    break
                sleep_for = base_sleep * (2 ** (attempt - 1)) + random.uniform(0.0, 0.5)
                await asyncio.sleep(sleep_for)
        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM call failed without a captured exception")


async def _run_single_case(
    *,
    case: Dict[str, Any],
    variant: str,
    query_data_path: Path,
    skills_root: Path,
    negative_skills_root: Optional[Path] = None,
    train_data_path: Path,
    skill_top_k: int,
    inject_skills: bool,
    allow_online_skill_build: bool,
    max_turns: int,
    stage5_prompt_variant: str,
    semaphore: asyncio.Semaphore,
    case_index: int,
    total_cases: int,
) -> Dict[str, Any]:
    """Run a single test case with optional skill retrieval."""
    async with semaphore:
        case_id = case.get("id", "unknown")
        user_message = next((msg["content"] for msg in case.get("messages", []) if msg.get("role") == "user"), "")
        gt = case.get("ground_truth", {})
        raw_core_entities = gt.get("core_entities", [])
        raw_core_relations = gt.get("core_relations", [])
        prompt_context = parse_prompt_context(user_message)

        # Extract clean question text early (used by both intent analysis and skill retrieval)
        clean_question = _extract_clean_question(user_message) if inject_skills else user_message[:200]

        # Retrieve similar skills if enabled
        stage_skill_hints: Dict[str, Any] = {}
        skill_bundle_info: Dict[str, Any] = {}
        intent_info: Dict[str, Any] = {}
        agg_hints: Dict[str, str] = {}
        bundle = None
        _aggregated_skill = None
        _question_intent: Optional[IntentSignature] = None

        # Intent analysis (feature-gated, default off)
        if inject_skills and is_intent_analysis_enabled():
            try:
                _question_intent = await analyze_question_intent(clean_question)
                intent_info = {
                    "answer_type": _question_intent.answer_type,
                    "cardinality": _question_intent.cardinality,
                    "temporal_type": _question_intent.temporal_type,
                    "temporal_reference": _question_intent.temporal_reference,
                    "intent_signature": _question_intent.intent_signature_str,
                    "ambiguity_flags": _question_intent.ambiguity_flags,
                }
            except Exception as exc:
                intent_info = {"error": str(exc)}

        if inject_skills:
            try:
                # Derive query domains from ground-truth core_relations for domain filtering.
                # This extracts top-level domains (e.g. "people" from "people.person.nationality").
                _all_rels = list(raw_core_relations) + list(prompt_context.get("core_relations") or [])
                _query_domains = {r.split(".", 1)[0] for r in _all_rels if "." in r} if _all_rels else None

                bundle = await build_retrieved_skill_bundle(
                    question_text=clean_question,
                    data_path=train_data_path,
                    skills_root=skills_root,
                    exclude_case_id=None,  # Test cases are different from train cases
                    query_case_id=case_id,
                    query_data_path=query_data_path,
                    top_k=skill_top_k,
                    use_llm=allow_online_skill_build,
                    negative_skills_root=negative_skills_root,
                    query_domains=_query_domains,
                )

                # Intent-aware conflict resolution
                _conflict_log: List[Dict[str, str]] = []
                if _question_intent is not None and bundle.retrieved_cards:
                    bundle.retrieved_cards, _conflict_log = resolve_skill_conflicts(
                        bundle.retrieved_cards, _question_intent, max_skills=skill_top_k,
                    )

                stage_skill_hints = {
                    "stage:1": build_classified_discovery_hint(bundle.retrieved_cards),
                    "stage:4": bundle.reasoning_stage_hint,
                }

                # Inject intent hints into stage:4 (prepended to reasoning hint)
                if _question_intent is not None and _question_intent.intent_signature_str:
                    intent_hint_parts = ["[INTENT ANALYSIS]"]
                    if _question_intent.answer_type:
                        intent_hint_parts.append(f"Expected answer type: {_question_intent.answer_type}"
                                                 + (f" ({_question_intent.cardinality})" if _question_intent.cardinality else ""))
                    if _question_intent.temporal_type and _question_intent.temporal_type != "none":
                        intent_hint_parts.append(f"Temporal scope: {_question_intent.temporal_type}"
                                                 + (f" reference={_question_intent.temporal_reference}" if _question_intent.temporal_reference else ""))
                    if _question_intent.scope_entity_types:
                        intent_hint_parts.append(f"Scope includes: {', '.join(_question_intent.scope_entity_types)}")
                    if _question_intent.ambiguity_flags:
                        intent_hint_parts.append(f"Ambiguity flags: {', '.join(_question_intent.ambiguity_flags)}")
                    intent_hint_parts.append("Use this intent analysis to guide your answer filtering and selection.\n")
                    intent_prefix = "\n".join(intent_hint_parts)
                    stage_skill_hints["stage:4"] = intent_prefix + "\n" + stage_skill_hints["stage:4"]

                # --- Skill Aggregation (feature-gated, independent of intent analysis) ---
                # LLM synthesizes retrieved skills into unified, context-aware guidance
                _aggregated_skill = None
                if (
                    inject_skills
                    and is_skill_aggregation_enabled()
                    and bundle is not None
                    and bundle.retrieved_cards
                    and len(bundle.retrieved_cards) > 0
                ):
                    try:
                        _aggregated_skill = await aggregate_skills(
                            clean_question, bundle.retrieved_cards,
                            negative_skills=bundle.retrieved_negative_cards if bundle.retrieved_negative_cards else None,
                        )
                        aggregated_skill_summary = {
                            "question_analysis": _aggregated_skill.question_analysis,
                            "answer_type_guidance": _aggregated_skill.answer_type_guidance,
                            "temporal_guidance": _aggregated_skill.temporal_guidance,
                            "scope_guidance": _aggregated_skill.scope_guidance,
                            "pitfalls": _aggregated_skill.pitfalls,
                            "conflict_notes": _aggregated_skill.conflict_notes,
                        }
                        skill_bundle_info["aggregated_skill"] = aggregated_skill_summary
                        # Replace stage:4 hint with aggregated skill
                        if _aggregated_skill.combined_reasoning_hint:
                            stage_skill_hints["stage:4"] = format_aggregated_skill_for_prompt(_aggregated_skill)

                        # Inject error-type-aware stage hints from aggregated skill
                        # Pass separately to InferenceRuntime for proper labeling
                        agg_hints = format_aggregated_stage_hints(_aggregated_skill)
                    except Exception as exc:
                        skill_bundle_info["aggregation_error"] = str(exc)

                skill_bundle_info.update({
                    "retrieved_case_ids": bundle.retrieved_case_ids,
                    "shortlisted_case_ids": bundle.shortlisted_case_ids,
                    "selected_case_ids": bundle.selected_case_ids,
                    "retrieval_note": bundle.retrieval_note,
                    "audit_candidate_ids": bundle.audit_candidate_ids,
                    "audit_kept_ids": bundle.audit_kept_ids,
                    "audit_dropped_ids": bundle.audit_dropped_ids,
                    "audit_reason": bundle.audit_reason,
                    "audit_mode": bundle.audit_mode,
                    "audit_conflict_detected": bundle.audit_conflict_detected,
                    "audit_trigger_reason": bundle.audit_trigger_reason,
                    "intent_conflicts": _conflict_log,
                    "negative_matched": len(bundle.retrieved_negative_cards) if bundle.retrieved_negative_cards else 0,
                })
            except Exception as exc:
                skill_bundle_info = {"error": str(exc)}

        # When aggregation produced a markdown_narrative, negative experiences are
        # already woven into the narrative — suppress separate negative hints to
        # avoid duplication (codex review finding).
        _suppress_negative = (
            _aggregated_skill is not None and bool(_aggregated_skill.markdown_narrative)
        )

        # Create runtime with optional skill hints
        runtime = InferenceRuntime(
            backend_client=GraphBackendClient(),
            system_prompt=get_system_prompt(variant),
            followup_hint=get_prompt_variant_followup_hint(variant),
            stage_skill_hints=stage_skill_hints if stage_skill_hints else None,
            aggregated_stage_hints=agg_hints if agg_hints else None,
            retrieved_skill_cards=bundle.retrieved_cards if bundle is not None else None,
            shortlisted_skill_cards=bundle.shortlisted_cards if bundle is not None else None,
            skill_target_question=bundle.target_question if bundle is not None else "",
            negative_plan_hint="" if _suppress_negative else (bundle.negative_plan_hint if bundle is not None else ""),
            negative_action_hint="" if _suppress_negative else (bundle.negative_action_hint if bundle is not None else ""),
        )
        conversation, state = runtime.create_session(
            question=user_message,
            core_entities=raw_core_entities,
            core_relations=prompt_context["core_relations"] | set(raw_core_relations),
            available_domains=prompt_context["available_domains"],
        )

        turns = []
        predicted: List[str] = []
        error_text = ""
        consistency_traces: List[Dict[str, Any]] = []
        timeout = aiohttp.ClientTimeout(total=180.0)
        try:
            async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as backend_session:
                current_turn_budget = max_turns
                max_turn_cap = _dynamic_turn_cap(max_turns)
                turn_idx = 0
                _audit_count = 0
                MAX_AUDIT_ROUNDS = 2
                while turn_idx < current_turn_budget:
                    prepared_messages = runtime.prepare_messages(conversation, state)
                    _cumulative_frontend_errors = sum(len(getattr(t, "frontend_errors", [])) for t in turns)
                    # Skip consistency for post-audit turn to prevent loops
                    if _audit_count >= MAX_AUDIT_ROUNDS:
                        raw_response = await _call_llm(prepared_messages)
                        _turn_consistency = {"consistency_used": False, "verification_bypassed": True}
                    else:
                        raw_response, _turn_consistency = await consistent_call(
                            _call_llm,
                            prepared_messages,
                            turn_number=turn_idx + 1,
                            trajectory_context={
                                "frontend_errors": _cumulative_frontend_errors,
                                "turn_count": turn_idx,
                                "candidates": sorted(list(state.get("retrieved_candidates", set()) or [])),
                            },
                        )
                    if _turn_consistency:
                        consistency_traces.append(dict(_turn_consistency))
                    turn = await runtime.apply_model_response(
                        session=backend_session,
                        sample_id=case_id,
                        conversation=conversation,
                        state=state,
                        raw_response=raw_response,
                    )
                    turns.append(turn)

                    # Context cleanup: strip reasoning drafts from assistant messages
                    # Keeps tool calls and backend results intact
                    if os.getenv("KGQA_ENABLE_CONTEXT_CLEANUP", "0").strip().lower() in {"1", "true", "yes", "on"}:
                        for msg in conversation:
                            if msg.get("role") == "assistant":
                                content = msg.get("content", "")
                                cleaned = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL).strip()
                                if cleaned:
                                    msg["content"] = cleaned

                    if turn.frontend_errors and current_turn_budget < max_turn_cap:
                        current_turn_budget = min(max_turn_cap, current_turn_budget + 1)
                    if turn.parsed_output.get("final_answer") and not turn.frontend_errors:
                        # --- Audit Agent (replaces critique when enabled) ---
                        if is_audit_agent_enabled() and _audit_count < MAX_AUDIT_ROUNDS:
                            try:
                                # If model flagged graph evidence as insufficient,
                                # run web search and pass results to audit agent
                                _web_search_text = None
                                if state.get("graph_sufficiency") == "insufficient":
                                    try:
                                        from subgraph_kgqa.search_agent import BackgroundSearchAgent, SearchAgentConfig, SearchResultFormatter
                                        _search_cfg = SearchAgentConfig()
                                        _search_cfg.enabled = True  # force enable for sufficiency-triggered search
                                        search_agent = BackgroundSearchAgent(config=_search_cfg)
                                        search_res = await search_agent.search_kgqa(
                                            question=clean_question,
                                            candidates=set(state.get("retrieved_candidates", set())),
                                            missing_attribute="",
                                            reasoning_summary=raw_response[:500],
                                        )
                                        if search_res and search_res.results:
                                            _web_search_text = SearchResultFormatter.format_for_model(
                                                search_res, clean_question,
                                            )
                                            state["web_search_used"] = True
                                    except Exception:
                                        pass  # search failure is non-fatal

                                audit_result = await audit_final_answer(
                                    question=clean_question,
                                    final_answer=turn.parsed_output["final_answer"],
                                    candidates=turn.parsed_output.get("candidates", []),
                                    raw_response=raw_response,
                                    conversation_history=conversation,
                                    web_search_results=_web_search_text,
                                )
                                _audit_count += 1
                                skill_bundle_info["audit_agent"] = {
                                    "passed": audit_result.passed,
                                    "any_issue": audit_result.any_issue,
                                    "issues": audit_result.issues,
                                    "suggest_websearch": audit_result.suggest_websearch,
                                    "audit_failed": audit_result.audit_failed,
                                    "audit_round": _audit_count,
                                }
                                if audit_result.passed:
                                    predicted = turn.parsed_output["final_answer"]
                                    break
                                else:
                                    rereasoning_prompt = format_audit_feedback_for_rereasoning(audit_result)
                                    conversation.append({"role": "user", "content": rereasoning_prompt})
                            except Exception as exc:
                                _audit_count += 1
                                skill_bundle_info["audit_agent_error"] = str(exc)
                                predicted = turn.parsed_output["final_answer"]
                                break
                        # --- Fallback: critique mechanism (legacy) ---
                        elif _turn_consistency.get("critique_recommended") and _audit_count < MAX_AUDIT_ROUNDS:
                            critique_issues = _turn_consistency.get("critique_issues", [])
                            critique_feedback = _turn_consistency.get("critique_feedback", "")
                            issues_str = "; ".join(critique_issues) if critique_issues else "See feedback below"
                            rereasoning_prompt = (
                                f"[CRITIQUE FEEDBACK — Defects detected]\n"
                                f"Your previous answer was critiqued and found potentially problematic.\n"
                                f"Issues identified: {issues_str}\n\n"
                                f"Critique details:\n{critique_feedback[:2000]}\n\n"
                                "Please re-examine the candidates and graph evidence carefully.\n"
                                "Address each issue above. If your original answer was correct after all, "
                                "you may keep it. Otherwise, provide a corrected final answer.\n"
                            )
                            conversation.append({"role": "user", "content": rereasoning_prompt})
                            _audit_count += 1
                        else:
                            predicted = turn.parsed_output["final_answer"]
                            break
                    turn_idx += 1
        except Exception as exc:
            exc_type = type(exc).__name__
            exc_msg = str(exc) if str(exc) else "(no message)"
            error_text = f"{exc_type}: {exc_msg}"

        gt_answers = gt.get("global_truth_answers", []) or case.get("solution", [])
        f1 = calculate_f1(predicted, gt_answers)
        frontend_errors = sum(len(turn.frontend_errors) for turn in turns)

        # Print progress
        print(f"[{case_index}/{total_cases}] {case_id}: F1={f1:.2f} | Turns={len(turns)} | Skills={len(skill_bundle_info.get('retrieved_case_ids', []))}")

        # Store clean question for reporting (always extract, even without skills)
        clean_question = _extract_clean_question(user_message)

        # Serialize turn trajectories
        turn_trajectories = []
        for t_idx, turn in enumerate(turns):
            turn_consistency = consistency_traces[t_idx] if t_idx < len(consistency_traces) else {}
            turn_data: Dict[str, Any] = {
                "turn": t_idx + 1,
                "raw_response": turn.raw_response,
                "parsed_output": {
                    "queries": [
                        {"tool_name": q.get("tool_name"), "arguments": q.get("arguments")}
                        for q in turn.parsed_output.get("queries", [])
                    ],
                    "candidates": list(turn.parsed_output.get("candidates", [])),
                    "final_answer": list(turn.parsed_output.get("final_answer", [])),
                },
                "frontend_errors": [
                    {"code": getattr(e, "code", ""), "message": getattr(e, "message", str(e))}
                    for e in turn.frontend_errors
                ],
                "executed_queries": [
                    {"tool_name": q.get("tool_name"), "arguments": q.get("arguments")}
                    for q in turn.executed_queries
                ],
                "backend_results": [
                    {
                        "tool_name": getattr(r, "tool_name", ""),
                        "is_success": getattr(r, "is_success", False),
                        "status": getattr(r, "status", ""),
                        "response_text": getattr(r, "response_text", "")[:2000],
                    }
                    for r in turn.backend_results
                ],
                "feedback": turn.feedback,
                "state_snapshot": turn.state_snapshot,
                "consistency": turn_consistency,
            }
            turn_trajectories.append(turn_data)

        consistency_summary = {
            "used_turns": sum(1 for meta in consistency_traces if meta.get("consistency_used")),
            "disagreement_turns": sum(
                1
                for meta in consistency_traces
                if meta.get("consistency_used") and not meta.get("consistency_agreed_initially", True)
            ),
            "turns": consistency_traces,
        }

        # Hit@1: does the first predicted answer appear in ground truth?
        hit_at_1_val = 1.0 if (predicted and predicted[0] in gt_answers) else 0.0
        stage_scores = _extract_stage_scores(
            trajectory=turn_trajectories,
            core_entities=raw_core_entities,
            core_relations=list(prompt_context["core_relations"] | set(raw_core_relations)),
            truth_answers=gt_answers,
        )

        return {
            "case_id": case_id,
            "question": clean_question,
            "f1": f1,
            "hit_at_1": hit_at_1_val,
            "stage5_prompt_variant": stage5_prompt_variant,
            "predicted": predicted,
            "ground_truth": gt_answers,
            "turns": len(turns),
            "frontend_errors": frontend_errors,
            "repair_mode": turns[-1].state_snapshot.get("repair_mode") if turns else None,
            "error": error_text,
            "skill_bundle": skill_bundle_info,
            "intent": intent_info,
            "consistency": consistency_summary,
            "stage_scores": stage_scores,
            "trajectory": turn_trajectories,
        }


def _render_report(
    *,
    label: str,
    variant: str,
    inject_skills: bool,
    skill_top_k: int,
    stage5_prompt_variant: str,
    results: List[Dict[str, Any]],
    duration_seconds: float,
    requested_cases: int,
    failed_count: int,
) -> str:
    """Generate markdown report."""
    avg_f1 = mean(item["f1"] for item in results) if results else 0.0
    hit_at_1 = mean(item.get("hit_at_1", 0.0) for item in results) if results else 0.0
    hit_at_08 = mean(1.0 if item["f1"] >= 0.8 else 0.0 for item in results) if results else 0.0
    exact_match = mean(1.0 if item["f1"] >= 0.95 else 0.0 for item in results) if results else 0.0
    avg_turns = mean(item["turns"] for item in results) if results else 0.0
    total_frontend_errors = sum(item["frontend_errors"] for item in results)
    avg_plan_f1 = mean(item.get("stage_scores", {}).get("plan", {}).get("f1", 0.0) for item in results) if results else 0.0
    avg_action_f1 = mean(item.get("stage_scores", {}).get("action", {}).get("f1", 0.0) for item in results) if results else 0.0
    avg_reason_f1 = mean(item.get("stage_scores", {}).get("final_reasoning", {}).get("f1", 0.0) for item in results) if results else 0.0

    # Skill retrieval stats
    cases_with_skills = sum(1 for r in results if r.get("skill_bundle", {}).get("retrieved_case_ids"))
    cases_with_negative = sum(1 for r in results if r.get("skill_bundle", {}).get("negative_matched", 0) > 0)
    avg_negative_matched = mean(
        [r.get("skill_bundle", {}).get("negative_matched", 0) for r in results if r.get("skill_bundle", {}).get("negative_matched", 0) > 0]
    ) if cases_with_negative > 0 else 0.0
    avg_retrieved_skills = mean(
        len(r.get("skill_bundle", {}).get("retrieved_case_ids", []))
        for r in results
        if r.get("skill_bundle", {}).get("retrieved_case_ids")
    ) if cases_with_skills > 0 else 0.0
    cases_with_consistency = sum(1 for r in results if r.get("consistency", {}).get("used_turns", 0) > 0)
    total_consistency_turns = sum(r.get("consistency", {}).get("used_turns", 0) for r in results)
    total_consistency_disagreements = sum(r.get("consistency", {}).get("disagreement_turns", 0) for r in results)

    lines = [
        "# Skill-Enhanced Batch Test Report",
        "",
        f"- Label: `{label}`",
        f"- Variant: `{variant}`",
        f"- Skills Enabled: `{inject_skills}`",
        f"- Skill Top-K: `{skill_top_k}`",
        f"- Stage 5 Prompt Variant: `{stage5_prompt_variant}`",
        f"- Timestamp: `{datetime.now().isoformat()}`",
        f"- Duration: `{duration_seconds:.1f}s`",
        "",
        "## Configuration",
        "",
        f"- Temperature: `{os.getenv('KGQA_TEMPERATURE', '0.2')}`",
        f"- Top-P: `{os.getenv('KGQA_TOP_P', '0.8')}`",
        f"- Background Search: `{os.getenv('KGQA_ENABLE_BACKGROUND_SEARCH', '0')}`",
        f"- Graph Snapshot Date: `{os.getenv('KGQA_GRAPH_SNAPSHOT_DATE', 'N/A')}`",
        f"- Skill Audit Temperature: `{os.getenv('KGQA_SKILL_AUDIT_TEMPERATURE', '0')}`",
        f"- Skill Audit Mode: `{os.getenv('KGQA_SKILL_AUDIT_MODE', 'off')}`",
        f"- Intent Analysis: `{os.getenv('KGQA_ENABLE_INTENT_ANALYSIS', '0')}`",
        f"- Skill Aggregation: `{os.getenv('KGQA_ENABLE_SKILL_AGGREGATION', '0')}`",
        f"- Consistency Check: `{os.getenv('KGQA_ENABLE_CONSISTENCY_CHECK', '1')}`",
        f"- Context Cleanup: `{os.getenv('KGQA_ENABLE_CONTEXT_CLEANUP', '0')}`",
        "",
        "## Overall Metrics",
        "",
        f"- Cases: `{len(results)}` (requested: {requested_cases}, completed: {requested_cases - failed_count}, failed: {failed_count})",
        f"- **Avg F1: `{avg_f1:.4f}`**",
        f"- **Hit@1 (first pred in GT): `{hit_at_1:.4f}`**",
        f"- **Hit@0.8 (F1>=0.8): `{hit_at_08:.4f}`**",
        f"- **Exact Match (F1>=0.95): `{exact_match:.4f}`**",
        f"- Avg Turns: `{avg_turns:.2f}`",
        f"- Frontend Errors: `{total_frontend_errors}`",
        "",
        "## Stage Scores",
        "",
        f"- **Plan F1: `{avg_plan_f1:.4f}`**",
        f"- **Action F1: `{avg_action_f1:.4f}`**",
        f"- **Final Reasoning F1: `{avg_reason_f1:.4f}`**",
        "",
        "## Skill Retrieval Stats",
        "",
        f"- Cases with retrieved skills: `{cases_with_skills}/{len(results)}`",
        f"- Avg retrieved skills per case: `{avg_retrieved_skills:.1f}`",
        f"- Cases with negative skills: `{cases_with_negative}/{len(results)}`",
        f"- Avg negative skills matched: `{avg_negative_matched:.1f}`",
        "",
        "## Consistency Stats",
        "",
        f"- Cases using consistency: `{cases_with_consistency}/{len(results)}`",
        f"- Total consistency turns: `{total_consistency_turns}`",
        f"- Total disagreement turns: `{total_consistency_disagreements}`",
        "",
        "## Detailed Results",
        "",
        "| Case ID | F1 | Hit@1 | Plan | Action | Reason | Turns | FE | Skills | Consis | Disagree | Predicted | GT |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in sorted(results, key=lambda x: x["f1"], reverse=True):
        predicted = ", ".join(item["predicted"][:3]) if item["predicted"] else "-"
        if len(item["predicted"]) > 3:
            predicted += "..."
        gt = ", ".join(item["ground_truth"][:3]) if item["ground_truth"] else "-"
        if len(item["ground_truth"]) > 3:
            gt += "..."
        num_skills = len(item.get("skill_bundle", {}).get("retrieved_case_ids", []))
        consistency = item.get("consistency", {})
        stage_scores = item.get("stage_scores", {})
        lines.append(
            f"| {item['case_id']} | {item['f1']:.2f} | {item.get('hit_at_1', 0.0):.2f} | "
            f"{stage_scores.get('plan', {}).get('f1', 0.0):.2f} | "
            f"{stage_scores.get('action', {}).get('f1', 0.0):.2f} | "
            f"{stage_scores.get('final_reasoning', {}).get('f1', 0.0):.2f} | "
            f"{item['turns']} | {item['frontend_errors']} | {num_skills} | {consistency.get('used_turns', 0)} | "
            f"{consistency.get('disagreement_turns', 0)} | {predicted} | {gt} |"
        )

    # Failed cases
    failed = [r for r in results if r["f1"] < 0.5]
    if failed:
        lines.extend([
            "",
            "## Failed Cases (F1 < 0.5)",
            "",
            "| Case ID | F1 | Turns | Question | Predicted | GT |",
            "|---|---:|---:|---|---|---|",
        ])
        for item in sorted(failed, key=lambda x: x["f1"]):
            question = item["question"]
            predicted = ", ".join(item["predicted"][:2]) if item["predicted"] else "-"
            gt = ", ".join(item["ground_truth"][:2]) if item["ground_truth"] else "-"
            lines.append(
                f"| {item['case_id']} | {item['f1']:.2f} | {item['turns']} | "
                f"{question} | {predicted} | {gt} |"
            )

    return "\n".join(lines)


def _save_json_results(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Save detailed results as JSON for further analysis."""
    serializable_results = []
    for r in results:
        item = dict(r)
        if "ground_truth" in item and isinstance(item["ground_truth"], list):
            item["ground_truth"] = list(item["ground_truth"])
        if "predicted" in item and isinstance(item["predicted"], list):
            item["predicted"] = list(item["predicted"])
        serializable_results.append(item)

    output_path.write_text(json.dumps(serializable_results, ensure_ascii=False, indent=2))


def _save_trajectory_md(item: Dict[str, Any], traj_dir: Path) -> None:
    """Save a single case trajectory as a readable markdown file."""
    case_id = item["case_id"]
    lines = [
        f"# Trajectory: {case_id}",
        "",
        f"- Question: `{item['question']}`",
        f"- F1: `{item['f1']:.4f}`",
        f"- Hit@1: `{item.get('hit_at_1', 0.0):.4f}`",
        f"- Predicted: `{item['predicted']}`",
        f"- Ground Truth: `{item['ground_truth']}`",
        f"- Turns: `{item['turns']}`",
        f"- Frontend Errors: `{item['frontend_errors']}`",
        f"- Repair Mode: `{item.get('repair_mode') or '-'}`",
    ]
    stage_scores = item.get("stage_scores", {})
    if stage_scores:
        lines.extend([
            f"- Plan F1: `{stage_scores.get('plan', {}).get('f1', 0.0):.4f}`",
            f"- Action F1: `{stage_scores.get('action', {}).get('f1', 0.0):.4f}`",
            f"- Filter F1: `{stage_scores.get('filter', {}).get('f1', 0.0):.4f}`",
            f"- Final Reasoning F1: `{stage_scores.get('final_reasoning', {}).get('f1', 0.0):.4f}`",
        ])
    consistency_summary = item.get("consistency", {})
    if consistency_summary:
        lines.extend([
            f"- Consistency Turns: `{consistency_summary.get('used_turns', 0)}`",
            f"- Consistency Disagreements: `{consistency_summary.get('disagreement_turns', 0)}`",
        ])

    # Skill info
    skill_bundle = item.get("skill_bundle", {})
    retrieved = skill_bundle.get("retrieved_case_ids", [])
    if retrieved:
        lines.extend([
            "",
            "## Retrieved Skills",
            "",
            f"- Retrieved Case IDs: `{retrieved}`",
            f"- Retrieval Note: `{skill_bundle.get('retrieval_note', '')}`",
        ])
    elif skill_bundle.get("error"):
        lines.extend(["", f"- Skill Error: `{skill_bundle['error']}`"])

    # Turn-by-turn trajectory
    trajectory = item.get("trajectory", [])
    for turn_data in trajectory:
        lines.extend([
            "",
            f"## Turn {turn_data['turn']}",
            "",
        ])

        # Parsed output summary
        parsed = turn_data["parsed_output"]
        turn_consistency = turn_data.get("consistency", {})
        if parsed.get("final_answer"):
            lines.append(f"- **Final Answer: `{parsed['final_answer']}`**")
        if parsed.get("candidates"):
            lines.append(f"- Candidates: `{parsed['candidates'][:10]}`")
        if parsed.get("queries"):
            lines.append("- Queries:")
            for q in parsed["queries"]:
                lines.append(f"  - `{q.get('tool_name')}`: `{json.dumps(q.get('arguments', {}), ensure_ascii=False)[:300]}`")
        if turn_consistency:
            lines.append(
                f"- Consistency: used=`{turn_consistency.get('consistency_used')}` "
                f"| agreed_initially=`{turn_consistency.get('consistency_agreed_initially', '-')}` "
                f"| chosen=`{str(turn_consistency.get('consistency_chosen_signature', '-'))[:120]}`"
            )
            if turn_consistency.get("consistency_candidate_signatures"):
                lines.append(
                    f"- Consistency Candidates: `{turn_consistency.get('consistency_candidate_signatures')}`"
                )

        # Frontend errors
        if turn_data["frontend_errors"]:
            lines.append(f"- **Frontend Errors ({len(turn_data['frontend_errors'])}):**")
            for e in turn_data["frontend_errors"]:
                lines.append(f"  - [{e.get('code')}] {e.get('message')}")

        # Backend results
        if turn_data["backend_results"]:
            lines.append("- Backend Results:")
            for br in turn_data["backend_results"]:
                icon = "OK" if br["is_success"] else "FAIL"
                resp = br["response_text"][:500]
                lines.append(f"  - [{icon}] `{br['tool_name']}` ({br['status']}):")
                lines.append("    ```text")
                for rline in resp.split("\n")[:20]:
                    lines.append(f"    {rline}")
                lines.append("    ```")

        # State snapshot
        state = turn_data.get("state_snapshot", {})
        repair = state.get("repair_mode")
        candidates = state.get("retrieved_candidates")
        if repair or candidates:
            lines.append(f"- State: repair=`{repair}` | candidates=`{str(candidates)[:200] if candidates else '-'}`")

        feedback_text = turn_data["feedback"]
        skill_blocks = re.findall(
            r"\[(RETRIEVED SKILL EXPERIENCE:[^\]]+)\](.*?)(?=\n\n\[[A-Z][^\]]+\]|\Z)",
            feedback_text,
            flags=re.S,
        )
        if skill_blocks:
            lines.extend([
                "",
                "### Skill Hints",
            ])
            for title, body in skill_blocks:
                lines.extend([
                    f"#### [{title}]",
                    "```text",
                    body.strip()[:4000],
                    "```",
                ])

        # Raw response
        lines.extend([
            "",
            "### Raw Response",
            "```text",
            turn_data["raw_response"][:3000],
            "```",
            "",
            "### Feedback",
            "```text",
            feedback_text[:6000],
            "```",
        ])

    traj_path = traj_dir / f"{case_id}.md"
    traj_path.write_text("\n".join(lines))


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run batch evaluation on WebQSP test set with training skills as hints."
    )
    parser.add_argument("--data-path", default="data/webqsp/webqsp_test.jsonl", help="Path to test dataset")
    parser.add_argument("--train-data-path", default="data/webqsp/webqsp_train.jsonl", help="Path to train dataset (for skill retrieval)")
    parser.add_argument(
        "--skills-root",
        default="skills/webqsp_train_case_skills_en",
        help="Path to prebuilt skill corpus root (default: full train English skill corpus)",
    )
    parser.add_argument("--variant", default="original", help="Prompt variant to use")
    parser.add_argument("--limit-cases", type=int, default=None, help="Limit number of cases (None = all)")
    parser.add_argument("--case-id", action="append", default=[], help="Run only specific case id(s); may be repeated")
    parser.add_argument("--max-turns", type=int, default=8, help="Maximum turns per case")
    parser.add_argument("--max-concurrency", type=int, default=32, help="Maximum concurrent cases")
    parser.add_argument("--skill-top-k", type=int, default=int(os.getenv("KGQA_SKILL_TOP_K", "10")), help="Number of similar skills to retrieve (env: KGQA_SKILL_TOP_K)")
    parser.add_argument("--negative-skills-root", default="skills/webqsp_negative_skills", help="Path to negative skills directory")
    parser.add_argument(
        "--allow-online-skill-build",
        action="store_true",
        help="If a retrieved training case has no prebuilt skill card, synthesize it online instead of relying on the prebuilt corpus only.",
    )
    parser.add_argument("--label", default=None, help="Label for this run (default: auto-generated)")
    parser.add_argument("--no-skills", action="store_true", help="Disable skill retrieval (baseline)")
    parser.add_argument("--output-dir", default="reports/skill_enhanced_test", help="Output directory for reports")
    parser.add_argument(
        "--stage5-prompt-variant",
        default=os.getenv("KGQA_STAGE5_PROMPT_VARIANT", "v0_baseline"),
        help="Stage-5 prompt variant (default: env KGQA_STAGE5_PROMPT_VARIANT or v0_baseline).",
    )
    args = parser.parse_args()

    # Generate label if not provided
    if args.label is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        skills_suffix = "no_skills" if args.no_skills else f"skills_k{args.skill_top_k}"
        args.label = f"{timestamp}_{skills_suffix}"

    # Load test cases
    print(f"Loading test data from: {args.data_path}")
    cases = _load_cases(Path(args.data_path), args.limit_cases, args.case_id)
    print(f"Loaded {len(cases)} test cases")

    # Setup paths
    test_data_path = Path(args.data_path)
    skills_root = Path(args.skills_root)
    train_data_path = Path(args.train_data_path)
    if not skills_root.exists():
        raise FileNotFoundError(f"Skills root does not exist: {skills_root}")
    output_dir = Path(args.output_dir) / args.label
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup negative skills root (feature-gated)
    negative_skills_root = Path(args.negative_skills_root) if not args.no_skills and args.negative_skills_root else None
    neg_injection = os.getenv("KGQA_NEGATIVE_SKILL_INJECTION", "").strip()
    if negative_skills_root and not negative_skills_root.exists():
        print(f"WARNING: Negative skills root does not exist: {negative_skills_root}")
        negative_skills_root = None

    # Configuration summary
    print("\n=== Test Configuration ===")
    print(f"Test cases: {len(cases)}")
    print(f"Max concurrency: {args.max_concurrency}")
    print(f"Max turns: {args.max_turns}")
    print(f"Skills enabled: {not args.no_skills}")
    print(f"Stage 5 prompt variant: {args.stage5_prompt_variant}")
    if not args.no_skills:
        print(f"Skill top-k: {args.skill_top_k}")
        print(f"Skills root: {skills_root}")
        if negative_skills_root:
            print(f"Negative skills: {negative_skills_root} ({neg_injection or 'disabled'})")
        print(f"Train data for retrieval: {train_data_path}")
        print(f"Allow online skill build: {args.allow_online_skill_build}")
        embed_device = os.getenv("KGQA_SKILL_EMBED_DEVICE", "cpu").strip() or "cpu"
        print(f"Skill embed device: {embed_device}")
        if embed_device == "cpu":
            print("WARNING: KGQA_SKILL_EMBED_DEVICE is cpu; query embedding retrieval will be slow.")
    print(f"Output dir: {output_dir}")
    print("==========================\n")

    os.environ["KGQA_STAGE5_PROMPT_VARIANT"] = args.stage5_prompt_variant

    if not args.no_skills:
        print("Prewarming precomputed question embedding indexes...")
        prewarm_precomputed_embedding_indexes(
            train_data_path=train_data_path,
            query_data_path=test_data_path,
        )
        print("Prewarm complete.\n")

    # Run evaluation
    semaphore = asyncio.Semaphore(args.max_concurrency)
    start_time = asyncio.get_event_loop().time()

    tasks = [
        _run_single_case(
            case=case,
            variant=args.variant,
            query_data_path=test_data_path,
            skills_root=skills_root,
            negative_skills_root=negative_skills_root,
            train_data_path=train_data_path,
            skill_top_k=args.skill_top_k,
            inject_skills=not args.no_skills,
            allow_online_skill_build=args.allow_online_skill_build,
            max_turns=args.max_turns,
            stage5_prompt_variant=args.stage5_prompt_variant,
            semaphore=semaphore,
            case_index=i + 1,
            total_cases=len(cases),
        )
        for i, case in enumerate(cases)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions — include as f1=0 failures
    clean_results = []
    failed_count = 0
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            failed_count += 1
            case = cases[i]
            case_id = case.get("id", case.get("question", f"case_{i}"))
            print(f"[ERROR] Case {case_id} failed: {r}")
            clean_results.append({
                "case_id": case_id,
                "question": case.get("question", ""),
                "f1": 0.0,
                "hit_at_1": 0.0,
                "predicted": [],
                "ground_truth": case.get("answer", []),
                "turns": 0,
                "frontend_errors": 0,
                "skill_count": 0,
                "consistency_turns": 0,
                "disagreement_turns": 0,
                "error": str(r),
                "is_failed": True,
            })
        else:
            clean_results.append(r)

    duration = asyncio.get_event_loop().time() - start_time

    # Generate and save reports
    print("\n=== Evaluation Complete ===")
    print(f"Completed {len(clean_results)}/{len(cases)} cases")
    print(f"Duration: {duration:.1f}s ({duration/len(cases):.1f}s per case)")

    # Markdown report
    report_text = _render_report(
        label=args.label,
        variant=args.variant,
        inject_skills=not args.no_skills,
        skill_top_k=args.skill_top_k,
        stage5_prompt_variant=args.stage5_prompt_variant,
        results=clean_results,
        duration_seconds=duration,
        requested_cases=len(cases),
        failed_count=failed_count,
    )
    report_path = output_dir / "report.md"
    report_path.write_text(report_text)
    print(f"Report saved to: {report_path}")

    # JSON results
    json_path = output_dir / "results.json"
    _save_json_results(clean_results, json_path)
    print(f"JSON results saved to: {json_path}")

    # Per-case trajectory reports
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    for item in clean_results:
        _save_trajectory_md(item, traj_dir)
    print(f"Trajectories saved to: {traj_dir}/ ({len(clean_results)} files)")

    # Print summary
    avg_f1 = mean(item["f1"] for item in clean_results) if clean_results else 0.0
    hit_at_1 = mean(item.get("hit_at_1", 0.0) for item in clean_results) if clean_results else 0.0
    print("\nFinal Metrics:")
    print(f"  Avg F1: {avg_f1:.4f}")
    print(f"  Hit@1 (first pred in GT): {hit_at_1:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
