"""Orchestrate k repeated inference runs for skill instability analysis.

This module runs a given case (or list of cases) through the inference pipeline
k times, collects RawAttemptRecord objects from each run, passes them to the
k_run_analyzer, and saves the resulting enriched CaseSkillCard.

It reuses the existing InferenceRuntime and GraphBackendClient infrastructure.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from .k_run_analyzer import analyze_k_runs
from ._helpers import (
    _extract_question_surface,
    _safe_slug,
    _utc_now,
)
from .schemas import (
    CaseSkillCard,
    RawAttemptRecord,
    read_json,
    write_json,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ReplayConfig:
    """Configuration for a k-run replay session."""

    def __init__(
        self,
        *,
        case_ids: List[str],
        k: int = 5,
        variant: str = "replay",
        data_path: Path | str | None = None,
        skills_root: Path | str | None = None,
        output_root: Path | str | None = None,
        max_turns: int = 12,
        concurrency: int = 4,
        kg_api_url: str | None = None,
        llm_api_url: str | None = None,
        llm_api_key: str | None = None,
        llm_model_name: str | None = None,
    ) -> None:
        self.case_ids = list(case_ids)
        self.k = max(1, int(k))
        self.variant = variant or "replay"
        self.data_path = Path(data_path) if data_path else None
        self.skills_root = Path(skills_root) if skills_root else None
        self.output_root = Path(output_root) if output_root else (self.skills_root or Path("skills"))
        self.max_turns = max_turns
        self.concurrency = max(1, concurrency)
        self.kg_api_url = (kg_api_url or os.getenv("KGQA_KG_API_URL", "http://localhost:8001")).rstrip("/")
        self.llm_api_url = (llm_api_url or os.getenv("KGQA_LLM_API_URL", "http://127.0.0.1:8000/v1")).rstrip("/")
        self.llm_api_key = llm_api_key or os.getenv("KGQA_LLM_API_KEY", "EMPTY")
        self.llm_model_name = llm_model_name or os.getenv("KGQA_MODEL_NAME", "qwen35-9b-local")


def _load_dataset_lookup(data_path: Path) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    with data_path.open() as handle:
        for line in handle:
            case = json.loads(line)
            lookup[str(case.get("id", ""))] = case
    return lookup


async def _call_llm(
    messages: List[Dict[str, str]],
    *,
    session: aiohttp.ClientSession,
) -> str:
    from ..llm_client import call_llm
    return await call_llm(messages, max_tokens=1400, session=session)


def _compute_f1(predicted: List[str], ground_truth: List[str]) -> float:
    """Token-level F1 between predicted and ground truth answer sets."""
    pred_tokens = set()
    for answer in predicted:
        pred_tokens.update(answer.strip().lower().split())
    gt_tokens = set()
    for answer in ground_truth:
        gt_tokens.update(answer.strip().lower().split())
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    overlap = len(pred_tokens & gt_tokens)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Single inference pass using InferenceRuntime
# ---------------------------------------------------------------------------

async def _run_single_inference(
    *,
    case: Dict[str, Any],
    case_id: str,
    variant: str,
    config: ReplayConfig,
    session: aiohttp.ClientSession,
) -> RawAttemptRecord:
    """Run a single inference pass through the InferenceRuntime and collect a RawAttemptRecord."""
    from subgraph_kgqa.inference.backend import DEFAULT_ENDPOINTS, GraphBackendClient
    from subgraph_kgqa.inference.runtime import InferenceRuntime
    from subgraph_kgqa.rl.plugin import PHASE1_HINT, parse_prompt_context

    question_text = next(
        (msg["content"] for msg in case.get("messages", []) if msg.get("role") == "user"),
        "",
    )
    question_surface = _extract_question_surface(question_text)
    prompt_context = parse_prompt_context(question_text)
    gt = case.get("ground_truth", {}) or {}
    gt_answers = list(gt.get("global_truth_answers", []) or case.get("solution", []))
    core_entities = list(gt.get("core_entities", []))
    core_relations = list(gt.get("core_relations", []))

    backend = GraphBackendClient(endpoints=DEFAULT_ENDPOINTS)

    # Build a minimal system prompt for the inference runtime
    system_prompt = (
        "You are a KGQA agent. Answer the question using the available graph tools.\n"
        "Use <act>...</act> blocks to call tools, and <answer>...</answer> to give your final answer."
    )

    runtime = InferenceRuntime(
        backend_client=backend,
        system_prompt=system_prompt,
    )

    conversation, state = runtime.create_session(
        question=question_text,
        core_entities=core_entities,
        core_relations=core_relations,
        phase1_hint=PHASE1_HINT,
    )

    explored_domains: List[str] = []
    planned_relations: List[str] = []
    constraint_relations: List[str] = []
    constraint_entities: List[str] = []
    predicted_answers: List[str] = []
    total_frontend_errors = 0
    turns = 0
    _has_final_answer = False

    for turn_idx in range(config.max_turns):
        messages = runtime.prepare_messages(conversation, state)
        try:
            raw_response = await _call_llm(
                messages,
                session=session,
            )
        except Exception:
            break

        turn_result = await runtime.apply_model_response(
            session=session,
            sample_id=case_id,
            conversation=conversation,
            state=state,
            raw_response=raw_response,
        )
        turns += 1
        total_frontend_errors += len(turn_result.frontend_errors)

        # Collect exploration data from parsed output and backend results
        parsed = turn_result.parsed_output
        for query in parsed.get("queries", []):
            args = query.get("arguments", {})
            tool = query.get("tool_name", "")
            if tool == "explore_schema":
                pattern = args.get("pattern", "")
                if pattern and pattern not in explored_domains:
                    explored_domains.append(pattern)
            if tool in {"plan", "plan_subquestion"}:
                for rel in args.get("related", []):
                    if rel and rel not in planned_relations:
                        planned_relations.append(rel)
                for rel in args.get("maybe_related", []):
                    if rel and rel not in planned_relations:
                        planned_relations.append(rel)
                for rel in args.get("constraint_relations", []):
                    if rel and rel not in constraint_relations:
                        constraint_relations.append(rel)
                for ent in args.get("constraint_entities", []):
                    if ent and ent not in constraint_entities:
                        constraint_entities.append(ent)

        # Also collect from backend results
        for result in turn_result.backend_results:
            if getattr(result, "tool_name", "") == "explore_schema":
                for entity in getattr(result, "found_entities", []):
                    if entity and entity not in explored_domains:
                        explored_domains.append(entity)

        if turn_result.parsed_output.get("final_answer"):
            predicted_answers = turn_result.parsed_output["final_answer"]
            break

        if state.get("has_final_answer"):
            # Final answer may have been set by the runtime even without explicit parse
            if not predicted_answers:
                predicted_answers = turn_result.parsed_output.get("final_answer", [])
            break

    f1_score = _compute_f1(predicted_answers, gt_answers)
    success = f1_score >= 1.0

    return RawAttemptRecord(
        record_id=f"replay-{case_id}-{_safe_slug(variant)}-{uuid.uuid4().hex[:8]}",
        created_at=_utc_now(),
        case_id=case_id,
        variant=variant,
        question_text=question_surface,
        ground_truth_answers=gt_answers,
        predicted_answers=predicted_answers,
        predicted_text=", ".join(predicted_answers),
        f1=f1_score,
        success=success,
        turns=turns,
        frontend_errors=total_frontend_errors,
        repair_mode=None,
        error_text="",
        question_fields={
            "wh_word": "",
            "lexical_cues": [],
            "anchor_pattern": "multi_anchor" if len(core_entities) > 1 else "single_anchor",
            "has_temporal_cue": False,
        },
        prompt_context={
            "available_domains": sorted(list(prompt_context.get("available_domains", set()))),
            "core_relations": sorted(list(prompt_context.get("core_relations", set()))),
        },
        explored_domains=explored_domains,
        planned_relations=planned_relations,
        candidate_constraint_relations=constraint_relations,
        candidate_constraint_entities=constraint_entities,
    )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

async def run_k_replay(config: ReplayConfig) -> Dict[str, Any]:
    """Run k repeated inference passes for each configured case_id.

    Parameters
    ----------
    config : ReplayConfig
        Full configuration including case_ids, k, variant, paths, etc.

    Returns
    -------
    dict
        Summary with per-case results and enriched card paths.
    """
    if config.data_path is None:
        raise ValueError("data_path is required in ReplayConfig")
    dataset_lookup = _load_dataset_lookup(config.data_path)

    semaphore = asyncio.Semaphore(config.concurrency)
    timeout = aiohttp.ClientTimeout(total=300.0)
    results: Dict[str, Any] = {
        "config": {
            "k": config.k,
            "variant": config.variant,
            "case_count": len(config.case_ids),
            "concurrency": config.concurrency,
        },
        "cases": {},
    }

    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:

        async def _process_case(case_id: str) -> Dict[str, Any]:
            case = dataset_lookup.get(case_id)
            if case is None:
                return {"case_id": case_id, "error": "case not found in dataset"}

            async with semaphore:
                runs: List[RawAttemptRecord] = []
                for run_idx in range(config.k):
                    try:
                        record = await _run_single_inference(
                            case=case,
                            case_id=case_id,
                            variant=f"{config.variant}_run{run_idx}",
                            config=config,
                            session=session,
                        )
                        runs.append(record)
                    except Exception as exc:
                        # Record a failed attempt
                        runs.append(RawAttemptRecord(
                            record_id=f"replay-{case_id}-failed-{run_idx}-{uuid.uuid4().hex[:6]}",
                            created_at=_utc_now(),
                            case_id=case_id,
                            variant=f"{config.variant}_run{run_idx}",
                            question_text=_extract_question_surface(
                                next((msg["content"] for msg in case.get("messages", []) if msg.get("role") == "user"), "")
                            ),
                            ground_truth_answers=list(
                                (case.get("ground_truth", {}) or {}).get("global_truth_answers", [])
                            ),
                            predicted_answers=[],
                            predicted_text="",
                            f1=0.0,
                            success=False,
                            turns=0,
                            frontend_errors=0,
                            repair_mode=None,
                            error_text=str(exc)[:500],
                        ))

            # Load an existing base card if available
            base_card: Optional[CaseSkillCard] = None
            if config.skills_root:
                from .case_skill import _resolve_case_skill_output_root
                case_skill_root = _resolve_case_skill_output_root(config.skills_root)
                existing_path = case_skill_root / f"{case_id}.json"
                if existing_path.exists():
                    try:
                        payload = read_json(existing_path)
                        base_card = CaseSkillCard(**payload)
                    except Exception:
                        pass

            # Analyze the k runs
            enriched_card = analyze_k_runs(
                case_id=case_id,
                runs=runs,
                base_card=base_card,
            )

            # Save results
            case_output_dir = config.output_root / "k_run_results" / case_id
            case_output_dir.mkdir(parents=True, exist_ok=True)

            # Save individual run records
            for record in runs:
                run_path = case_output_dir / f"{record.record_id}.json"
                write_json(run_path, record)

            # Save enriched card
            enriched_path = case_output_dir / f"{case_id}_enriched_card.json"
            write_json(enriched_path, enriched_card)

            # Also upsert into the case_skills store if skills_root is configured
            upsert_paths: Dict[str, str] = {}
            if config.skills_root:
                from .case_skill import upsert_case_skill_outputs
                upsert_paths = upsert_case_skill_outputs(config.skills_root, enriched_card)

            return {
                "case_id": case_id,
                "run_count": len(runs),
                "successful_runs": sum(1 for r in runs if r.success),
                "failed_runs": sum(1 for r in runs if not r.success),
                "instability_score": enriched_card.instability_score,
                "common_misreadings": enriched_card.common_misreadings,
                "instability_triggers": enriched_card.instability_triggers,
                "enriched_card_path": str(enriched_path),
                "upsert_paths": upsert_paths,
            }

        # Process all cases concurrently (bounded by semaphore)
        case_tasks = [_process_case(case_id) for case_id in config.case_ids]
        case_results = await asyncio.gather(*case_tasks, return_exceptions=True)

    for result in case_results:
        if isinstance(result, Exception):
            case_id = getattr(result, "case_id", "unknown")
            results["cases"][case_id] = {"error": str(result)}
        else:
            results["cases"][result["case_id"]] = result

    # Save a summary
    summary_path = config.output_root / "k_run_results" / "replay_summary.json"
    write_json(summary_path, results)

    return results


def run_k_replay_sync(config: ReplayConfig) -> Dict[str, Any]:
    """Synchronous wrapper for run_k_replay."""
    return asyncio.run(run_k_replay(config))
