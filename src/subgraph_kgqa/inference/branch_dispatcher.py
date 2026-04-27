"""
Branch Dispatcher for Multi-Sub-Question Parallel Execution.

When a CWQ (Complex WebQuestion) contains multiple sub-questions,
this module forks each sub-question into an independent pipeline
instance after the plan phase.

Flow:
1. Original question enters pipeline normally
2. After plan phase, if multiple sub-questions detected, fork branches
3. Each branch runs independently through InferenceRuntime
4. Aggregator combines final answers
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

import aiohttp

from .runtime import InferenceRuntime

logger = logging.getLogger(__name__)

__all__ = [
    "BranchConfig",
    "BranchResult",
    "BranchDispatcher",
    "AnswerAggregator",
    "run_multi_branch",
]


@dataclass
class BranchConfig:
    """Configuration for a single branch execution."""

    sub_question: str
    branch_id: str
    shared_entities: Set[str]
    shared_relations: Set[str]
    shared_domains: Set[str]


@dataclass
class BranchResult:
    """Result from a single branch execution."""

    branch_id: str
    sub_question: str
    answer: List[str]
    f1: float
    turns_used: int
    state_snapshot: Dict[str, Any]


class BranchDispatcher:
    """
    Dispatches multiple sub-questions to independent pipeline instances.

    Each branch runs in parallel with its own state and conversation,
    but shares seed context (entities, relations, domains) from the
    discovery phase.
    """

    def __init__(
        self,
        runtime_factory: Callable[[], InferenceRuntime],
        *,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        """
        Initialize the dispatcher.

        Parameters
        ----------
        runtime_factory : callable
            A callable that creates InferenceRuntime instances.
            Each branch gets its own runtime instance.
        session : aiohttp.ClientSession, optional
            Shared HTTP session. If provided, all branches reuse it.
            If None, each branch creates its own session.
        """
        self.runtime_factory = runtime_factory
        self._shared_session = session

    async def dispatch(
        self,
        question: str,
        sub_questions: List[str],
        shared_context: Dict[str, Any],
        max_turns_per_branch: int = 8,
        semaphore: Optional[asyncio.Semaphore] = None,
        branch_timeout_seconds: float = 600.0,
    ) -> List[BranchResult]:
        """
        Fork and run branches in parallel.

        Parameters
        ----------
        question : str
            The original complex question.
        sub_questions : list of str
            Sub-questions identified during planning.
        shared_context : dict
            Shared context from discovery phase with keys:
            - core_entities: set of entities
            - core_relations: set of relations
            - available_domains: set of domains
        max_turns_per_branch : int
            Maximum inference turns per branch (default: 8).
        semaphore : asyncio.Semaphore, optional
            Concurrency control. If None, creates one capped at 4.
        branch_timeout_seconds : float
            Hard timeout per branch in seconds (default: 600).
        """
        if semaphore is None:
            semaphore = asyncio.Semaphore(min(len(sub_questions), 4))

        # Build branch configs
        branches = [
            BranchConfig(
                sub_question=sub_q,
                branch_id=f"branch_{idx}",
                shared_entities=set(shared_context.get("core_entities", set())),
                shared_relations=set(shared_context.get("core_relations", set())),
                shared_domains=set(shared_context.get("available_domains", set())),
            )
            for idx, sub_q in enumerate(sub_questions)
        ]

        # Run branches in parallel
        tasks = [
            self._run_single_branch(
                config=branch,
                max_turns=max_turns_per_branch,
                semaphore=semaphore,
                timeout_seconds=branch_timeout_seconds,
            )
            for branch in branches
        ]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except BaseException as exc:
            logger.exception("Unexpected error in gather: %s", exc)
            results = [exc] * len(tasks)

        # Process results, converting exceptions to error results
        processed_results: List[BranchResult] = []
        all_failed = True
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Branch %s failed: %s", branches[idx].branch_id, result
                )
                processed_results.append(
                    BranchResult(
                        branch_id=branches[idx].branch_id,
                        sub_question=sub_questions[idx],
                        answer=[],
                        f1=0.0,
                        turns_used=0,
                        state_snapshot={"error": str(result)},
                    )
                )
            else:
                all_failed = False
                processed_results.append(result)

        if all_failed:
            logger.error(
                "All %d branches failed for question: %s",
                len(sub_questions),
                question[:100],
            )

        return processed_results

    async def _run_single_branch(
        self,
        config: BranchConfig,
        max_turns: int,
        semaphore: asyncio.Semaphore,
        timeout_seconds: float = 600.0,
    ) -> BranchResult:
        """
        Run a single branch to completion with hard timeout.
        """
        async with semaphore:
            try:
                return await asyncio.wait_for(
                    self._execute_branch(config, max_turns),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Branch %s timed out after %.0fs",
                    config.branch_id,
                    timeout_seconds,
                )
                return BranchResult(
                    branch_id=config.branch_id,
                    sub_question=config.sub_question,
                    answer=[],
                    f1=0.0,
                    turns_used=0,
                    state_snapshot={"error": f"timeout after {timeout_seconds}s"},
                )

    async def _execute_branch(
        self,
        config: BranchConfig,
        max_turns: int,
    ) -> BranchResult:
        """Core branch execution logic (no timeout wrapper)."""
        from subgraph_kgqa.llm_client import call_llm
        from subgraph_kgqa.rl.legacy_fallback import calculate_f1

        runtime = self.runtime_factory()
        conversation, state = runtime.create_session(
            question=config.sub_question,
            core_entities=config.shared_entities,
            core_relations=config.shared_relations,
            available_domains=config.shared_domains,
        )

        predicted: List[str] = []
        turns_used = 0
        final_state_snapshot: Dict[str, Any] = {}

        # Use shared session if available, else create per-branch
        own_session = False
        session = self._shared_session
        if session is None:
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=180.0), trust_env=False
            )
            own_session = True

        try:
            for turn_idx in range(max_turns):
                prepared_messages = runtime.prepare_messages(conversation, state)
                raw_response = await call_llm(prepared_messages, session=session)

                turn_result = await runtime.apply_model_response(
                    session=session,
                    sample_id=config.branch_id,
                    conversation=conversation,
                    state=state,
                    raw_response=raw_response,
                )

                turns_used = turn_idx + 1
                final_state_snapshot = turn_result.state_snapshot

                if turn_result.parsed_output.get("final_answer") and not turn_result.frontend_errors:
                    predicted = turn_result.parsed_output["final_answer"]
                    break

        except Exception as exc:
            logger.error("Branch %s execution error: %s", config.branch_id, exc)
            final_state_snapshot = {"error": str(exc)}
        finally:
            if own_session:
                await session.close()

        # Calculate F1 if ground truth available
        f1 = 0.0
        if "ground_truth_answers" in final_state_snapshot:
            gt_answers = final_state_snapshot["ground_truth_answers"]
            f1 = calculate_f1(predicted, gt_answers)

        return BranchResult(
            branch_id=config.branch_id,
            sub_question=config.sub_question,
            answer=predicted,
            f1=f1,
            turns_used=turns_used,
            state_snapshot=final_state_snapshot,
        )


class AnswerAggregator:
    """
    Combines answers from multiple branches.

    Current strategy: union of all branch answers.
    This is the safest default for independent sub-questions.
    """

    def aggregate(
        self,
        question: str,
        branch_results: List[BranchResult],
    ) -> List[str]:
        """
        Combine branch answers into final answer set.

        Default strategy: union of all branch answers.
        """
        if not branch_results:
            return []

        all_answers: Set[str] = set()
        for result in branch_results:
            all_answers.update(result.answer)

        return sorted(list(all_answers))


async def run_multi_branch(
    runtime: InferenceRuntime,
    session: aiohttp.ClientSession,
    question: str,
    sub_questions: List[str],
    shared_context: Dict[str, Any],
    max_turns_per_branch: int = 8,
    max_concurrency: int = 4,
    branch_timeout_seconds: float = 600.0,
) -> List[str]:
    """
    Run multiple branches and return aggregated answers.

    This is a convenience wrapper for the full multi-branch flow.
    """
    # Create dispatcher with runtime factory and shared session
    def runtime_factory() -> InferenceRuntime:
        return InferenceRuntime(
            backend_client=runtime.backend,
            system_prompt=runtime.system_prompt,
            followup_hint=runtime.followup_hint,
            stage_skill_hints=runtime.stage_skill_hints,
            retrieved_skill_cards=runtime.retrieved_skill_cards,
            skill_target_question=runtime.skill_target_question,
            negative_plan_hint=runtime.negative_plan_hint,
            negative_action_hint=runtime.negative_action_hint,
            aggregated_stage_hints=runtime.aggregated_stage_hints,
        )

    dispatcher = BranchDispatcher(
        runtime_factory=runtime_factory,
        session=session,
    )
    aggregator = AnswerAggregator()

    # Run branches
    semaphore = asyncio.Semaphore(max_concurrency)
    branch_results = await dispatcher.dispatch(
        question=question,
        sub_questions=sub_questions,
        shared_context=shared_context,
        max_turns_per_branch=max_turns_per_branch,
        semaphore=semaphore,
        branch_timeout_seconds=branch_timeout_seconds,
    )

    # Aggregate answers
    return aggregator.aggregate(
        question=question,
        branch_results=branch_results,
    )
