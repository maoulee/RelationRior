from __future__ import annotations

from dataclasses import dataclass, replace
import os
import re
from typing import Any, Dict, Iterable, List, Optional

import aiohttp

from subgraph_kgqa.skill_mining.aggregator import (
    aggregate_skills_to_markdown,
    extract_action_stage_guidance,
    extract_final_stage_guidance,
    get_aggregation_mode,
)
from subgraph_kgqa.skill_mining.skill_aggregator import is_skill_aggregation_enabled
from subgraph_kgqa.skill_mining.retriever import (
    build_action_stage_hint,
    build_classified_discovery_hint,
    build_reasoning_stage_hint,
    build_relation_stage_hint,
    filter_skills_post_plan,
)
from subgraph_kgqa.skill_mining.schemas import CaseSkillCard
from subgraph_kgqa.rl.plugin import (
    BackendChecker,
    ErrorFeedback,
    FrontendValidator,
    PhaseHintGenerator,
    PHASE1_HINT,
    RepairPolicy,
)

from .backend import GraphBackendClient
from .hints import OneShotStageHintManager
from .history import compress_history_messages, prepare_conversation_for_llm
from .parser import InferenceOutputParser
from .state import (
    _relation_aliases,
    build_initial_state,
    normalize_query_relations,
    promote_stable_state,
    snapshot_state,
    update_state_from_backend_results,
)
from subgraph_kgqa.search_agent import BackgroundSearchAgent, SearchResultFormatter, SearchAgentConfig


def _graph_snapshot_date() -> str:
    return os.getenv("KGQA_GRAPH_SNAPSHOT_DATE", "").strip()


def _dedupe_feedback_parts(parts: List[str]) -> List[str]:
    deduped: List[str] = []
    seen: set[str] = set()
    for part in parts:
        normalized = str(part or "").strip()
        if not normalized or normalized in seen:
            continue
        deduped.append(normalized)
        seen.add(normalized)
    return deduped


def _skill_pre_plan_injection_enabled() -> bool:
    """When True, use legacy pre-plan skill injection (hints during planning).

    Default is False (KGQA_SKILL_PRE_PLAN_INJECTION=0), meaning skills are
    consumed AFTER plan generation, not before.
    """
    return os.getenv("KGQA_SKILL_PRE_PLAN_INJECTION", "0").strip() in {"1", "true", "yes", "on"}


def _extract_missing_attribute(parsed: Dict, state: Dict) -> str:
    """Extract missing attribute from model reasoning."""
    import re
    reasoning = parsed.get("reasoning", "")
    if not reasoning:
        reasoning = str(parsed.get("queries", ""))
    reasoning_lower = reasoning.lower()

    # Pattern: "missing: X" or "cannot find X"
    match = re.search(
        r'(?:missing|cannot find|cannot determine|lack|lacking)\s*[:\-]?\s*([a-z_ ]{2,30})',
        reasoning_lower
    )
    if match:
        return match.group(1).strip()

    # Infer from question type
    question = str(state.get("prompt_context", {}).get("original_question", "")).lower()
    if any(w in question for w in ["current", "latest", "present", "now"]):
        if any(w in question for w in ["team", "club", "play"]):
            return "current team"
        if any(w in question for w in ["coach", "manager"]):
            return "current coach"

    return "relevant attribute"


def _detect_truncation(raw_response: str, parsed: Dict[str, Any]) -> bool:
    """Detect whether the model output was truncated mid-tag or mid-expression."""
    if not raw_response:
        return False
    # Already has valid output — not a truncation issue
    if parsed.get("queries") or parsed.get("final_answer"):
        return False
    text = raw_response.strip()
    # Check for unclosed XML-style tags that the parser looks for
    open_tags = ["<act>", "<query>", "<answer>", "<reasoning>", "<candidates>"]
    for tag in open_tags:
        close_tag = tag.replace("<", "</")
        if tag in text and close_tag not in text:
            return True
    # Check for mid-tag ending: line ends with '<' or partial tag
    last_80 = text[-80:] if len(text) > 80 else text
    if re.search(r'<[a-zA-Z]?$', last_80):
        return True
    # Check for unclosed parentheses (function call truncation)
    open_parens = text.count('(') - text.count(')')
    if open_parens > 0 and '<query>' in text:
        return True
    # Check for mid-string truncation: ends with backslash or partial escape
    if text.endswith('\\') or text.endswith(',"') or text.endswith("='"):
        return True
    return False


@dataclass
class InferenceTurnResult:
    raw_response: str
    parsed_output: Dict[str, Any]
    frontend_errors: List[Any]
    executed_queries: List[Dict[str, Any]]
    backend_results: List[Any]
    feedback: str
    state_snapshot: Dict[str, Any]


class InferenceRuntime:
    """
    Standalone inference runtime.

    Responsibilities:
    - maintain state
    - inject stage/repair hints once
    - parse model output
    - execute graph backend tools
    - compress history before the next model call

    Non-responsibilities:
    - dataset evaluation
    - report aggregation
    """

    def __init__(
        self,
        *,
        backend_client: Optional[GraphBackendClient] = None,
        system_prompt: str,
        followup_hint: str = "",
        stage_skill_hints: Optional[Dict[str, str]] = None,
        retrieved_skill_cards: Optional[Iterable[CaseSkillCard]] = None,
        shortlisted_skill_cards: Optional[Iterable[CaseSkillCard]] = None,
        skill_target_question: str = "",
        negative_plan_hint: str = "",
        negative_action_hint: str = "",
        aggregated_stage_hints: Optional[Dict[str, str]] = None,
    ) -> None:
        self.backend = backend_client or GraphBackendClient()
        self.system_prompt = system_prompt
        self.followup_hint = followup_hint.strip()
        self.stage_skill_hints = {
            str(key): str(value).strip()
            for key, value in (stage_skill_hints or {}).items()
            if str(value or "").strip()
        }
        self.retrieved_skill_cards = list(retrieved_skill_cards or [])
        self.shortlisted_skill_cards = list(shortlisted_skill_cards or [])
        self.skill_target_question = skill_target_question.strip()
        self.negative_plan_hint = negative_plan_hint.strip()
        self.negative_action_hint = negative_action_hint.strip()
        self.aggregated_stage_hints = aggregated_stage_hints or {}
        self._has_precomputed_aggregated_hints = bool(self.aggregated_stage_hints)
        self.plan_relations: List[str] = []
        self._aggregated_markdown: str = ""

    def _filter_skill_cards_to_observed_relations(self, state: Dict[str, Any]) -> List[CaseSkillCard]:
        from subgraph_kgqa.skill_mining.retriever import _extract_relation_tokens
        observed = {
            relation
            for relation in (state.get("observed_schema_relations", set()) or set())
            if isinstance(relation, str) and relation
        }
        if not observed:
            return []
        observed_aliases = set()
        for relation in observed:
            observed_aliases.update(_relation_aliases(relation))
        filtered_cards: List[CaseSkillCard] = []
        for card in self.retrieved_skill_cards:
            surviving_relations = [
                relation
                for relation in (card.core_relations or [])
                if _relation_aliases(relation) & observed_aliases
            ]
            if not surviving_relations:
                continue
            filtered_cards.append(
                replace(
                    card,
                    core_relations=surviving_relations,
                    core_relation_domains=sorted({rel.split(".", 1)[0] for rel in surviving_relations if "." in rel}),
                )
            )
        return filtered_cards

    @staticmethod
    def _has_filter_evidence(state: Dict[str, Any]) -> bool:
        tool_results = state.get("tool_structured_results", {}) or {}
        return any(
            item.get("tool_name") == "filter"
            for items in tool_results.values()
            for item in (items or [])
            if isinstance(item, dict)
        )

    @staticmethod
    def _pending_filter_relations(state: Dict[str, Any]) -> List[str]:
        current_candidates = {
            str(candidate).strip()
            for candidate in (state.get("retrieved_candidates", set()) or set())
            if str(candidate).strip()
        }
        if len(current_candidates) < 2:
            return []
        for block in reversed(state.get("filter_suggestion_blocks", []) or []):
            block_candidates = {
                str(candidate).strip()
                for candidate in (block.get("candidates", []) or [])
                if str(candidate).strip()
            }
            if current_candidates and current_candidates.issubset(block_candidates):
                relations = [str(rel).strip() for rel in (block.get("suggested_relations", []) or []) if str(rel).strip()]
                if relations:
                    return relations
        return []

    @staticmethod
    def _stable_state_repair_hint(state: Dict[str, Any]) -> str:
        stable = state.get("stable_state", {}) or {}
        stable_candidates = sorted(list(stable.get("retrieved_candidates", set()) or set()))
        stable_action_id = stable.get("last_selected_action_id")
        stable_reason = str(stable.get("reason", "")).strip()
        if not stable_candidates and not stable_action_id:
            return ""
        parts = ["[STABLE STATE REMINDER]"]
        if stable_reason:
            parts.append(f"Last preserved checkpoint reason: {stable_reason}")
        if stable_action_id:
            parts.append(f"Previously validated action_id: {stable_action_id}")
        if stable_candidates:
            preview = ", ".join(stable_candidates[:8])
            parts.append(
                "Do not rediscover from scratch if unnecessary. Reuse the preserved candidate/action context unless the current repair explicitly requires changing it."
            )
            parts.append(f"Previously preserved candidates: {preview}")
        return "\n".join(parts)

    @staticmethod
    def _handle_stage_decision(parsed: Dict[str, Any], state: Dict[str, Any]) -> str:
        decision = str(parsed.get("stage_decision", "") or "").strip().lower()

        if state.get("awaiting_action_judgment"):
            if decision == "retry_action":
                failed_id = state.get("last_selected_action_id")
                if failed_id:
                    state["failed_action_ids"] = state.get("failed_action_ids", set()) | {failed_id}
                state["retrieved_candidates"] = set()
                state["candidates_collected"] = False
                state["has_final_answer"] = False
                state["awaiting_action_judgment"] = False
                state["awaiting_filter_decision"] = False
                state["repair_mode"] = "backtrack"
                parsed["queries"] = []
                parsed["final_answer"] = []
                return PhaseHintGenerator.generate(state)

            if decision == "proceed":
                if "candidates" not in parsed:
                    parsed["queries"] = []
                    parsed["final_answer"] = []
                    return (
                        "[ACTION RESULT JUDGMENT REQUIRED]\n"
                        "You chose `proceed`, but you did not provide a `<candidates>` block.\n"
                        "When proceeding, you must list the candidates to carry forward."
                    )
                state["awaiting_action_judgment"] = False
                if InferenceRuntime._pending_filter_relations(state) and not InferenceRuntime._has_filter_evidence(state):
                    state["awaiting_filter_decision"] = True
                    parsed["queries"] = []
                    parsed["final_answer"] = []
                    return PhaseHintGenerator.generate(state)
                parsed["queries"] = []
                parsed["final_answer"] = []
                return PhaseHintGenerator.generate(state)

            parsed["queries"] = []
            parsed["final_answer"] = []
            return (
                "[ACTION RESULT JUDGMENT REQUIRED]\n"
                "You must explicitly choose one decision for the current action result:\n"
                "- `<decision>retry_action</decision>`\n"
                "- `<decision>proceed</decision>` plus `<candidates>`"
            )

        if state.get("awaiting_filter_decision"):
            if any(query.get("tool_name") == "filter" for query in (parsed.get("queries", []) or [])):
                state["awaiting_filter_decision"] = False
                return ""
            if decision == "continue":
                state["awaiting_filter_decision"] = False
                state["filter_stage_waived_action_id"] = state.get("last_selected_action_id")
                parsed["queries"] = []
                parsed["final_answer"] = []
                return PhaseHintGenerator._stage5_reasoning(state)
            parsed["queries"] = []
            parsed["final_answer"] = []
            return (
                "[FILTER STAGE REQUIRED]\n"
                "Suggested filter relations are available.\n"
                "You must either call `filter(...)` or output `<decision>continue</decision>`."
            )

        return ""

    def create_session(
        self,
        *,
        question: str,
        core_entities: Optional[Iterable[str]] = None,
        core_relations: Optional[Iterable[str]] = None,
        available_domains: Optional[Iterable[str]] = None,
        phase1_hint: str = PHASE1_HINT,
    ) -> tuple[List[Dict[str, str]], Dict[str, Any]]:
        state = build_initial_state(
            core_entities=core_entities,
            core_relations=core_relations,
            available_domains=available_domains,
        )
        user_message = f"{question}\n\n{phase1_hint}"
        if self.followup_hint:
            user_message = f"{user_message}\n\n{self.followup_hint}"
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        return conversation, state

    def prepare_messages(
        self,
        conversation: Iterable[Dict[str, Any]],
        state: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        compressed = compress_history_messages(conversation, state)
        return prepare_conversation_for_llm(compressed)

    async def apply_model_response(
        self,
        *,
        session: aiohttp.ClientSession,
        sample_id: str,
        conversation: List[Dict[str, Any]],
        state: Dict[str, Any],
        raw_response: str,
    ) -> InferenceTurnResult:
        state["turn"] = state.get("turn", 0) + 1
        parsed = InferenceOutputParser.parse(raw_response)
        parsed["queries"] = [normalize_query_relations(query, state) for query in parsed.get("queries", [])]

        # === Capture plan relations for post-plan skill filtering ===
        for plan_args in parsed.get("parsed_plans", []):
            if isinstance(plan_args, dict):
                self.plan_relations.extend(
                    rel
                    for rel in (plan_args.get("related") or [])
                    if isinstance(rel, str) and rel
                )
                self.plan_relations.extend(
                    rel
                    for rel in (plan_args.get("maybe_related") or [])
                    if isinstance(rel, str) and rel
                )
        if self.plan_relations:
            # Deduplicate while preserving order
            seen: set = set()
            deduped: List[str] = []
            for rel in self.plan_relations:
                if rel not in seen:
                    seen.add(rel)
                    deduped.append(rel)
            self.plan_relations = deduped

        # === Post-plan skill aggregation ===
        # Use the wider shortlisted pool (top-20) for relation-based filtering
        # if available; fall back to the standard retrieved cards (top-3).
        _skill_pool = self.shortlisted_skill_cards or self.retrieved_skill_cards
        if (
            self.plan_relations
            and _skill_pool
            and is_skill_aggregation_enabled()
            and get_aggregation_mode() == "aggregated"
            and not self._has_precomputed_aggregated_hints
            and not self._aggregated_markdown
        ):
            filtered_cards = filter_skills_post_plan(
                _skill_pool,
                self.plan_relations,
            )
            if filtered_cards:
                self._aggregated_markdown = aggregate_skills_to_markdown(
                    filtered_cards,
                    self.skill_target_question,
                )
                action_guidance = extract_action_stage_guidance(self._aggregated_markdown)
                final_guidance = extract_final_stage_guidance(self._aggregated_markdown)
                if action_guidance:
                    self.aggregated_stage_hints["stage:2"] = action_guidance
                    self.aggregated_stage_hints["stage:3"] = action_guidance
                if final_guidance:
                    self.aggregated_stage_hints["stage:4"] = final_guidance
                    self.aggregated_stage_hints["stage:5"] = final_guidance
                    state["aggregated_final_stage_hint"] = final_guidance

        # === Truncation detection ===
        is_truncated = _detect_truncation(raw_response, parsed)
        if is_truncated:
            state["truncation_count"] = state.get("truncation_count", 0) + 1
        else:
            state["truncation_count"] = 0

        # After 3 consecutive truncations, force answer from candidates
        force_answer_from_candidates = (
            state.get("truncation_count", 0) >= 3
            and state.get("retrieved_candidates")
            and not parsed.get("final_answer")
        )
        if force_answer_from_candidates:
            candidates = sorted(state.get("retrieved_candidates", set()))
            parsed["final_answer"] = candidates[:3]
            state["truncation_count"] = 0

        # Track last selected action_id for backtrack
        for query in parsed.get("queries", []):
            if query.get("tool_name") == "select_action":
                action_id = query.get("arguments", {}).get("action_id", "")
                if action_id:
                    if state.get("last_selected_action_id") != action_id:
                        state["filter_stage_waived_action_id"] = None
                    state["last_selected_action_id"] = action_id

                    # Re-filter aggregated final-stage hint by the selected
                    # action's relations so that only experience about
                    # relations actually in the action path reaches Stage 5.
                    active_rels = self._get_active_action_relations(state)
                    if self._aggregated_markdown and state.get("aggregated_final_stage_hint"):
                        if active_rels:
                            filtered = extract_final_stage_guidance(
                                self._aggregated_markdown,
                                active_relations=active_rels,
                            )
                            state["aggregated_final_stage_hint"] = filtered
                            self.aggregated_stage_hints["stage:4"] = filtered
                            self.aggregated_stage_hints["stage:5"] = filtered

                    # Inject answer_strategy from skill cards matching the
                    # selected action's core relation.  On backtrack the
                    # last action wins automatically (last_selected_action_id
                    # is overwritten above).
                    if active_rels and self.retrieved_skill_cards:
                        strategy_hint = self._build_answer_strategy_hint(active_rels)
                        if strategy_hint:
                            existing = self.aggregated_stage_hints.get("stage:5", "")
                            self.aggregated_stage_hints["stage:5"] = (
                                f"{existing}\n\n{strategy_hint}" if existing else strategy_hint
                            )
                            self.aggregated_stage_hints["stage:4"] = self.aggregated_stage_hints["stage:5"]

        # Detect backtrack intent at Stage 5
        backtrack_detected = False
        if PhaseHintGenerator.detect_stage(state) >= 4:
            for query in parsed.get("queries", []):
                if query.get("tool_name") == "backtrack":
                    backtrack_detected = True
                    break
            if not backtrack_detected and parsed.get("final_answer") and not parsed.get("queries"):
                fa_text = " ".join(str(a) for a in parsed["final_answer"]).lower()
                if any(p in fa_text for p in ("backtrack", "rewind", "try different action")):
                    backtrack_detected = True

        if backtrack_detected and state.get("backtrack_attempts", 0) < state.get("max_backtracks", 1):
            state["backtrack_attempts"] = state.get("backtrack_attempts", 0) + 1
            failed_id = state.get("last_selected_action_id")
            if failed_id:
                state["failed_action_ids"] = state.get("failed_action_ids", set()) | {failed_id}
            state["backtrack_checkpoints"] = state.get("backtrack_checkpoints", []) + [{
                "attempt": state["backtrack_attempts"],
                "failed_action_id": failed_id,
                "candidates": sorted(state.get("retrieved_candidates", set())),
            }]
            state["retrieved_candidates"] = set()
            state["has_final_answer"] = False
            state["repair_mode"] = "backtrack"
            parsed["final_answer"] = []
            parsed["queries"] = []

        # === Insufficient streak detection ===
        insufficient_detected = False
        if PhaseHintGenerator.detect_stage(state) >= 4 and not parsed.get("final_answer"):
            response_lower = raw_response.lower()
            insufficient_patterns = [
                "information insufficient", "no valid candidate", "cannot determine",
                "cannot be determined", "cannot verify", "no factual answer",
                "no entity satisfies", "no candidate satisfies",
            ]
            if any(p in response_lower for p in insufficient_patterns):
                insufficient_detected = True

        if insufficient_detected:
            state["insufficient_streak"] = state.get("insufficient_streak", 0) + 1
        else:
            state["insufficient_streak"] = 0

        if "candidates" in parsed:
            state["candidates_collected"] = True
            state["retrieved_candidates"] = set(parsed.get("candidates", []))
            if state["retrieved_candidates"]:
                promote_stable_state(state, reason="candidate_capture")

        forced_stage_feedback = self._handle_stage_decision(parsed, state)
        if forced_stage_feedback:
            parsed["queries"] = []
            parsed["final_answer"] = []

        forced_filter_feedback = "" if forced_stage_feedback else self._maybe_enforce_filter_gate(parsed, state)
        forced_web_search_feedback = self._maybe_enforce_web_search_gate(parsed, state)
        if forced_stage_feedback or forced_filter_feedback or forced_web_search_feedback:
            parsed["final_answer"] = []
            parsed["queries"] = []

        # === Trigger background search ===
        # IMPORTANT: This must run AFTER filter/websearch gates clear invalid answers.
        # Trigger when: (a) model outputs "insufficient" pattern, OR
        # (b) late turn with candidates but no answer after gate checks (model stuck)
        should_search = (
            not state.get("search_injected", False)
            and SearchAgentConfig().enabled
            and (
                state.get("insufficient_streak", 0) >= 1
                or (
                    state.get("turn", 0) >= 4
                    and state.get("retrieved_candidates")
                    and not parsed.get("final_answer")
                    and not parsed.get("queries")
                )
            )
        )
        if should_search:
            missing_attr = _extract_missing_attribute(parsed, state)
            search_agent = BackgroundSearchAgent(config=SearchAgentConfig())
            search_results = await search_agent.search_kgqa(
                question=state.get("prompt_context", {}).get("original_question", ""),
                candidates=sorted(state.get("retrieved_candidates", set())),
                missing_attribute=missing_attr,
                reasoning_summary=raw_response[:500],
            )
            state["search_results"] = search_results
            state["search_injected"] = True

        frontend_errors = FrontendValidator.validate(parsed, state)
        state["has_final_answer"] = bool(parsed.get("final_answer")) and not frontend_errors
        executed_queries: List[Dict[str, Any]] = []
        backend_results: List[Any] = []

        if frontend_errors:
            RepairPolicy.note_frontend_errors(frontend_errors, state)
            feedback = ErrorFeedback.format_frontend(frontend_errors)
            hint = OneShotStageHintManager.next_hint(state)
            if hint:
                feedback = f"{feedback}\n\n{hint}"
            skill_hint = self._maybe_stage_skill_hint(state)
            if skill_hint:
                feedback = f"{feedback}\n\n{skill_hint}"
            web_hint = self._maybe_web_search_hint(state)
            if web_hint:
                feedback = f"{feedback}\n\n{web_hint}"
            stable_hint = self._stable_state_repair_hint(state)
            if stable_hint:
                feedback = f"{feedback}\n\n{stable_hint}"
        elif forced_stage_feedback or forced_filter_feedback or forced_web_search_feedback:
            feedback_parts: List[str] = [forced_stage_feedback or forced_filter_feedback or forced_web_search_feedback]
            hint = OneShotStageHintManager.next_hint(state)
            if hint:
                feedback_parts.append(hint)
            skill_hint = self._maybe_stage_skill_hint(state)
            if skill_hint:
                feedback_parts.append(skill_hint)
            stable_hint = self._stable_state_repair_hint(state)
            if stable_hint:
                feedback_parts.append(stable_hint)
            feedback = "\n\n".join(_dedupe_feedback_parts(feedback_parts)).strip()
        else:
            executed_queries = self._materialize_queries(parsed.get("queries", []), state)
            raw_results = [
                await self.backend.execute_tool(session, query, sample_id, state)
                for query in executed_queries
            ]
            backend_results, has_plan_errors = BackendChecker.check(executed_queries, raw_results)
            update_state_from_backend_results(state, backend_results)

            feedback_parts: List[str] = []
            if backend_results:
                tool_blocks = []
                for result in backend_results:
                    icon = "✓" if result.is_success else "✗"
                    tool_blocks.append(
                        f"=== {result.tool_name} [{icon} {result.status}] ===\n{result.response_text}"
                    )
                feedback_parts.append("[TOOL RESULTS]\n" + "\n\n".join(tool_blocks))

            if has_plan_errors:
                RepairPolicy.note_backend_results(backend_results, state)
                feedback_parts.append(ErrorFeedback.format_backend(backend_results))
                hint = OneShotStageHintManager.next_hint(state)
                if hint:
                    feedback_parts.append(hint)
                skill_hint = self._maybe_stage_skill_hint(state)
                if skill_hint:
                    feedback_parts.append(skill_hint)
                web_hint = self._maybe_web_search_hint(state)
                if web_hint:
                    feedback_parts.append(web_hint)
                stable_hint = self._stable_state_repair_hint(state)
                if stable_hint:
                    feedback_parts.append(stable_hint)
            else:
                RepairPolicy.clear(state)
                OneShotStageHintManager.clear_repair_tracking(state)
                executed_tool_names = {result.tool_name for result in backend_results}
                if executed_tool_names & {"match_pattern", "action"}:
                    state["awaiting_action_judgment"] = True
                    state["awaiting_filter_decision"] = False
                if state.get("retrieved_candidates") or state.get("last_selected_action_id"):
                    promote_stable_state(state, reason="backend_success")
                hint = OneShotStageHintManager.next_hint(state)
                if hint:
                    feedback_parts.append(hint)
                skill_hint = self._maybe_stage_skill_hint(state)
                if skill_hint:
                    feedback_parts.append(skill_hint)
                web_hint = self._maybe_web_search_hint(state)
                if web_hint:
                    feedback_parts.append(web_hint)

            feedback = "\n\n".join(_dedupe_feedback_parts(feedback_parts)).strip()
            if not feedback:
                if is_truncated:
                    feedback = (
                        "[OUTPUT TRUNCATED]\n"
                        "Your previous response was cut off before completion. "
                        "Your output likely exceeded the token limit.\n"
                        "Please provide a SHORTER response. Specifically:\n"
                        "- Use fewer words in reasoning\n"
                        "- Output only the essential <query> or <answer> block\n"
                        "- Avoid repeating previous context\n"
                        "Try again with a concise response."
                    )
                else:
                    feedback = "[SYSTEM] No tools executed and no errors found. Please output <act> or <answer>."

        conversation.append({"role": "assistant", "content": raw_response})
        if self.followup_hint:
            feedback = "\n\n".join(_dedupe_feedback_parts([feedback, self.followup_hint]))
        conversation.append({"role": "user", "content": feedback})

        return InferenceTurnResult(
            raw_response=raw_response,
            parsed_output=parsed,
            frontend_errors=frontend_errors,
            executed_queries=executed_queries,
            backend_results=backend_results,
            feedback=feedback,
            state_snapshot=snapshot_state(state),
        )

    def _get_active_action_relations(self, state: Dict[str, Any]) -> set[str]:
        """Extract relations from the currently selected action's path.

        Returns the set of fully-qualified relation names used in the
        selected action's signature, or an empty set if no action has been
        selected yet.
        """
        action_id = state.get("last_selected_action_id")
        if not action_id:
            return set()
        action_map = state.get("action_id_map", {})
        hint = action_map.get(action_id, {})
        if not isinstance(hint, dict):
            return set()
        signature = hint.get("signature", [])
        if not isinstance(signature, list):
            return set()
        rels: set[str] = set()
        for step in signature:
            if isinstance(step, dict):
                rel = step.get("relation", "")
                if rel and isinstance(rel, str):
                    rels.add(rel)
        return rels

    def _build_answer_strategy_hint(self, active_relations: set[str]) -> str:
        """Build a Stage-5 answer strategy hint from skill cards whose
        core_relations intersect with the selected action space relations.

        Only the answer_strategy field is extracted; no other skill info.
        If multiple cards match, they are listed in retrieval rank order.
        """
        matched: List[CaseSkillCard] = []
        for card in self.retrieved_skill_cards:
            if set(card.core_relations or []) & active_relations:
                matched.append(card)
        if not matched:
            return ""

        lines = ["[ANSWER STRATEGY FROM MATCHED CASES]"]
        for card in matched[:3]:  # cap at 3 to limit context
            strategy = card.answer_strategy or {}
            if not strategy:
                continue
            lines.append(f"Case `{card.case_id}` (\"{card.question[:60]}\"):")
            for key, val in strategy.items():
                if val and val not in ("", [], "none", "unknown"):
                    lines.append(f"  - {key}: {val}")
        return "\n".join(lines) if len(lines) > 1 else ""

    def _maybe_stage_skill_hint(self, state: Dict[str, Any]) -> str:
        stage = PhaseHintGenerator.detect_stage(state)
        repair_mode = state.get("repair_mode")
        if repair_mode:
            if stage != 1:
                return ""
            key = f"repair:{repair_mode}:stage:{stage}"
        else:
            key = f"stage:{stage}"
        hint = self.stage_skill_hints.get(key, "")
        if not hint:
            hint = self.stage_skill_hints.get(f"stage:{stage}", "")
        emitted = state.setdefault("_emitted_stage_skill_keys", set())
        if key in emitted:
            return ""
        dynamic_hint = ""
        if stage >= 4 and state.get("aggregated_final_stage_hint"):
            # Final-stage skill is injected inside the Stage 5 hint itself.
            # Suppress a second standalone stage-skill block here.
            emitted.add(key)
            return ""
        if self.retrieved_skill_cards:
            if _skill_pre_plan_injection_enabled():
                # Legacy mode: inject skills during planning (original behaviour).
                if stage == 1:
                    filtered_cards = self._filter_skill_cards_to_observed_relations(state)
                    if filtered_cards:
                        dynamic_hint = build_relation_stage_hint(filtered_cards, include_action_space=False)
                elif stage in (2, 3):
                    if os.getenv("KGQA_ENABLE_ACTION_STAGE_HINTS", "0").strip().lower() in {"1", "true", "yes", "on"}:
                        dynamic_hint = build_action_stage_hint(self.retrieved_skill_cards)
                elif stage == 4:
                    dynamic_hint = build_reasoning_stage_hint(
                        self.retrieved_skill_cards,
                        self.skill_target_question,
                        include_action_space=False,
                    )
            else:
                # Post-plan mode: classified discovery hint at stage 1.
                # Stages 2+ use plan-relation-filtered cards.
                if stage == 1:
                    dynamic_hint = build_classified_discovery_hint(self.retrieved_skill_cards)
                else:
                    agg_mode = get_aggregation_mode()
                    agg_key = f"stage:{stage}"
                    precomputed_agg_hint = self.aggregated_stage_hints.get(agg_key, "")
                    if precomputed_agg_hint:
                        # Prefer the externally prepared aggregated hint when available.
                        # This lets the newer LLM-built post-plan aggregation override
                        # older deterministic aggregation paths.
                        dynamic_hint = precomputed_agg_hint
                    elif agg_mode == "aggregated" and self._aggregated_markdown:
                        # Aggregated mode: use pre-built stage hints from
                        # the markdown artifact.  Stages 2-3 get action
                        # guidance, stages 4-5 get final-selection guidance.
                        dynamic_hint = self.aggregated_stage_hints.get(agg_key, "")
                    else:
                        # Per-skill (legacy) or markdown not yet built:
                        # fall back to per-card hint builders.
                        post_plan_cards = filter_skills_post_plan(
                            self.retrieved_skill_cards,
                            self.plan_relations,
                        )
                        if post_plan_cards:
                            if stage in (2, 3):
                                if os.getenv("KGQA_ENABLE_ACTION_STAGE_HINTS", "0").strip().lower() in {"1", "true", "yes", "on"}:
                                    dynamic_hint = build_action_stage_hint(post_plan_cards)
                            elif stage == 4:
                                dynamic_hint = build_reasoning_stage_hint(
                                    post_plan_cards,
                                    self.skill_target_question,
                                    include_action_space=False,
                                )
        chosen_hint = dynamic_hint or hint
        if not chosen_hint and stage not in (1, 2, 3):
            return ""

        # Append negative experience hints at plan/action stages
        negative_suffix = ""
        if stage == 1 and self.negative_plan_hint:
            negative_suffix = "\n\n" + self.negative_plan_hint
        elif stage in (2, 3) and self.negative_action_hint:
            negative_suffix = "\n\n" + self.negative_action_hint

        if not chosen_hint and not negative_suffix:
            return ""

        # Check for aggregated stage hints (only when NOT already in dynamic_hint
        # via aggregated mode, to avoid double-injection).
        _agg_in_dynamic = (
            not _skill_pre_plan_injection_enabled()
            and dynamic_hint == self.aggregated_stage_hints.get(f"stage:{stage}", "")
            and bool(dynamic_hint)
        )
        if _agg_in_dynamic:
            aggregated_hint = ""
        else:
            aggregated_hint = self.aggregated_stage_hints.get(f"stage:{stage}", "")

        # Filter aggregated hint by selected action's relations at stages 4-5
        # to remove experience about relations not in the active action path.
        if aggregated_hint and stage >= 4:
            active_rels = self._get_active_action_relations(state)
            if active_rels and self._aggregated_markdown:
                aggregated_hint = extract_final_stage_guidance(
                    self._aggregated_markdown,
                    active_relations=active_rels,
                )

        emitted.add(key)
        if aggregated_hint:
            emitted.add("aggregated_stage:" + str(stage))

        # Combine hints when both aggregated and negative hints exist
        if stage in (2, 3):
            historical_header = (
                "[Historical Skill Check — Before choosing or retrying an action, "
                "you MUST compare the current action space against these solved cases. "
                "Use their key relations and action patterns as a required decision signal. "
                "Do not copy their answers directly.]"
            )
        else:
            historical_header = (
                "[Historical Case Reference — These are solved cases grouped by answer direction. "
                "Use their experience as REFERENCE ONLY, not as direct answers.]"
            )

        if aggregated_hint and negative_suffix and chosen_hint:
            return f"{chosen_hint}\n\n{historical_header}\n{aggregated_hint}\n\n[Negative Experience Reference]\n{negative_suffix.lstrip()}"
        elif aggregated_hint and chosen_hint:
            return f"{chosen_hint}\n\n{historical_header}\n{aggregated_hint}"
        elif aggregated_hint and negative_suffix:
            emitted.add("negative_stage:" + str(stage))
            return f"{historical_header}\n{aggregated_hint}\n\n[Negative Experience Reference]\n{negative_suffix.lstrip()}"
        elif negative_suffix and chosen_hint:
            return chosen_hint + negative_suffix
        elif negative_suffix:
            emitted.add("negative_stage:" + str(stage))
            return negative_suffix.lstrip("\n")
        return chosen_hint

    @staticmethod
    def _maybe_web_search_hint(state: Dict[str, Any]) -> str:
        if os.getenv("KGQA_ENABLE_WEB_SEARCH", "0").strip().lower() not in {"1", "true", "yes", "on"}:
            return ""
        candidates = sorted(list(state.get("retrieved_candidates", set()) or []))
        if len(candidates) < 2 or state.get("web_search_used"):
            return ""
        if state.get("turn", 0) < 4:
            return ""
        if InferenceRuntime._pending_filter_relations(state) and not InferenceRuntime._has_filter_evidence(state):
            return ""
        emitted = state.setdefault("_emitted_stage_skill_keys", set())
        key = "stage:4:web_search_decision"
        if key in emitted:
            return ""
        emitted.add(key)
        preview = ", ".join(candidates[:6])
        snapshot_date = _graph_snapshot_date()
        snapshot_clause = ""
        if snapshot_date:
            snapshot_clause = (
                f"\nGraph snapshot date: {snapshot_date}\n"
                "Interpret CURRENT / LATEST relative to this snapshot date, not real-world today."
            )
        return (
            "[HINT] If graph-visible evidence cannot distinguish among the candidates above, "
            "you may call `search()` to compare them externally. "
            "The final answer must still be an exact graph string from the current graph candidates."
            f"{snapshot_clause}"
        )

    def _maybe_enforce_filter_gate(self, parsed: Dict[str, Any], state: Dict[str, Any]) -> str:
        current_candidates = sorted(list(state.get("retrieved_candidates", set()) or []))
        if len(current_candidates) < 2:
            return ""
        if self._has_filter_evidence(state):
            return ""
        if any(query.get("tool_name") == "filter" for query in (parsed.get("queries", []) or [])):
            return ""
        if not parsed.get("candidates") and not parsed.get("final_answer"):
            return ""

        relations = self._pending_filter_relations(state)
        if not relations:
            return ""
        if state.get("filter_stage_waived_action_id") == state.get("last_selected_action_id"):
            return ""
        relation_list = ", ".join(f'"{relation}"' for relation in relations)
        preview = ", ".join(current_candidates[:6])

        gate_msg = (
            "[FILTER GATE — DISAMBIGUATION REQUIRED]\n"
            f"Multiple candidates remain: {preview}\n"
            f"Suggested relations where candidates differ: {', '.join(relations)}\n\n"
            "These candidates cannot proceed without disambiguation.\n"
            "You MUST call filter() to retrieve per-candidate attribute values.\n\n"
            "You decide which relation(s) to use as filter parameters.\n"
            "Choose the relation(s) most relevant to the question's intent.\n"
        )

        # Enhance gate with skill-sourced comparison criteria (feature-gated)
        if os.getenv("KGQA_ENHANCED_FILTER_GATE", "0").strip().lower() in {"1", "true", "yes", "on"}:
            skill_guidance = self._build_skill_filter_guidance()
            if skill_guidance:
                gate_msg += f"\n\n{skill_guidance}"

        return gate_msg

    def _build_skill_filter_guidance(self) -> str:
        """Build skill-sourced filter interpretation guidance."""
        if not self.retrieved_skill_cards:
            return ""
        lines = ["[SKILL FILTER GUIDANCE]"]
        lines.append("Experience from similar cases suggests:")

        for card in self.retrieved_skill_cards[:2]:
            # Final selection experience: concrete selection rules
            if card.final_selection_experience:
                for exp in card.final_selection_experience[:2]:
                    lines.append(f"  - {exp}")

            # Answer strategy: how to interpret filter results
            strategy = card.answer_strategy or {}
            selection_rule = strategy.get("selection_rule", "")
            if selection_rule:
                lines.append(f"  - Selection strategy: {selection_rule}")

            # Common pitfalls relevant to filtering
            for pitfall in (card.common_pitfalls or [])[:1]:
                lines.append(f"  - Pitfall: {pitfall}")

        if len(lines) <= 2:
            return ""
        lines.append("Apply this experience when interpreting filter() results.")
        return "\n".join(lines)

    def _maybe_enforce_web_search_gate(self, parsed: Dict[str, Any], state: Dict[str, Any]) -> str:
        if os.getenv("KGQA_ENFORCE_WEB_SEARCH_GATE", "0").strip().lower() not in {"1", "true", "yes", "on"}:
            return ""
        final_answers = list(parsed.get("final_answer", []) or [])
        current_candidates = sorted(list(state.get("retrieved_candidates", set()) or []))
        if state.get("web_search_used"):
            return ""

        # --- Case A: empty final answer but candidates exist (websearch fallback) ---
        if not final_answers and current_candidates and state.get("turn", 0) >= 3:
            # One-shot: only emit once per session to avoid infinite loops
            emitted = state.setdefault("_emitted_stage_skill_keys", set())
            gate_key = "gate:web_search_validation"
            if gate_key in emitted:
                return ""
            # Only tell model to use web_search if the tool is actually enabled
            web_enabled = os.getenv("KGQA_ENABLE_WEB_SEARCH", "0").strip().lower() in {"1", "true", "yes", "on"}
            if not web_enabled:
                # Web search unavailable: nudge model toward final_answer from candidates
                emitted.add(gate_key)
                preview = ", ".join(current_candidates[:6])
                return (
                    "[CANDIDATE ANSWER REMINDER]\n"
                    f"Graph queries returned candidates ({preview}) but no final answer was selected.\n"
                    "Review the candidates above and output the subset that matches the question as final_answer.\n"
                    "If none match, output an empty final_answer."
                )
            emitted.add(gate_key)
            preview = ", ".join(current_candidates[:6])
            snapshot_date = _graph_snapshot_date()
            snapshot_clause = ""
            if snapshot_date:
                snapshot_clause = (
                    f"\nGraph snapshot date: {snapshot_date}\n"
                    "Interpret CURRENT / LATEST relative to this snapshot date."
                )
            return (
                "[ANSWER VALIDATION GATE]\n"
                f"Graph queries returned candidates ({preview}) but no final answer was selected.\n"
                "Before concluding that no answer exists, verify externally.\n"
                "Next step:\n"
                "- call `search()` to let the external search agent compare the current graph candidates.\n"
                "- If web search confirms a valid answer from the candidates, output it as final_answer.\n"
                "- If web search confirms no valid answer exists among candidates, output an empty final_answer."
                f"{snapshot_clause}"
            )

        # --- Case B: single answer but multiple candidates remain (existing behavior) ---
        if len(final_answers) != 1 or len(current_candidates) < 2:
            return ""
        if self._has_filter_evidence(state):
            return ""
        if self._pending_filter_relations(state):
            return ""

        chosen = final_answers[0]
        preview = ", ".join(current_candidates[:6])
        snapshot_date = _graph_snapshot_date()
        snapshot_clause = ""
        if snapshot_date:
            snapshot_clause = (
                f"\nGraph snapshot date: {snapshot_date}\n"
                "Interpret CURRENT / LATEST relative to this snapshot date while deciding whether a single answer is justified."
            )
        return (
            "[EVIDENCE SUFFICIENCY GATE]\n"
            f"You are trying to output a SINGLE answer (`{chosen}`) while multiple graph candidates remain: {preview}\n"
            "No graph-side disambiguation step has been used yet (for example filter), and web_search has not been used.\n"
            "Do NOT guess a single answer from parametric knowledge alone.\n"
            "Next step:\n"
            "- either call `search()` to compare the CURRENT graph candidates externally, OR\n"
            "- if you still cannot separate them, keep the graph-supported candidates instead of forcing one."
            f"{snapshot_clause}"
        )

    @staticmethod
    def _materialize_queries(queries: List[Dict[str, Any]], state: Dict[str, Any]) -> List[Dict[str, Any]]:
        materialized: List[Dict[str, Any]] = []
        action_id_map = state.get("action_id_map", {})
        for query in queries:
            if query.get("tool_name") != "select_action":
                materialized.append(query)
                continue
            action_id = query.get("arguments", {}).get("action_id", "")
            hint = action_id_map.get(action_id)
            if not hint:
                materialized.append(query)
                continue
            arguments: Dict[str, Any] = {
                "anchor": hint.get("start_entity", ""),
                "path": hint.get("steps", []),
            }
            if hint.get("constraint_relations"):
                arguments["constraint_relations"] = hint["constraint_relations"]
            if hint.get("constraint_entities"):
                arguments["constraint_entities"] = hint["constraint_entities"]
            materialized.append({"tool_name": "action", "arguments": arguments})
        return materialized
