from __future__ import annotations

import ast
import json
import os
import re
import tempfile
from typing import Any, Dict, Iterable, Literal, Optional

from subgraph_kgqa.rl.plugin import RepairPolicy


LEAF_PATTERN = re.compile(r"Leaf Entities \(\d+\):\s*\n\s*(\[[^\n]*\])", re.S)
CVT_PATTERN = re.compile(r"CVT-Expanded Entities \(\d+\):\s*\n\s*(\[[^\n]*\])", re.S)
SUGGESTED_RELATION_PATTERN = re.compile(
    r"\[Suggested Filter Relations\]:\s*\n(?:The following relations.*\n)?((?:\s*-\s*[^\n]+\n)+)",
    re.S,
)


def _relation_aliases(relation: str) -> set[str]:
    aliases = set()
    if not relation or not isinstance(relation, str):
        return aliases
    aliases.add(relation)
    parts = relation.split(".")
    if parts:
        aliases.add(parts[-1])
    if len(parts) >= 2:
        aliases.add(".".join(parts[-2:]))
    return aliases


def add_relations_to_alias_map(state: Dict[str, Any], relations: Optional[Iterable[str]]) -> None:
    alias_map = state.setdefault("relation_alias_map", {})
    for relation in relations or []:
        if not relation or not isinstance(relation, str):
            continue
        for alias in _relation_aliases(relation):
            alias_map.setdefault(alias, set()).add(relation)


def _normalize_relation_value(value: Any, alias_map: Dict[str, set[str]]) -> Any:
    if not isinstance(value, str) or not value:
        return value
    matches = alias_map.get(value, set())
    if len(matches) == 1:
        return next(iter(matches))
    return value


def _extract_entities_from_response_text(response_text: str) -> list[str]:
    for pattern in (LEAF_PATTERN, CVT_PATTERN):
        match = pattern.search(response_text or "")
        if not match:
            continue
        try:
            values = ast.literal_eval(match.group(1))
            return [str(v) for v in values if str(v).strip()]
        except Exception:
            continue
    return []


def _extract_suggested_filter_relations(response_text: str) -> list[str]:
    match = SUGGESTED_RELATION_PATTERN.search(response_text or "")
    if not match:
        return []
    relations: list[str] = []
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        relation = line[2:].strip()
        if relation:
            relations.append(relation)
    return relations


def normalize_query_relations(query: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    tool_name = query.get("tool_name")
    args = query.get("arguments", {})
    alias_map = state.get("relation_alias_map", {})
    if not alias_map or not isinstance(args, dict):
        return query

    field_names = {
        "plan": ("related", "maybe_related", "constraint_relations", "select_relations"),
        "plan_subquestion": ("related", "maybe_related", "constraint_relations", "select_relations"),
        "filter": ("constraint_relations", "select_relations"),
        "action": ("constraint_relations", "select_relations"),
        "match_pattern": ("constraint_relations", "select_relations"),
    }.get(tool_name, ())

    for field_name in field_names:
        field_value = args.get(field_name)
        if isinstance(field_value, list):
            args[field_name] = [_normalize_relation_value(item, alias_map) for item in field_value]
        elif isinstance(field_value, str):
            args[field_name] = _normalize_relation_value(field_value, alias_map)

    if tool_name in {"action", "match_pattern"}:
        path = args.get("path", [])
        if isinstance(path, list):
            normalized_path = []
            for step in path:
                if isinstance(step, dict) and "relation" in step:
                    updated_step = dict(step)
                    updated_step["relation"] = _normalize_relation_value(step.get("relation"), alias_map)
                    normalized_path.append(updated_step)
                else:
                    normalized_path.append(step)
            args["path"] = normalized_path

    query["arguments"] = args
    return query


def build_initial_state(
    *,
    core_entities: Optional[Iterable[str]] = None,
    core_relations: Optional[Iterable[str]] = None,
    available_domains: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    state = {
        "turn": 0,
        "current_phase": 1,
        "phase1_entity_done": False,
        "phase1_schema_done": False,
        "phase2_complete": False,
        "verified_entities": set(),
        "schema_relations": set(core_relations or []),
        "observed_schema_relations": set(),
        "executed_commands": set(),
        "retrieved_candidates": set(),
        "all_leaf_entities": set(),
        "any_match_executed": False,
        "has_action_hint": False,
        "candidates_collected": False,
        "has_final_answer": False,
        "awaiting_action_judgment": False,
        "awaiting_filter_decision": False,
        "filter_stage_waived_action_id": None,
        "plan": None,
        "constraints": [],
        "valid_actions": set(),
        "action_id_map": {},
        "relation_alias_map": {},
        "valid_action_tuples": [],
        "match_pattern_results": {},
        "plan_hallucinations": [],
        "repair_mode": None,
        "repair_reason": "",
        "repair_error_types": [],
        "repair_backend_statuses": [],
        # Backtrack state
        "last_selected_action_id": None,
        "backtrack_attempts": 0,
        "max_backtracks": 1,
        "failed_action_ids": set(),
        "backtrack_checkpoints": [],
        # Search tracking state
        "insufficient_streak": 0,
        "search_injected": False,
        "search_results": None,
        "history": [],
        "tool_structured_results": {},
        "filter_suggestion_blocks": [],
        "web_search_used": False,
        "web_search_results": [],
        "stable_state": {
            "verified_entities": set(),
            "schema_relations": set(),
            "observed_schema_relations": set(),
            "retrieved_candidates": set(),
            "all_leaf_entities": set(),
            "filter_suggestion_blocks": [],
            "last_selected_action_id": None,
            "turn": 0,
            "reason": "",
        },
        "stable_checkpoints": [],
        "truncation_count": 0,
        "hint_history": [],
        "_emitted_stage_hint_keys": set(),
        "_emitted_stage_skill_keys": set(),
        "_last_hint_key": None,
        "prompt_context": {
            "core_entities": set(core_entities or []),
            "core_relations": set(core_relations or []),
            "available_domains": set(available_domains or []),
        },
    }
    add_relations_to_alias_map(state, core_relations)
    return state


def promote_stable_state(state: Dict[str, Any], *, reason: str) -> None:
    stable = state.setdefault("stable_state", {})
    stable.update(
        {
            "verified_entities": set(state.get("verified_entities", set()) or set()),
            "schema_relations": set(state.get("schema_relations", set()) or set()),
            "observed_schema_relations": set(state.get("observed_schema_relations", set()) or set()),
            "retrieved_candidates": set(state.get("retrieved_candidates", set()) or set()),
            "all_leaf_entities": set(state.get("all_leaf_entities", set()) or set()),
            "filter_suggestion_blocks": list(state.get("filter_suggestion_blocks", []) or []),
            "last_selected_action_id": state.get("last_selected_action_id"),
            "turn": state.get("turn", 0),
            "reason": reason,
        }
    )

    checkpoints = state.setdefault("stable_checkpoints", [])
    checkpoints.append(
        {
            "turn": state.get("turn", 0),
            "reason": reason,
            "candidate_count": len(stable.get("retrieved_candidates", set()) or set()),
            "action_id": stable.get("last_selected_action_id"),
        }
    )
    if len(checkpoints) > 20:
        del checkpoints[:-20]


def snapshot_state(state: Dict[str, Any]) -> Dict[str, Any]:
    stable = state.get("stable_state", {}) or {}
    return {
        "turn": state.get("turn", 0),
        "current_phase": state.get("current_phase", 1),
        "phase1_entity_done": bool(state.get("phase1_entity_done")),
        "phase1_schema_done": bool(state.get("phase1_schema_done")),
        "verified_entities": sorted(list(state.get("verified_entities", set())))[:20],
        "schema_relations": sorted(list(state.get("schema_relations", set())))[:40],
        "observed_schema_relations": sorted(list(state.get("observed_schema_relations", set())))[:40],
        "relation_alias_count": len(state.get("relation_alias_map", {})),
        "filter_suggestion_blocks": len(state.get("filter_suggestion_blocks", [])),
        "has_action_hint": bool(state.get("has_action_hint")),
        "valid_actions_count": len(state.get("valid_actions", set())),
        "any_match_executed": bool(state.get("any_match_executed")),
        "retrieved_candidates": sorted(list(state.get("retrieved_candidates", set())))[:20],
        "all_leaf_entities": sorted(list(state.get("all_leaf_entities", set())))[:20],
        "candidates_collected": bool(state.get("candidates_collected")),
        "has_final_answer": bool(state.get("has_final_answer")),
        "awaiting_action_judgment": bool(state.get("awaiting_action_judgment")),
        "awaiting_filter_decision": bool(state.get("awaiting_filter_decision")),
        "filter_stage_waived_action_id": state.get("filter_stage_waived_action_id"),
        "web_search_used": bool(state.get("web_search_used")),
        "repair_mode": state.get("repair_mode"),
        "repair_reason": state.get("repair_reason", ""),
        "truncation_count": state.get("truncation_count", 0),
        "stable_candidate_count": len(stable.get("retrieved_candidates", set()) or set()),
        "stable_candidates": sorted(list(stable.get("retrieved_candidates", set()) or set()))[:20],
        "stable_last_action_id": stable.get("last_selected_action_id"),
        "stable_reason": stable.get("reason", ""),
    }


def update_state_from_backend_results(state: Dict[str, Any], results: list[Any]) -> None:
    if state.get("verified_entities"):
        state["phase1_entity_done"] = True
    if state.get("schema_relations"):
        state["phase1_schema_done"] = True

    turn_key = f"turn_{state.get('turn', 0)}"
    state.setdefault("tool_structured_results", {})
    state["tool_structured_results"].setdefault(turn_key, [])

    for result in results:
        if not getattr(result, "is_success", False):
            continue

        state["tool_structured_results"][turn_key].append(
            {
                "tool_name": result.tool_name,
                "structured_data": getattr(result, "structured_data", {}) or {},
            }
        )

        if result.tool_name in {"check_entities", "find_entities"}:
            state["verified_entities"].update(result.found_entities)
            state["phase1_entity_done"] = True
            if state.get("repair_mode") == RepairPolicy.REOPEN_DISCOVERY:
                RepairPolicy.clear(state)

        if result.tool_name == "explore_schema":
            if result.found_entities:
                state["schema_relations"].update(result.found_entities)
                state["observed_schema_relations"].update(result.found_entities)
                add_relations_to_alias_map(state, result.found_entities)
                state["phase1_schema_done"] = True
                if state.get("repair_mode") == RepairPolicy.REOPEN_DISCOVERY:
                    RepairPolicy.clear(state)

        if result.tool_name in {"match_pattern", "action"}:
            state["any_match_executed"] = True
            if state.get("repair_mode") == RepairPolicy.RECOPY_ACTION:
                RepairPolicy.clear(state)
            if result.found_entities:
                state["all_leaf_entities"].update(result.found_entities)
                state["verified_entities"].update(result.found_entities)
            response_text = getattr(result, "response_text", "") or ""
            suggested_relations = _extract_suggested_filter_relations(response_text)
            action_entities = _extract_entities_from_response_text(response_text)
            if suggested_relations and action_entities:
                add_relations_to_alias_map(state, suggested_relations)
                state["observed_schema_relations"].update(suggested_relations)
                blocks = state.setdefault("filter_suggestion_blocks", [])
                blocks.append(
                    {
                        "candidates": sorted(set(action_entities)),
                        "suggested_relations": suggested_relations,
                    }
                )
                if len(blocks) > 20:
                    del blocks[:-20]

        if result.tool_name in {"plan_subquestion", "plan"}:
            if result.action_hints:
                state["has_action_hint"] = True
                if state.get("repair_mode") == RepairPolicy.FORCED_REPLAN:
                    RepairPolicy.clear(state)
                for idx, hint in enumerate(result.action_hints, 1):
                    action_id = hint.get("action_id")
                    if not action_id:
                        action_id = f"A{idx}"
                        hint["action_id"] = action_id
                    if action_id:
                        state["action_id_map"][action_id] = hint
                    signature = hint.get("signature", [])
                    start_entity = hint.get("start_entity", "") or hint.get("anchor", "")
                    if not signature or not start_entity:
                        continue
                    add_relations_to_alias_map(
                        state,
                        [step.get("relation") for step in signature if isinstance(step, dict)],
                    )
                    add_relations_to_alias_map(state, hint.get("constraint_relations", []))
                    add_relations_to_alias_map(state, hint.get("select_relations", []))
                    state["observed_schema_relations"].update(
                        step.get("relation")
                        for step in signature
                        if isinstance(step, dict) and isinstance(step.get("relation"), str) and step.get("relation")
                    )
                    state["observed_schema_relations"].update(
                        relation
                        for relation in (hint.get("constraint_relations", []) or [])
                        if isinstance(relation, str) and relation
                    )
                    state["observed_schema_relations"].update(
                        relation
                        for relation in (hint.get("select_relations", []) or [])
                        if isinstance(relation, str) and relation
                    )
                    path_sig = "|".join(
                        f"{step['relation']}:{step.get('direction', 'out')}"
                        for step in signature
                    )
                    state["valid_actions"].add(f"{start_entity}|{path_sig}")

        if result.tool_name == "web_search":
            state["web_search_used"] = True
            state.setdefault("web_search_results", []).append(getattr(result, "structured_data", {}) or {})


# ---------------------------------------------------------------------------
# State persistence helpers
# ---------------------------------------------------------------------------

# Keys whose values are sets that should be serialized as sorted lists.
_SET_KEYS = frozenset({
    "verified_entities",
    "schema_relations",
    "observed_schema_relations",
    "executed_commands",
    "retrieved_candidates",
    "all_leaf_entities",
    "valid_actions",
    "failed_action_ids",
})

# Keys that hold callables or other non-serializable objects -- excluded on save.
_EXCLUDED_KEYS = frozenset({
    "_emitted_stage_hint_keys",
    "_emitted_stage_skill_keys",
    "_last_hint_key",
})


def _serialize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-safe copy of *state*, converting sets and dropping callables."""
    out: Dict[str, Any] = {}
    for key, value in state.items():
        if key in _EXCLUDED_KEYS:
            continue
        if callable(value):
            continue
        if isinstance(value, set):
            out[key] = sorted(list(value))
        elif isinstance(value, dict):
            # Recurse for nested dicts (e.g. stable_state, prompt_context)
            out[key] = _serialize_state(value)
        else:
            try:
                json.dumps(value)
                out[key] = value
            except (TypeError, ValueError):
                pass  # skip non-serializable
    return out


def _deserialize_state(data: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct sets from sorted lists for known set keys."""
    out: Dict[str, Any] = dict(data)
    for key in _SET_KEYS:
        if key in out and isinstance(out[key], list):
            out[key] = set(out[key])
    # Handle nested stable_state
    stable = out.get("stable_state")
    if isinstance(stable, dict):
        for key in ("verified_entities", "schema_relations", "observed_schema_relations",
                     "retrieved_candidates", "all_leaf_entities"):
            if key in stable and isinstance(stable[key], list):
                stable[key] = set(stable[key])
    # Handle nested prompt_context
    pc = out.get("prompt_context")
    if isinstance(pc, dict):
        for key in ("core_entities", "core_relations", "available_domains"):
            if key in pc and isinstance(pc[key], list):
                pc[key] = set(pc[key])
    return out


def save_state_to_disk(state: Dict[str, Any], path: str) -> None:
    """Serialize *state* to JSON at *path* atomically (write-to-temp then rename)."""
    serialized = _serialize_state(state)
    directory = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up the temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def restore_state_from_disk(path: str) -> Optional[Dict[str, Any]]:
    """Load and deserialize state from *path*.  Returns ``None`` if missing or corrupted."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return _deserialize_state(data)
    except (json.JSONDecodeError, OSError, ValueError):
        return None
