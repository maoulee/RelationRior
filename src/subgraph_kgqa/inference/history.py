from __future__ import annotations

import copy
import os
from typing import Any, Dict, Iterable, List, Set

from subgraph_kgqa.rl.plugin import compress_with_structured_data, strip_reason_blocks


def prepare_conversation_for_llm(
    conversation: Iterable[Dict[str, Any]],
    *,
    strip_history_reasoning: bool = True,
) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for message in conversation:
        content = message.get("content", "")
        if strip_history_reasoning and message.get("role") == "assistant" and content:
            content = strip_reason_blocks(content)
        prepared.append({**message, "content": content})
    return prepared


def compress_history_messages(
    conversation: Iterable[Dict[str, Any]],
    state: Dict[str, Any],
    *,
    candidate_entities: Set[str] | None = None,
    branch_threshold: int | None = None,
) -> List[Dict[str, Any]]:
    messages = copy.deepcopy(list(conversation))
    threshold = branch_threshold if branch_threshold is not None else int(os.getenv("BRANCH_K_THRESHOLD", "10"))
    candidates = set(candidate_entities or state.get("retrieved_candidates", set()) or set())
    structured_results = state.get("tool_structured_results", {})

    user_msg_index = 0
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "assistant" and content:
            message["content"] = strip_reason_blocks(content)
            continue
        if role != "user" or "Tool Execution Results" not in content:
            if role == "user":
                user_msg_index += 1
            continue

        tool_part, hint_part = _split_feedback_content(content)
        turn_structured = structured_results.get(f"turn_{user_msg_index}", [])
        match_results = [item for item in turn_structured if item.get("tool_name") in {"match_pattern", "action"}]
        if match_results and candidates:
            total_branches = 0
            for result in match_results:
                structured = result.get("structured_data", {}) or {}
                branches = structured.get("branches", [])
                total_branches += len(branches) if branches else len(structured.get("leaf_entities", []))
            if total_branches > threshold:
                tool_part = compress_with_structured_data(
                    match_results,
                    candidates,
                    fallback_text=tool_part,
                )

        message["content"] = tool_part + hint_part
        user_msg_index += 1

    return messages


def _split_feedback_content(content: str) -> tuple[str, str]:
    hint_start = -1
    for marker in ("\n[REPAIR MODE", "\n[STAGE", "\n[DISCOVERY PHASE", "\n[Phase"):
        idx = content.find(marker)
        if idx != -1 and (hint_start == -1 or idx < hint_start):
            hint_start = idx
    if hint_start == -1:
        return content, ""
    return content[:hint_start], content[hint_start:]
