from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


from ._helpers import (
    _extract_json_object,
    _to_list,
)
from ..llm_client import call_llm


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cn_scope(policy: str) -> str:
    mapping = {
        "single_best": "通常收敛到单个最合适答案",
        "all_valid": "通常保留所有满足题意的答案",
        "current_only": "若无额外时间限定，通常优先当前/现行答案",
        "attribute_filtered_subset": "通常先按属性过滤，再输出剩余子集",
        "extremum_only": "通常只保留最早/最晚/最大/最小这类极值答案",
        "undetermined": "答案范围需要结合题意和候选再判断",
    }
    return mapping.get(policy, policy or "答案范围暂未稳定")


def _normalize_answer_policy_text(policy: str) -> str:
    cleaned = (policy or "").strip()
    mapping = {
        "current": "若题目未显式要求历史范围，通常优先当前/最近有效答案。",
        "current_only": "若题目未显式要求历史范围，通常优先当前/最近有效答案。",
        "latest": "若题目未显式要求历史范围，通常优先最新/最近有效答案。",
        "single_best": "通常收敛到单个最合适答案。",
        "all_valid": "通常保留所有满足题意的答案。",
        "attribute_filtered_subset": "通常先按关键属性过滤，再输出剩余子集。",
        "extremum_only": "通常只保留最早/最晚/最大/最小这类极值答案。",
        "undetermined": "答案范围需要结合题意和候选再判断。",
    }
    return mapping.get(cleaned, cleaned or "答案范围需要结合题意和候选再判断。")


def _short_type(question_type: Dict[str, Any]) -> str:
    parent_type = str(question_type.get("parent_type", "")).strip()
    subtype = str(question_type.get("candidate_subtype", "")).strip()
    if parent_type and subtype:
        return f"{parent_type} / {subtype}"
    return subtype or parent_type or "未分类"


def _fallback_runtime_card(source_card: Dict[str, Any], atomic_card: Dict[str, Any]) -> Dict[str, Any]:
    relation_profile = atomic_card.get("relation_profile", {})
    constraint_profile = atomic_card.get("constraint_profile", {})
    answer_profile = atomic_card.get("answer_profile", {})
    prompt_context = source_card.get("prompt_context", {})

    domains = _to_list(relation_profile.get("primary_relation_domains")) or _to_list(prompt_context.get("available_domains"))[:6]
    likely_relations = _to_list(relation_profile.get("primary_relations"))
    backup_relations = _to_list(relation_profile.get("maybe_relations"))
    constraint_relations = _to_list(constraint_profile.get("candidate_constraint_relations"))
    question_fields = atomic_card.get("question_fields", {}) or source_card.get("question_fields", {})
    temporal_policy = str(constraint_profile.get("temporal_policy", "")).strip()
    constraint_entity_pattern = str(constraint_profile.get("constraint_entity_pattern", "")).strip()

    if source_card.get("constraint_entities_seen"):
        possible_constraint_entities = list(source_card["constraint_entities_seen"])
    elif constraint_entity_pattern == "non_anchor_verified_only":
        possible_constraint_entities = ["题面中除锚点外、已验证且直接约束答案的实体"]
    elif question_fields.get("anchor_pattern") == "multi_anchor":
        possible_constraint_entities = ["题面中除主锚点外的已验证实体"]
    else:
        possible_constraint_entities = ["通常无显式约束实体，更多依赖关系或属性约束"]

    answer_policy = _normalize_answer_policy_text(str(answer_profile.get("answer_scope_policy", "")).strip())
    if temporal_policy == "current_only":
        answer_policy = "若题目未显式要求历史全集，通常优先当前/最近有效答案。"
    elif temporal_policy and temporal_policy != "not_needed":
        answer_policy = f"{answer_policy} 时间解释策略：{temporal_policy}。"

    common_pitfalls = _to_list(atomic_card.get("common_errors"))
    if not common_pitfalls:
        common_pitfalls = ["不要把语义不同但图上相邻的关系混进主体关系。"]

    lines = [
        "以下是从skill库中检索的经验：",
        f"- 相关问题类型：{_short_type(atomic_card.get('question_type', {}))}",
        f"- 相关域：{', '.join(domains) if domains else '未稳定'}",
        f"- 可能的主体关系：{', '.join(likely_relations) if likely_relations else '未稳定'}",
        f"- 可能的备选关系：{', '.join(backup_relations) if backup_relations else '如当前主体关系失败，再考虑其它同语义路径'}",
        f"- 这种题目可能存在用于约束的关系：{', '.join(constraint_relations) if constraint_relations else '若候选过多，优先找能区分候选的属性关系'}",
        f"- 这种题目可能存在用于约束的实体或值：{', '.join(possible_constraint_entities)}",
        f"- 作答策略：{answer_policy}",
        f"- 常见错误：{', '.join(common_pitfalls)}",
    ]
    return {
        "question_type_label": _short_type(atomic_card.get("question_type", {})),
        "relevant_domains": domains,
        "likely_answer_relations": likely_relations,
        "backup_relations": backup_relations,
        "possible_constraint_relations": constraint_relations,
        "possible_constraint_entities": possible_constraint_entities,
        "answer_policy": answer_policy,
        "common_pitfalls": common_pitfalls,
        "injection_text": "\n".join(lines),
    }


async def _call_llm(messages: List[Dict[str, str]]) -> str:
    return await call_llm(messages, max_tokens=1200)


async def _synthesize_runtime_card(source_card: Dict[str, Any], atomic_card: Dict[str, Any], *, use_llm: bool) -> Dict[str, Any]:
    fallback = _fallback_runtime_card(source_card, atomic_card)
    if not use_llm:
        return fallback

    prompt_payload = {
        "task": "Rewrite this case into a runtime skill-retrieval note for a KGQA agent.",
        "rules": [
            "Return JSON only.",
            "Do not mention source_case, source_attempt_ids, provenance, extraction confidence, or where the experience came from.",
            "Write as a reusable retrieved experience note for runtime injection.",
            "Do not expose exact surface wording cues from the current question. Summarize the reusable problem family instead.",
            "likely_answer_relations must keep only answer-bearing relations: relations that are semantically intended to land on the answer type and answer scope asked by the question.",
            "Remove graph-near but semantically off-target relations from likely_answer_relations.",
            "Do not keep relations that systematically land on a different answer subspace, such as school team vs current pro team, historical holder vs current holder, or parent event node vs asked entity.",
            "backup_relations may contain close alternatives, but do not keep relations whose tail semantics drift away from the asked answer type or answer scope.",
            "answer_policy should be a short Chinese instruction sentence, not a one-word label.",
            "possible_constraint_entities may describe a pattern or a likely value type; it does not have to be a concrete entity string.",
            "Use concise Chinese.",
        ],
        "json_schema": {
            "question_type_label": "string",
            "relevant_domains": ["string"],
            "likely_answer_relations": ["string"],
            "backup_relations": ["string"],
            "possible_constraint_relations": ["string"],
            "possible_constraint_entities": ["string"],
            "answer_policy": "string",
            "common_pitfalls": ["string"],
            "injection_text": "multiline string starting with '以下是从skill库中检索的经验：'",
        },
        "question_text": source_card.get("question_text", ""),
        "source_card": source_card,
        "atomic_card": atomic_card,
        "fallback_card": fallback,
    }

    text = await _call_llm(
        [
            {
                "role": "system",
                "content": "You are curating runtime KGQA skill notes. Return JSON only.",
            },
            {
                "role": "user",
                "content": json.dumps(prompt_payload, ensure_ascii=False, indent=2),
            },
        ]
    )
    payload = _extract_json_object(text)
    if not payload:
        return fallback
    merged = dict(fallback)
    for key in [
        "question_type_label",
        "relevant_domains",
        "likely_answer_relations",
        "backup_relations",
        "possible_constraint_relations",
        "possible_constraint_entities",
        "answer_policy",
        "common_pitfalls",
        "injection_text",
    ]:
        if key in payload and payload[key]:
            merged[key] = payload[key]
    return merged


def _render_runtime_markdown(source_card: Dict[str, Any], runtime_card: Dict[str, Any]) -> str:
    lines = [
        f"# {source_card.get('case_id', 'unknown')}",
        "",
        f"- 题目：`{source_card.get('question_text', '').strip()}`",
        "",
        "## 检索经验",
        "",
        runtime_card.get("injection_text", "").strip(),
        "",
        "## 结构化摘要",
        "",
        f"- 相关问题类型：{runtime_card.get('question_type_label', '未分类')}",
        f"- 相关域：{', '.join(_to_list(runtime_card.get('relevant_domains'))) if _to_list(runtime_card.get('relevant_domains')) else '未稳定'}",
        f"- 可能的主体关系：{', '.join(_to_list(runtime_card.get('likely_answer_relations'))) if _to_list(runtime_card.get('likely_answer_relations')) else '未稳定'}",
        f"- 可能的备选关系：{', '.join(_to_list(runtime_card.get('backup_relations'))) if _to_list(runtime_card.get('backup_relations')) else '无明显备选'}",
        f"- 约束关系：{', '.join(_to_list(runtime_card.get('possible_constraint_relations'))) if _to_list(runtime_card.get('possible_constraint_relations')) else '未稳定'}",
        f"- 约束实体/值：{', '.join(_to_list(runtime_card.get('possible_constraint_entities'))) if _to_list(runtime_card.get('possible_constraint_entities')) else '未稳定'}",
        f"- 作答策略：{_normalize_answer_policy_text(str(runtime_card.get('answer_policy', '')))}",
        f"- 易错点：{', '.join(_to_list(runtime_card.get('common_pitfalls'))) if _to_list(runtime_card.get('common_pitfalls')) else '未稳定'}",
        "",
    ]
    return "\n".join(lines)


def _build_cluster_summary(rows: Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]) -> str:
    grouped: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = defaultdict(list)
    for source_card, runtime_card in rows:
        grouped[str(runtime_card.get("question_type_label", "未分类"))].append((source_card, runtime_card))

    lines = [
        "# Runtime Skill Summary",
        "",
        "| 类型 | 样本数 | 常见域 | 常见主体关系 | 常见作答策略 |",
        "|---|---:|---|---|---|",
    ]
    for key, items in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        domain_counter = Counter()
        relation_counter = Counter()
        policy_counter = Counter()
        for _, runtime_card in items:
            for domain in _to_list(runtime_card.get("relevant_domains")):
                domain_counter[domain] += 1
            for relation in _to_list(runtime_card.get("likely_answer_relations")):
                relation_counter[relation] += 1
            policy = str(runtime_card.get("answer_policy", "")).strip()
            if policy:
                policy_counter[policy] += 1
        domains = ", ".join(domain for domain, _ in domain_counter.most_common(3)) or "-"
        relations = ", ".join(relation for relation, _ in relation_counter.most_common(3)) or "-"
        policies = " / ".join(policy for policy, _ in policy_counter.most_common(2)) or "-"
        lines.append(f"| {key} | {len(items)} | {domains} | {relations} | {policies} |")
    return "\n".join(lines) + "\n"


async def render_readable_outputs(
    skills_root: Path,
    *,
    use_llm: bool = True,
    case_ids: List[str] | None = None,
) -> Dict[str, Any]:
    source_root = skills_root / "source_cards"
    atomic_root = skills_root / "atomic_cards"
    readable_root = skills_root / "readable_cards"
    runtime_json_root = skills_root / "runtime_cards"
    readable_root.mkdir(parents=True, exist_ok=True)
    runtime_json_root.mkdir(parents=True, exist_ok=True)

    rows: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    written: List[str] = []
    selected = set(case_ids or [])
    for atomic_path in sorted(atomic_root.glob("*.json")):
        atomic_card = _read_json(atomic_path)
        case_id = atomic_card.get("case_id", "")
        if not case_id:
            continue
        if selected and case_id not in selected:
            continue
        source_path = source_root / f"{case_id}.json"
        if not source_path.exists():
            continue
        source_card = _read_json(source_path)
        runtime_card = await _synthesize_runtime_card(source_card, atomic_card, use_llm=use_llm)
        rows.append((source_card, runtime_card))

        readable_path = readable_root / f"{case_id}.md"
        readable_path.write_text(_render_runtime_markdown(source_card, runtime_card), encoding="utf-8")
        written.append(str(readable_path))

        runtime_json_path = runtime_json_root / f"{case_id}.json"
        runtime_json_path.write_text(json.dumps(runtime_card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary_path = readable_root / "summary.md"
    summary_path.write_text(_build_cluster_summary(rows), encoding="utf-8")
    return {
        "cases": len(rows),
        "written_cards": written,
        "summary_path": str(summary_path),
        "runtime_json_root": str(runtime_json_root),
    }


def render_readable_outputs_sync(
    skills_root: Path,
    *,
    use_llm: bool = True,
    case_ids: List[str] | None = None,
) -> Dict[str, Any]:
    return asyncio.run(render_readable_outputs(skills_root, use_llm=use_llm, case_ids=case_ids))
