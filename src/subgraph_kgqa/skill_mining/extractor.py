from __future__ import annotations

import ast
import asyncio
import json
import re
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import aiohttp

from subgraph_kgqa.inference.backend import GraphBackendClient
from subgraph_kgqa.rl.plugin import parse_prompt_context

from .schemas import (
    AtomicExperienceCard,
    BackendSchemaSnapshot,
    RawAttemptRecord,
    SourceCaseCard,
    read_json,
    write_json,
)

from ._helpers import (
    _extract_json_object,
    _extract_question_surface,
    _safe_slug,
    _utc_now,
)


def _split_listish(raw: str) -> List[str]:
    value = raw.strip()
    if not value or value == "-":
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass
    return [item.strip() for item in value.split(",") if item.strip()]


def _extract_available_domains(question_text: str) -> List[str]:
    match = re.search(
        r"Domains available for explore_schema:\n(.*?)(?:\n\s*\n\[Retrieval Context\]|\Z)",
        question_text,
        re.DOTALL,
    )
    if not match:
        return []
    domains: List[str] = []
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            domains.append(stripped[2:].strip())
    return domains


class SkillMiningExtractor:
    def __init__(
        self,
        *,
        output_root: Path,
        data_path: Path,
        batch_report_path: Path,
        smoke_reports_dir: Path,
        kg_api_url: str,
        use_llm: bool = True,
    ) -> None:
        self.output_root = output_root
        self.data_path = data_path
        self.batch_report_path = batch_report_path
        self.smoke_reports_dir = smoke_reports_dir
        self.kg_api_url = kg_api_url.rstrip("/")
        self.use_llm = use_llm

    def load_case_lookup(self) -> Dict[str, Dict[str, Any]]:
        lookup: Dict[str, Dict[str, Any]] = {}
        with self.data_path.open() as handle:
            for line in handle:
                case = json.loads(line)
                lookup[case["id"]] = case
        return lookup

    def parse_batch_report(self) -> Dict[str, Any]:
        text = self.batch_report_path.read_text(encoding="utf-8")
        variant_match = re.search(r"- Variant: `([^`]+)`", text)
        variant = variant_match.group(1) if variant_match else self.batch_report_path.stem

        rows: List[Dict[str, Any]] = []
        for line in text.splitlines():
            if not line.startswith("| WebQ"):
                continue
            parts = [part.strip() for part in line.strip().strip("|").split("|")]
            if len(parts) < 7:
                continue
            rows.append(
                {
                    "case_id": parts[0],
                    "f1": float(parts[1]),
                    "turns": int(parts[2]),
                    "frontend_errors": int(parts[3]),
                    "repair_mode": None if parts[4] == "-" else parts[4],
                    "predicted_text": parts[5],
                    "error_text": "" if parts[6] == "-" else parts[6],
                    "variant": variant,
                }
            )
        return {"variant": variant, "rows": rows}

    def find_smoke_report(self, case_id: str, variant: str) -> Optional[Path]:
        path = self.smoke_reports_dir / f"inference_runtime_smoke_{case_id}_{variant}.md"
        return path if path.exists() else None

    def parse_smoke_report(self, path: Path) -> Dict[str, Any]:
        text = path.read_text(encoding="utf-8")
        gt_match = re.search(r"- Ground Truth: `([^`]+)`", text)
        pred_match = re.search(r"- Predicted: `([^`]+)`", text)
        domains = sorted(
            {
                domain
                for domain in re.findall(r'explore_schema\(pattern="([^"]+)"\)', text)
                if re.search(r"[A-Za-z]", domain)
            }
        )
        planned_relations = sorted(
            set(
                re.findall(
                    r"\[Relations\]: ([^\n]+)",
                    text,
                )
            )
        )
        flat_planned: List[str] = []
        for item in planned_relations:
            for relation in item.split(","):
                rel = relation.strip()
                if rel:
                    flat_planned.append(rel)
        # Also recover relations directly from plan(...) queries in richer trajectory reports.
        plan_related_chunks = re.findall(r'related=\[([^\]]*)\]', text)
        plan_maybe_chunks = re.findall(r'maybe_related=\[([^\]]*)\]', text)
        for item in plan_related_chunks + plan_maybe_chunks:
            flat_planned.extend(_split_listish(f"[{item}]"))
        constraint_relations = sorted(
            set(re.findall(r'constraint_relations=\[([^\]]*)\]', text))
        )
        flat_constraint_relations: List[str] = []
        for item in constraint_relations:
            flat_constraint_relations.extend(_split_listish(f"[{item}]"))
        constraint_entities = sorted(
            set(re.findall(r'constraint_entities=\[([^\]]*)\]', text))
        )
        flat_constraint_entities: List[str] = []
        for item in constraint_entities:
            flat_constraint_entities.extend(_split_listish(f"[{item}]"))
        excerpt = text[:6000]
        return {
            "ground_truth_answers": _split_listish(gt_match.group(1)) if gt_match else [],
            "predicted_answers": _split_listish(pred_match.group(1)) if pred_match else [],
            "explored_domains": domains,
            "planned_relations": sorted(set(flat_planned)),
            "constraint_relations": sorted(set(flat_constraint_relations)),
            "constraint_entities": sorted(set(flat_constraint_entities)),
            "excerpt": excerpt,
        }

    async def _probe_backend_schema(
        self,
        *,
        case_id: str,
        domains: Iterable[str],
    ) -> List[BackendSchemaSnapshot]:
        domains = [domain for domain in domains if domain]
        if not domains:
            return []
        endpoints = {
            "explore_schema": f"{self.kg_api_url}/v2/explore_schema",
        }
        client = GraphBackendClient(endpoints=endpoints)
        snapshots: List[BackendSchemaSnapshot] = []
        timeout = aiohttp.ClientTimeout(total=60.0)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
            for domain in domains:
                result = await client.execute_tool(
                    session,
                    {"tool_name": "explore_schema", "arguments": {"pattern": domain}},
                    case_id,
                    None,
                )
                snapshots.append(
                    BackendSchemaSnapshot(
                        domain=domain,
                        status=str(result.get("status", "UNKNOWN")),
                        response_text=str(result.get("response_text", ""))[:3000],
                    )
                )
        return snapshots

    @staticmethod
    def _question_fields(question_surface: str, case: Dict[str, Any], prompt_context: Dict[str, Any]) -> Dict[str, Any]:
        lowered = question_surface.lower()
        wh_match = re.match(r"\s*(who|what|which|where|when|how many|how much)", lowered)
        lexical_cues = [
            cue
            for cue in ["current", "latest", "first", "last", "before", "after", "brother", "sister", "wife", "husband", "team", "president", "prime minister", "character", "role"]
            if cue in lowered
        ]
        gt = case.get("ground_truth", {})
        core_entities = gt.get("core_entities", []) or []
        return {
            "wh_word": wh_match.group(1) if wh_match else "",
            "lexical_cues": lexical_cues,
            "anchor_pattern": "multi_anchor" if len(core_entities) > 1 else "single_anchor",
            "has_temporal_cue": any(cue in lowered for cue in ["current", "latest", "first", "last", "before", "after"]),
            "available_domains_count": len(prompt_context.get("available_domains", set())),
            "core_entities": list(core_entities),
            "core_relations": sorted(list(prompt_context.get("core_relations", set())))[:20],
        }

    def _raw_record_paths(self, record: RawAttemptRecord) -> tuple[Path, Path]:
        bucket = "correct" if record.success else "error"
        raw_name = f"{record.created_at}_{_safe_slug(record.case_id)}_{_safe_slug(record.variant)}.json"
        raw_path = self.output_root / "raw_materials" / bucket / raw_name
        source_path = self.output_root / "source_cards" / f"{_safe_slug(record.case_id)}.json"
        return raw_path, source_path

    def _build_source_case_card(self, record: RawAttemptRecord, source_path: Path) -> SourceCaseCard:
        if source_path.exists():
            payload = read_json(source_path)
            card = SourceCaseCard(**payload)
        else:
            card = SourceCaseCard(
                case_id=record.case_id,
                question_text=record.question_text,
                question_fields=record.question_fields,
                prompt_context=record.prompt_context,
            )

        card.question_text = record.question_text
        card.question_fields = record.question_fields
        card.prompt_context = record.prompt_context
        if record.variant not in card.variants_seen:
            card.variants_seen.append(record.variant)
        if record.record_id not in card.raw_attempt_ids:
            card.raw_attempt_ids.append(record.record_id)
        target_ids = card.correct_attempt_ids if record.success else card.error_attempt_ids
        if record.record_id not in target_ids:
            target_ids.append(record.record_id)
        card.explored_domains_seen = sorted(set(card.explored_domains_seen) | set(record.explored_domains))
        card.planned_relations_seen = sorted(set(card.planned_relations_seen) | set(record.planned_relations))
        card.constraint_relations_seen = sorted(set(card.constraint_relations_seen) | set(record.candidate_constraint_relations))
        card.constraint_entities_seen = sorted(set(card.constraint_entities_seen) | set(record.candidate_constraint_entities))
        if record.smoke_report_path:
            card.source_detail_level = "smoke_enriched"
        card.updated_at = record.created_at
        return card

    async def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        from ..llm_client import call_llm
        return await call_llm(messages, max_tokens=1400)

    def _fallback_atomic_card(self, record: RawAttemptRecord) -> AtomicExperienceCard:
        lowered = record.question_text.lower()
        if any(token in lowered for token in ["prime minister", "president", "ceo", "mayor", "governor"]):
            parent_type = "role_holder"
        elif any(token in lowered for token in ["brother", "sister", "wife", "husband", "son", "daughter"]):
            parent_type = "kinship"
        elif "play for" in lowered or "team" in lowered:
            parent_type = "sports_affiliation"
        elif any(token in lowered for token in ["latest", "current", "first", "before", "after"]):
            parent_type = "temporal"
        elif any(token in lowered for token in ["where", "location"]):
            parent_type = "location_lookup"
        elif any(token in lowered for token in ["character", "role", "played"]):
            parent_type = "film_role"
        else:
            parent_type = "typed_entity"

        answer_scope = "undetermined"
        if record.success and len(record.predicted_answers) > 1:
            answer_scope = "all_valid"
        elif record.success:
            answer_scope = "single_best"
        elif record.f1 > 0 and len(record.predicted_answers) > 1:
            answer_scope = "attribute_filtered_subset"

        common_error = []
        if not record.success:
            if record.frontend_errors > 0:
                common_error.append("protocol_or_relation_validation_error")
            elif len(record.predicted_answers) > 1:
                common_error.append("answer_scope_over_retention")
            else:
                common_error.append("underdetermined_failure")

        return AtomicExperienceCard(
            card_id=f"atomic-{record.case_id}-{_safe_slug(record.variant)}",
            case_id=record.case_id,
            source_attempt_ids=[record.record_id],
            outcome_label="correct" if record.success else "error",
            question_type={
                "parent_type": parent_type,
                "candidate_subtype": f"{parent_type}.candidate",
                "confidence": 0.35,
            },
            question_fields=record.question_fields,
            relation_profile={
                "primary_relation_domains": sorted({rel.split(".")[0] for rel in record.planned_relations if "." in rel}),
                "primary_relations": record.planned_relations[:4],
                "maybe_relations": [],
                "tail_semantics": [],
            },
            constraint_profile={
                "constraint_signals": [],
                "candidate_constraint_relations": record.candidate_constraint_relations[:4],
                "constraint_entity_pattern": "non_anchor_verified_only",
                "temporal_policy": "llm_interpretation_needed" if record.question_fields.get("has_temporal_cue") else "not_needed",
            },
            answer_profile={
                "answer_type": record.question_fields.get("wh_word") or "entity",
                "answer_scope_policy": answer_scope,
            },
            common_errors=common_error,
            extraction_notes="Fallback card generated because strict LLM extraction was unavailable.",
            extraction_confidence=0.35,
            evidence_refs={
                "smoke_report_path": record.smoke_report_path,
                "batch_report_path": record.batch_report_path,
            },
        )

    async def _extract_atomic_card(self, record: RawAttemptRecord, source_card: SourceCaseCard) -> AtomicExperienceCard:
        if not self.use_llm:
            return self._fallback_atomic_card(record)

        system_prompt = (
            "You are extracting an updateable KGQA skill card. "
            "Return JSON only. Do not include exact answer strings in reusable fields. "
            "Use abstractions such as relation families, constraint signals, and answer scope policy."
        )
        user_payload = {
            "task": "Produce an atomic experience card from this source attempt.",
            "requirements": [
                "Question type should be dynamic but controlled: parent_type plus candidate_subtype.",
                "question_fields should summarize lexical cues and anchor pattern, not copy long outputs.",
                "relation_profile should focus on relation domains, primary relations, maybe relations, and tail semantics.",
                "primary_relations must keep only answer-bearing relations: relations whose intended tail semantics align with the answer the question asks for.",
                "Do not keep semantically different relations in primary_relations just because they are graph-near or appeared in failed alternative paths.",
                "If a relation tends to reach a different semantic target than the final answer type, move it to maybe_relations or omit it.",
                "constraint_profile should focus on constraint signals, candidate constraint relations, constraint entity pattern, and temporal policy.",
                "answer_profile should focus on answer type and answer_scope_policy.",
                "common_errors should be short codes or short phrases.",
                "Do not copy predicted answers or GT strings into reusable semantic fields.",
            ],
            "json_schema": {
                "question_type": {
                    "parent_type": "string",
                    "candidate_subtype": "string",
                    "confidence": "float_0_to_1",
                },
                "question_fields": {
                    "wh_word": "string",
                    "lexical_cues": ["string"],
                    "anchor_pattern": "single_anchor|multi_anchor",
                    "has_temporal_cue": "bool",
                },
                "relation_profile": {
                    "primary_relation_domains": ["string"],
                    "primary_relations": ["string"],
                    "maybe_relations": ["string"],
                    "tail_semantics": ["string"],
                },
                "constraint_profile": {
                    "constraint_signals": ["string"],
                    "candidate_constraint_relations": ["string"],
                    "constraint_entity_pattern": "string",
                    "temporal_policy": "string",
                },
                "answer_profile": {
                    "answer_type": "string",
                    "answer_scope_policy": "single_best|current_only|all_valid|attribute_filtered_subset|extremum_only|undetermined",
                },
                "common_errors": ["string"],
                "extraction_notes": "string",
                "extraction_confidence": "float_0_to_1",
            },
            "source_attempt": asdict(record),
            "source_card": asdict(source_card),
        }
        text = await self._call_llm(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
            ]
        )
        payload = _extract_json_object(text)
        if not payload:
            return self._fallback_atomic_card(record)

        return AtomicExperienceCard(
            card_id=f"atomic-{record.case_id}-{_safe_slug(record.variant)}",
            case_id=record.case_id,
            source_attempt_ids=[record.record_id],
            outcome_label="correct" if record.success else "error",
            question_type=payload.get("question_type", {}),
            question_fields=payload.get("question_fields", record.question_fields),
            relation_profile=payload.get("relation_profile", {}),
            constraint_profile=payload.get("constraint_profile", {}),
            answer_profile=payload.get("answer_profile", {}),
            common_errors=payload.get("common_errors", []),
            extraction_notes=payload.get("extraction_notes", ""),
            extraction_confidence=float(payload.get("extraction_confidence", 0.0) or 0.0),
            evidence_refs={
                "smoke_report_path": record.smoke_report_path,
                "batch_report_path": record.batch_report_path,
                "detail_level": source_card.source_detail_level,
            },
        )

    async def _process_row(
        self,
        row: Dict[str, Any],
        case_lookup: Dict[str, Dict[str, Any]],
    ) -> Dict[str, str]:
        case_id = row["case_id"]
        case = case_lookup[case_id]
        question_text = next((msg["content"] for msg in case.get("messages", []) if msg.get("role") == "user"), "")
        question_surface = _extract_question_surface(question_text)
        prompt_context = parse_prompt_context(question_text)
        prompt_context["available_domains"] = set(prompt_context.get("available_domains", set())) | set(
            _extract_available_domains(question_text)
        )
        smoke_path = self.find_smoke_report(case_id, row["variant"])
        smoke_info = self.parse_smoke_report(smoke_path) if smoke_path else {}
        predicted_answers = smoke_info.get("predicted_answers", [])
        gt_answers = smoke_info.get("ground_truth_answers") or case.get("ground_truth", {}).get("global_truth_answers", []) or case.get("solution", [])
        question_fields = self._question_fields(question_surface, case, prompt_context)
        backend_snapshots = await self._probe_backend_schema(
            case_id=case_id,
            domains=smoke_info.get("explored_domains", []),
        )
        created_at = _utc_now()
        record = RawAttemptRecord(
            record_id=f"raw-{case_id}-{_safe_slug(row['variant'])}-{uuid.uuid4().hex[:8]}",
            created_at=created_at,
            case_id=case_id,
            variant=row["variant"],
            question_text=question_surface,
            ground_truth_answers=list(gt_answers),
            predicted_answers=predicted_answers,
            predicted_text=row["predicted_text"],
            f1=float(row["f1"]),
            success=float(row["f1"]) >= 1.0,
            turns=int(row["turns"]),
            frontend_errors=int(row["frontend_errors"]),
            repair_mode=row.get("repair_mode"),
            error_text=row.get("error_text", ""),
            question_fields=question_fields,
            prompt_context={
                "available_domains": sorted(list(prompt_context.get("available_domains", set()))),
                "core_relations": sorted(list(prompt_context.get("core_relations", set()))),
            },
            explored_domains=smoke_info.get("explored_domains", []),
            planned_relations=smoke_info.get("planned_relations", []),
            candidate_constraint_relations=smoke_info.get("constraint_relations", []),
            candidate_constraint_entities=smoke_info.get("constraint_entities", []),
            smoke_report_path=str(smoke_path) if smoke_path else None,
            batch_report_path=str(self.batch_report_path),
            smoke_excerpt=smoke_info.get("excerpt", ""),
            backend_schema_snapshots=backend_snapshots,
        )

        raw_path, source_path = self._raw_record_paths(record)
        write_json(raw_path, record)

        source_card = self._build_source_case_card(record, source_path)
        write_json(source_path, source_card)

        atomic_card = await self._extract_atomic_card(record, source_card)
        atomic_path = self.output_root / "atomic_cards" / f"{_safe_slug(case_id)}__{_safe_slug(row['variant'])}.json"
        write_json(atomic_path, atomic_card)
        return {
            "case_id": case_id,
            "raw_path": str(raw_path),
            "source_path": str(source_path),
            "atomic_path": str(atomic_path),
            "outcome": "correct" if record.success else "error",
        }

    async def run(self, *, limit: Optional[int] = None, concurrency: int = 4) -> Dict[str, Any]:
        batch = self.parse_batch_report()
        rows = batch["rows"][:limit] if limit is not None else batch["rows"]
        case_lookup = self.load_case_lookup()
        semaphore = asyncio.Semaphore(concurrency)

        async def guarded(row: Dict[str, Any]) -> Dict[str, str]:
            async with semaphore:
                return await self._process_row(row, case_lookup)

        results = await asyncio.gather(*(guarded(row) for row in rows))
        correct_count = sum(1 for item in results if item["outcome"] == "correct")
        error_count = len(results) - correct_count
        summary = {
            "batch_report": str(self.batch_report_path),
            "variant": batch["variant"],
            "cases": len(results),
            "correct_raw_materials": correct_count,
            "error_raw_materials": error_count,
            "items": results,
        }
        write_json(self.output_root / "extraction_summary.json", summary)
        return summary
