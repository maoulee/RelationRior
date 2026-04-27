from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


@dataclass
class BackendSchemaSnapshot:
    domain: str
    status: str
    response_text: str


@dataclass
class RawAttemptRecord:
    record_id: str
    created_at: str
    case_id: str
    variant: str
    question_text: str
    ground_truth_answers: List[str]
    predicted_answers: List[str]
    predicted_text: str
    f1: float
    success: bool
    turns: int
    frontend_errors: int
    repair_mode: Optional[str]
    error_text: str
    question_fields: Dict[str, Any]
    prompt_context: Dict[str, Any]
    explored_domains: List[str] = field(default_factory=list)
    planned_relations: List[str] = field(default_factory=list)
    candidate_constraint_relations: List[str] = field(default_factory=list)
    candidate_constraint_entities: List[str] = field(default_factory=list)
    smoke_report_path: Optional[str] = None
    batch_report_path: Optional[str] = None
    smoke_excerpt: str = ""
    backend_schema_snapshots: List[BackendSchemaSnapshot] = field(default_factory=list)


@dataclass
class SourceCaseCard:
    case_id: str
    question_text: str
    question_fields: Dict[str, Any]
    prompt_context: Dict[str, Any]
    variants_seen: List[str] = field(default_factory=list)
    raw_attempt_ids: List[str] = field(default_factory=list)
    correct_attempt_ids: List[str] = field(default_factory=list)
    error_attempt_ids: List[str] = field(default_factory=list)
    explored_domains_seen: List[str] = field(default_factory=list)
    planned_relations_seen: List[str] = field(default_factory=list)
    constraint_relations_seen: List[str] = field(default_factory=list)
    constraint_entities_seen: List[str] = field(default_factory=list)
    source_detail_level: str = "batch_only"
    updated_at: str = ""


@dataclass
class AtomicExperienceCard:
    card_id: str
    case_id: str
    source_attempt_ids: List[str]
    outcome_label: str
    question_type: Dict[str, Any]
    question_fields: Dict[str, Any]
    relation_profile: Dict[str, Any]
    constraint_profile: Dict[str, Any]
    answer_profile: Dict[str, Any]
    common_errors: List[str] = field(default_factory=list)
    extraction_notes: str = ""
    extraction_confidence: float = 0.0
    evidence_refs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpectedAnswerType:
    """Structured answer type prediction for intent-aware retrieval."""
    entity_category: str = ""       # "person", "organization", "location", "team", "language", etc.
    cardinality: str = ""           # "single", "multiple", "unknown"
    examples: List[str] = field(default_factory=list)


@dataclass
class TemporalScope:
    """Structured temporal intent for disambiguation."""
    type: str = ""                  # "year_specific", "as_of", "latest", "all_time", "none"
    reference: str = ""             # "2010", "current", etc.
    operator: str = ""              # "during", "before", "after", "as_of"


@dataclass
class ScopeInclusions:
    """Verb/scope disambiguation for ambiguous queries."""
    entity_types: List[str] = field(default_factory=list)  # ["club", "national_team"]
    explanation: str = ""


@dataclass
class CaseSkillCard:
    case_id: str
    question: str
    question_type: str
    retrieval_fields: Dict[str, Any]
    core_relation_domains: List[str]
    core_relations: List[str]
    constraint_guidance: List[str]
    answer_strategy: Dict[str, Any]
    action_space_experience: str = ""
    final_selection_experience: List[str] = field(default_factory=list)
    candidate_constraint_relations: List[str] = field(default_factory=list)
    candidate_constraint_entities: List[str] = field(default_factory=list)
    common_pitfalls: List[str] = field(default_factory=list)
    notes: str = ""
    # K-run instability-aware fields
    intent_clarification: str = ""
    common_misreadings: List[str] = field(default_factory=list)
    instability_triggers: List[str] = field(default_factory=list)
    wrong_but_related_answer_families: List[str] = field(default_factory=list)
    run_count: int = 0
    instability_score: float = 0.0
    # Intent signature fields (backward-compatible, all Optional)
    expected_answer_type: Optional[ExpectedAnswerType] = None
    temporal_scope: Optional[TemporalScope] = None
    scope_inclusions: Optional[ScopeInclusions] = None
    intent_signature: str = ""      # "temporal:year_specific|answer:person_list|scope:club+national"
    ambiguity_flags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Auto-convert dict payloads to nested dataclasses for backward compat."""
        if isinstance(self.expected_answer_type, dict):
            self.expected_answer_type = ExpectedAnswerType(**self.expected_answer_type)
        if isinstance(self.temporal_scope, dict):
            self.temporal_scope = TemporalScope(**self.temporal_scope)
        if isinstance(self.scope_inclusions, dict):
            self.scope_inclusions = ScopeInclusions(**self.scope_inclusions)


@dataclass
class NegativeSkillCard:
    """A negative experience skill: records what went wrong and what should have been done."""
    source_case_id: str
    question: str
    error_type: str                                       # "action_error", "plan_error", "no_answer", etc.
    wrong_plan_relations: List[str] = field(default_factory=list)   # relations that led to wrong answers
    correct_plan_relations: List[str] = field(default_factory=list) # relations that should have been chosen
    wrong_action_relation: str = ""                       # action relation that failed
    correct_action_relation: str = ""                     # action relation that should have been used
    error_pattern: str = ""                               # human-readable description of what went wrong
    correct_approach: str = ""                            # human-readable description of the right approach
    ground_truth: List[str] = field(default_factory=list)
    wrong_predicted: List[str] = field(default_factory=list)

    @classmethod
    def from_negative_skill_file(cls, data: Dict[str, Any]) -> "NegativeSkillCard":
        """Construct from a negative skill JSON file (webqsp_negative_skills format)."""
        return cls(
            source_case_id=data.get("source_case_id", ""),
            question=data.get("question", ""),
            error_type=data.get("error_type", "unknown"),
            wrong_plan_relations=data.get("plan_relations_chosen", []),
            correct_plan_relations=data.get("correct_plan_relations", []),
            wrong_action_relation=data.get("action_relation_chosen", ""),
            correct_action_relation=data.get("correct_action_relation", ""),
            error_pattern=data.get("error_pattern", ""),
            correct_approach=data.get("correct_approach", ""),
            ground_truth=data.get("ground_truth", []),
            wrong_predicted=data.get("v2_predicted", []),
        )


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {key: _to_jsonable(value) for key, value in asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(item) for item in obj]
    return obj


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
