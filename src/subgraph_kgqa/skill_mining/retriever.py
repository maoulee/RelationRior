from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


from ._helpers import _extract_json_object
from ..llm_client import call_llm

from .case_skill import (
    extract_question_surface,
    synthesize_case_skill_from_dataset_case,
    upsert_case_skill_outputs,
)
from .schemas import CaseSkillCard, NegativeSkillCard, read_json


STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "who",
    "what",
    "where",
    "when",
    "which",
    "of",
    "for",
    "to",
    "in",
    "on",
    "at",
    "by",
    "with",
    "and",
    "or",
}

PATTERN_CUES = (
    "play for",
    "team",
    "college",
    "club",
    "first",
    "current",
    "now",
    "prime minister",
    "brother",
    "sister",
    "character",
    "capital",
)

DEFAULT_GTE_MODEL_PATH = Path(os.getenv("KGQA_SKILL_EMBED_MODEL", "/zhaoshu/llm/gte_large"))
DEFAULT_GTE_DEVICE = os.getenv("KGQA_SKILL_EMBED_DEVICE", "cpu").strip() or "cpu"
DEFAULT_GTE_BATCH_SIZE = int(os.getenv("KGQA_SKILL_EMBED_BATCH_SIZE", "32"))

_PRECOMPUTED_Q_EMB_CACHE: Dict[str, tuple[List[str], Any]] = {}
_GTE_MODEL_CACHE: Dict[tuple[str, str], tuple[Any, Any]] = {}
_DATASET_LOOKUP_CACHE: Dict[str, Dict[str, Dict[str, Any]]] = {}


def _skill_reasoning_injection_mode() -> str:
    value = os.getenv("KGQA_SKILL_REASONING_INJECTION_MODE", "aggregate").strip().lower()
    return value if value in {"aggregate", "per_skill"} else "aggregate"



def _skill_audit_enabled() -> bool:
    return os.getenv("KGQA_ENABLE_SKILL_AUDIT", "").strip() in {"1", "true", "yes", "on"}


def _skill_audit_mode() -> str:
    """Return the current skill audit mode.

    Values: "off" (default), "always", "conflict_only".
    Controlled by KGQA_SKILL_AUDIT_MODE env var.
    """
    value = os.getenv("KGQA_SKILL_AUDIT_MODE", "off").strip().lower()
    if value in {"off", "always", "conflict_only"}:
        return value
    # Backward compatibility: the older boolean flag meant "always" audit.
    if _skill_audit_enabled():
        return "always"
    return "off"


def _skill_audit_keep_k() -> int:
    try:
        return max(1, int(os.getenv("KGQA_SKILL_AUDIT_KEEP_K", "3")))
    except (ValueError, TypeError):
        return 3


@dataclass
class RetrievedSkillBundle:
    target_question: str
    retrieved_case_ids: List[str]
    retrieved_cards: List[CaseSkillCard]
    shortlisted_case_ids: List[str]
    selected_case_ids: List[str]
    relation_stage_hint: str
    reasoning_stage_hint: str
    shortlisted_cards: List[CaseSkillCard] = field(default_factory=list)
    retrieval_note: str = ""
    audit_candidate_ids: List[str] = field(default_factory=list)
    audit_kept_ids: List[str] = field(default_factory=list)
    audit_dropped_ids: List[str] = field(default_factory=list)
    audit_reason: str = ""
    audit_mode: str = "off"
    audit_conflict_detected: bool = False
    audit_trigger_reason: str = ""
    # Negative skill support (feature-gated)
    retrieved_negative_cards: List[NegativeSkillCard] = field(default_factory=list)
    negative_plan_hint: str = ""
    negative_action_hint: str = ""


def _candidate_skill_store_roots(skills_root: Path) -> List[Path]:
    """Return possible roots that may contain case skill JSON/MD files."""
    candidates: List[Path] = []

    def _add(path: Path) -> None:
        if path not in candidates:
            candidates.append(path)

    # Direct corpus root, e.g. skills/webqsp_train_case_skills_en
    if skills_root.exists():
        _add(skills_root)
    # Preferred full-train English corpus when caller passes the generic skills root.
    preferred_full_root = skills_root / "webqsp_train_case_skills_en"
    if preferred_full_root.exists():
        candidates.insert(0, preferred_full_root)
    # Legacy case_skills root.
    legacy_root = skills_root / "case_skills"
    if legacy_root.exists():
        _add(legacy_root)
    return candidates


def _resolve_case_skill_json_path(*, case_id: str, skills_root: Path) -> tuple[Path, Path]:
    """Resolve the JSON path for a case skill and return (json_path, store_root)."""
    candidate_roots = _candidate_skill_store_roots(skills_root)
    for root in candidate_roots:
        json_path = root / f"{case_id}.json"
        if json_path.exists():
            return json_path, root

    # Choose a sensible write target for fallback synthesis.
    if candidate_roots:
        preferred = candidate_roots[0]
        if preferred.name != "case_skills" and preferred.exists():
            return preferred / f"{case_id}.json", preferred
        return preferred / f"{case_id}.json", preferred

    fallback_root = skills_root / "case_skills"
    return fallback_root / f"{case_id}.json", fallback_root


def _tokenize(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if token and token not in STOPWORDS
    ]


def _surface(case: Dict[str, Any]) -> str:
    raw = next((msg.get("content", "") for msg in case.get("messages", []) if msg.get("role") == "user"), "")
    return extract_question_surface(raw)


def _lexical_score(query: str, candidate: str) -> float:
    query_tokens = set(_tokenize(query))
    candidate_tokens = set(_tokenize(candidate))
    if not query_tokens or not candidate_tokens:
        return 0.0
    overlap = len(query_tokens & candidate_tokens)
    jaccard = overlap / max(len(query_tokens | candidate_tokens), 1)
    query_bigrams = {" ".join(pair) for pair in zip(_tokenize(query), _tokenize(query)[1:])}
    cand_bigrams = {" ".join(pair) for pair in zip(_tokenize(candidate), _tokenize(candidate)[1:])}
    bigram_overlap = len(query_bigrams & cand_bigrams)
    cue_bonus = 0.0
    lowered_query = query.lower()
    lowered_candidate = candidate.lower()
    for cue in PATTERN_CUES:
        if cue in lowered_query and cue in lowered_candidate:
            cue_bonus += 0.6
    if re.search(r"\b(19|20)\d{2}\b", lowered_query) and re.search(r"\b(19|20)\d{2}\b", lowered_candidate):
        cue_bonus += 0.4
    return overlap + (2.0 * jaccard) + (0.8 * bigram_overlap) + cue_bonus


async def _call_llm(messages: List[Dict[str, str]]) -> str:
    audit_temp = float(os.getenv("KGQA_SKILL_AUDIT_TEMPERATURE", "0").strip() or "0")
    return await call_llm(messages, max_tokens=700, temperature=audit_temp)


async def detect_skill_conflict(
    *,
    target_question: str,
    cards: Sequence[CaseSkillCard],
) -> tuple[bool, str]:
    """Lightweight LLM check: do the given skills conflict for *target_question*?

    Returns (is_conflicting, reason).
    """
    if len(cards) <= 1:
        return False, "single_skill"

    summaries = []
    for card in cards[:6]:
        summaries.append({
            "case_id": card.case_id,
            "question": card.question,
            "question_type": card.question_type,
            "core_relations": card.core_relations[:5],
        })

    prompt_payload = {
        "task": (
            "Determine whether the following retrieved KGQA skills conflict with each other for the target question. "
            "Skills conflict if they suggest substantially different graph traversal strategies, answer-bearing relations, "
            "or selection rules that would lead to divergent answers."
        ),
        "target_question": target_question,
        "skills": summaries,
        "output_schema": {
            "conflict": True,
            "reason": "string",
        },
    }

    text = await _call_llm(
        [
            {"role": "system", "content": "You detect conflicts between KGQA skill cards. Return JSON only."},
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False, indent=2)},
        ]
    )
    payload = _extract_json_object(text) or {}
    conflict = bool(payload.get("conflict", False))
    reason = str(payload.get("reason", "")).strip()
    return conflict, (reason or "llm_assessed")


def load_dataset_lookup(data_path: Path) -> Dict[str, Dict[str, Any]]:
    cache_key = str(data_path.resolve())
    if cache_key in _DATASET_LOOKUP_CACHE:
        return _DATASET_LOOKUP_CACHE[cache_key]
    lookup: Dict[str, Dict[str, Any]] = {}
    with data_path.open() as handle:
        for line in handle:
            case = json.loads(line)
            lookup[str(case.get("id"))] = case
    _DATASET_LOOKUP_CACHE[cache_key] = lookup
    return lookup


def _infer_dataset_name(data_path: Path) -> str:
    lowered_parts = [part.lower() for part in data_path.parts]
    if "webqsp" in lowered_parts:
        return "webqsp"
    if "cwq" in lowered_parts:
        return "cwq"
    return data_path.parent.name.lower()


def _infer_dataset_name_from_case_ids(case_ids: Sequence[str]) -> str | None:
    counts: Counter[str] = Counter()
    for raw_case_id in list(case_ids)[:32]:
        case_id = str(raw_case_id or "").lower()
        if not case_id:
            continue
        if case_id.startswith("webq"):
            counts["webqsp"] += 1
        elif case_id.startswith("cwq"):
            counts["cwq"] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _infer_split_name(data_path: Path) -> str:
    stem = data_path.stem.lower()
    if "train" in stem:
        return "train"
    if "test" in stem:
        return "test"
    if "val" in stem or "valid" in stem:
        return "val"
    return "train"


def _precomputed_embedding_path(data_path: Path) -> Path:
    dataset_name = _infer_dataset_name(data_path)
    split_name = _infer_split_name(data_path)
    return Path("/zhaoshu/SubgraphRAG-main/retrieve/data_files") / dataset_name / "emb" / "gte-large-en-v1.5" / f"{split_name}.pth"


def _infer_split_name_from_case_ids(case_ids: Sequence[str]) -> str | None:
    counts: Counter[str] = Counter()
    for raw_case_id in list(case_ids)[:32]:
        case_id = str(raw_case_id or "").lower()
        if not case_id:
            continue
        if "test" in case_id:
            counts["test"] += 1
        elif "trn" in case_id or "train" in case_id:
            counts["train"] += 1
        elif "dev" in case_id or "val" in case_id or "valid" in case_id:
            counts["val"] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _candidate_precomputed_embedding_paths(
    *,
    data_path: Path,
    dataset_lookup: Dict[str, Dict[str, Any]],
) -> List[Path]:
    dataset_name = _infer_dataset_name(data_path)
    if dataset_name not in {"webqsp", "cwq"}:
        dataset_name = _infer_dataset_name_from_case_ids(list(dataset_lookup.keys())) or dataset_name
    emb_root = Path("/zhaoshu/SubgraphRAG-main/retrieve/data_files") / dataset_name / "emb" / "gte-large-en-v1.5"
    preferred_splits: List[str] = []

    split_from_case_ids = _infer_split_name_from_case_ids(list(dataset_lookup.keys()))
    if split_from_case_ids:
        preferred_splits.append(split_from_case_ids)

    split_from_path = _infer_split_name(data_path)
    if split_from_path not in preferred_splits:
        preferred_splits.append(split_from_path)

    for split_name in ("train", "test", "val"):
        if split_name not in preferred_splits:
            preferred_splits.append(split_name)

    candidates: List[Path] = []
    for split_name in preferred_splits:
        emb_path = emb_root / f"{split_name}.pth"
        if emb_path.exists() and emb_path not in candidates:
            candidates.append(emb_path)
    return candidates


def _load_best_precomputed_embedding_index(
    *,
    data_path: Path,
    dataset_lookup: Dict[str, Dict[str, Any]],
) -> tuple[List[str], Any] | None:
    for emb_path in _candidate_precomputed_embedding_paths(
        data_path=data_path,
        dataset_lookup=dataset_lookup,
    ):
        index = _load_precomputed_embedding_index(
            emb_path=emb_path,
            dataset_lookup=dataset_lookup,
        )
        if index is not None:
            return index
    return None


def _question_hash(items: Sequence[tuple[str, str]]) -> str:
    h = hashlib.sha1()
    for case_id, question in items:
        h.update(case_id.encode("utf-8"))
        h.update(b"\0")
        h.update(question.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _cache_file_for_questions(*, cache_root: Path, data_path: Path, model_path: Path) -> Path:
    dataset_name = _infer_dataset_name(data_path)
    split_name = _infer_split_name(data_path)
    model_tag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_path.name or "gte")
    return cache_root / "cache" / f"question_embeddings__{dataset_name}__{split_name}__{model_tag}.pt"


def _normalize_rows(tensor: Any) -> Any:
    import torch  # noqa: F401

    norms = tensor.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return tensor / norms


def _load_precomputed_embedding_index(
    *,
    emb_path: Path,
    dataset_lookup: Dict[str, Dict[str, Any]],
) -> tuple[List[str], Any] | None:
    import torch

    cache_key = str(emb_path.resolve())
    if cache_key in _PRECOMPUTED_Q_EMB_CACHE:
        return _PRECOMPUTED_Q_EMB_CACHE[cache_key]
    if not emb_path.exists():
        return None
    payload = torch.load(emb_path, map_location="cpu")
    case_ids: List[str] = []
    vectors = []
    for case_id in dataset_lookup.keys():
        item = payload.get(case_id)
        if not isinstance(item, dict) or "q_emb" not in item:
            continue
        vec = item["q_emb"]
        if getattr(vec, "ndim", 0) == 2:
            vec = vec[0]
        vec = vec.float().cpu()
        case_ids.append(case_id)
        vectors.append(vec)
    if not vectors:
        return None
    matrix = _normalize_rows(torch.stack(vectors, dim=0))
    _PRECOMPUTED_Q_EMB_CACHE[cache_key] = (case_ids, matrix)
    return case_ids, matrix


def _load_or_build_cached_embedding_index(
    *,
    data_path: Path,
    cache_root: Path,
    dataset_lookup: Dict[str, Dict[str, Any]],
    model_path: Path,
    device: str,
    batch_size: int,
) -> tuple[List[str], Any]:
    import torch

    indexed_questions = [(case_id, _surface(case)) for case_id, case in dataset_lookup.items() if _surface(case)]
    cache_file = _cache_file_for_questions(cache_root=cache_root, data_path=data_path, model_path=model_path)
    question_sig = _question_hash(indexed_questions)
    if cache_file.exists():
        try:
            payload = torch.load(cache_file, map_location="cpu")
            if payload.get("question_sig") == question_sig:
                case_ids = [str(item) for item in payload.get("case_ids", [])]
                matrix = payload.get("embeddings")
                if case_ids and matrix is not None and len(case_ids) == int(getattr(matrix, "shape", [0])[0]):
                    return case_ids, matrix.float()
        except Exception:
            pass

    case_ids = [case_id for case_id, _ in indexed_questions]
    texts = [question for _, question in indexed_questions]
    matrix = _encode_questions_with_gte(
        texts=texts,
        model_path=model_path,
        device=device,
        batch_size=batch_size,
    )
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "question_sig": question_sig,
            "case_ids": case_ids,
            "embeddings": matrix.cpu(),
        },
        cache_file,
    )
    return case_ids, matrix


def _get_gte_model(*, model_path: Path, device: str) -> tuple[Any, Any]:
    from transformers import AutoModel, AutoTokenizer

    cache_key = (str(model_path.resolve()), device)
    if cache_key in _GTE_MODEL_CACHE:
        return _GTE_MODEL_CACHE[cache_key]
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    model.to(device)
    _GTE_MODEL_CACHE[cache_key] = (tokenizer, model)
    return tokenizer, model


def _encode_questions_with_gte(
    *,
    texts: Sequence[str],
    model_path: Path,
    device: str,
    batch_size: int,
) -> Any:
    import torch

    tokenizer, model = _get_gte_model(model_path=model_path, device=device)
    all_vectors = []
    for start in range(0, len(texts), max(1, batch_size)):
        batch = list(texts[start : start + max(1, batch_size)])
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            vectors = outputs.last_hidden_state[:, 0, :].float()
            vectors = _normalize_rows(vectors).cpu()
            all_vectors.append(vectors)
    return torch.cat(all_vectors, dim=0) if all_vectors else torch.empty((0, 0), dtype=torch.float32)


def _encode_query_with_gte(*, text: str, model_path: Path, device: str) -> Any:
    matrix = _encode_questions_with_gte(
        texts=[text],
        model_path=model_path,
        device=device,
        batch_size=1,
    )
    return matrix[0]


def _select_similar_case_ids_by_embeddings(
    *,
    target_question: str,
    dataset_lookup: Dict[str, Dict[str, Any]],
    data_path: Path,
    cache_root: Path,
    exclude_case_id: str | None,
    query_case_id: str | None,
    query_data_path: Path | None,
    candidate_limit: int,
    top_k: int,
) -> tuple[List[str], str]:
    import torch

    precomputed = _load_best_precomputed_embedding_index(
        data_path=data_path,
        dataset_lookup=dataset_lookup,
    )
    method_note = "precomputed gte-large-en-v1.5 q_emb cosine"
    if precomputed is not None:
        case_ids, matrix = precomputed
        if exclude_case_id and exclude_case_id in dataset_lookup and exclude_case_id in case_ids:
            query_vec = matrix[case_ids.index(exclude_case_id)]
            method_note = "precomputed gte-large-en-v1.5 q_emb cosine (query from stored case embedding)"
        elif query_case_id and query_data_path is not None:
            query_lookup = load_dataset_lookup(query_data_path)
            query_precomputed = _load_best_precomputed_embedding_index(
                data_path=query_data_path,
                dataset_lookup=query_lookup,
            )
            if query_precomputed is not None:
                query_case_ids, query_matrix = query_precomputed
                if query_case_id in query_case_ids:
                    query_vec = query_matrix[query_case_ids.index(query_case_id)]
                    method_note = "precomputed train q_emb + precomputed query q_emb cosine"
                else:
                    query_vec = _encode_query_with_gte(
                        text=target_question,
                        model_path=DEFAULT_GTE_MODEL_PATH,
                        device=DEFAULT_GTE_DEVICE,
                    )
                    method_note = "precomputed train q_emb + on-the-fly gte_large query cosine"
            else:
                query_vec = _encode_query_with_gte(
                    text=target_question,
                    model_path=DEFAULT_GTE_MODEL_PATH,
                    device=DEFAULT_GTE_DEVICE,
                )
                method_note = "precomputed train q_emb + on-the-fly gte_large query cosine"
        else:
            query_vec = _encode_query_with_gte(
                text=target_question,
                model_path=DEFAULT_GTE_MODEL_PATH,
                device=DEFAULT_GTE_DEVICE,
            )
            method_note = "precomputed train q_emb + on-the-fly gte_large query cosine"
    else:
        case_ids, matrix = _load_or_build_cached_embedding_index(
            data_path=data_path,
            cache_root=cache_root,
            dataset_lookup=dataset_lookup,
            model_path=DEFAULT_GTE_MODEL_PATH,
            device=DEFAULT_GTE_DEVICE,
            batch_size=DEFAULT_GTE_BATCH_SIZE,
        )
        query_vec = _encode_query_with_gte(
            text=target_question,
            model_path=DEFAULT_GTE_MODEL_PATH,
            device=DEFAULT_GTE_DEVICE,
        )
        method_note = "cached gte_large question cosine"

    if not case_ids:
        return [], ""

    scores = torch.mv(matrix.float(), query_vec.float())
    ranked = torch.argsort(scores, descending=True).tolist()
    shortlist: List[str] = []
    for idx in ranked[: max(candidate_limit, top_k) * 3]:
        case_id = case_ids[idx]
        if exclude_case_id and case_id == exclude_case_id:
            continue
        shortlist.append(case_id)
        if len(shortlist) >= max(candidate_limit, top_k):
            break
    return shortlist[:top_k], method_note


def _select_similar_case_ids_by_lexical(
    *,
    target_question: str,
    dataset_lookup: Dict[str, Dict[str, Any]],
    exclude_case_id: str | None = None,
    candidate_limit: int = 20,
    top_k: int = 3,
) -> tuple[List[str], str]:
    scored: List[tuple[float, str, str]] = []
    for case_id, case in dataset_lookup.items():
        if exclude_case_id and case_id == exclude_case_id:
            continue
        surface = _surface(case)
        score = _lexical_score(target_question, surface)
        if score <= 0:
            continue
        scored.append((score, case_id, surface))
    scored.sort(key=lambda item: (-item[0], item[1]))
    candidates = scored[:candidate_limit]
    return [case_id for _, case_id, _ in candidates[:top_k]], "lexical question similarity fallback"


async def select_similar_case_ids(
    *,
    target_question: str,
    dataset_lookup: Dict[str, Dict[str, Any]],
    data_path: Path,
    cache_root: Path,
    exclude_case_id: str | None = None,
    query_case_id: str | None = None,
    query_data_path: Path | None = None,
    candidate_limit: int = 20,
    top_k: int = 3,
) -> tuple[List[str], str]:
    try:
        selected_case_ids, note = _select_similar_case_ids_by_embeddings(
            target_question=target_question,
            dataset_lookup=dataset_lookup,
            data_path=data_path,
            cache_root=cache_root,
            exclude_case_id=exclude_case_id,
            query_case_id=query_case_id,
            query_data_path=query_data_path,
            candidate_limit=candidate_limit,
            top_k=top_k,
        )
        if selected_case_ids:
            return selected_case_ids[:top_k], note
    except Exception as exc:
        note = f"embedding retrieval failed -> lexical fallback ({type(exc).__name__})"
        fallback_ids, _ = _select_similar_case_ids_by_lexical(
            target_question=target_question,
            dataset_lookup=dataset_lookup,
            exclude_case_id=exclude_case_id,
            candidate_limit=candidate_limit,
            top_k=top_k,
        )
        return fallback_ids[:top_k], note
    return _select_similar_case_ids_by_lexical(
        target_question=target_question,
        dataset_lookup=dataset_lookup,
        exclude_case_id=exclude_case_id,
        candidate_limit=candidate_limit,
        top_k=top_k,
    )


async def ensure_case_skill_card(
    *,
    case_id: str,
    dataset_lookup: Dict[str, Dict[str, Any]],
    skills_root: Path,
    use_llm: bool = True,
) -> CaseSkillCard:
    json_path, store_root = _resolve_case_skill_json_path(case_id=case_id, skills_root=skills_root)
    if json_path.exists():
        payload = read_json(json_path)
        return CaseSkillCard(**payload)
    case = dataset_lookup[case_id]
    card = await synthesize_case_skill_from_dataset_case(case, use_llm=use_llm)
    upsert_case_skill_outputs(store_root, card)
    return card


def _unique_nonempty_lines(items: Iterable[str], *, limit: int) -> List[str]:
    results: List[str] = []
    seen = set()
    for item in items:
        text = " ".join(str(item or "").strip().split())
        if not text or text in seen:
            continue
        results.append(text)
        seen.add(text)
        if len(results) >= limit:
            break
    return results


def build_relation_stage_hint(cards: Sequence[CaseSkillCard], *, include_action_space: bool = False) -> str:
    if not cards:
        return ""
    relation_counts: Counter[str] = Counter()
    relation_examples: Dict[str, List[str]] = defaultdict(list)
    domain_counts: Counter[str] = Counter()
    constraint_entity_counts: Counter[str] = Counter()
    constraint_entity_examples: Dict[str, List[str]] = defaultdict(list)
    for card in cards:
        for domain in card.core_relation_domains or []:
            if domain:
                domain_counts[domain] += 1
        for rel in card.core_relations or []:
            if not rel:
                continue
            relation_counts[rel] += 1
            if len(relation_examples[rel]) < 3 and card.question not in relation_examples[rel]:
                relation_examples[rel].append(card.question)
        for entity in getattr(card, "candidate_constraint_entities", []) or []:
            if not entity:
                continue
            constraint_entity_counts[entity] += 1
            if len(constraint_entity_examples[entity]) < 3 and card.question not in constraint_entity_examples[entity]:
                constraint_entity_examples[entity].append(card.question)
    top_domains = [domain for domain, _ in domain_counts.most_common(6)]
    top_relations = [rel for rel, _ in relation_counts.most_common(8)]
    top_constraint_entities = [entity for entity, _ in constraint_entity_counts.most_common(5)]
    lines = [
        "[RETRIEVED SKILL EXPERIENCE: RELATION CANDIDATES]",
        "Below are relation candidates aggregated from similar solved questions.",
        "Treat them as action-space priors only.",
        "Use a skill relation only if it also appears in the CURRENT explored schema or CURRENT suggested relations.",
        "",
    ]
    if include_action_space:
        action_space_experiences = _unique_nonempty_lines(
            (card.action_space_experience for card in cards),
            limit=4,
        )
        lines.extend(
            [
                "- Primary action-space rule:",
                "  - Select the action space whose whole path semantics best matches the question.",
                "  - Judge the overall path meaning and returned answer type, not a single tempting relation token.",
                "  - Prefer one primary action space that already matches the question's main answer semantics.",
                "",
            ]
        )
        if action_space_experiences:
            lines.append("- Action-space experience from similar questions:")
            for item in action_space_experiences:
                lines.append(f"  - {item}")
            lines.append("")
    if top_domains:
        lines.append(f"- Frequent domains in similar questions: `{', '.join(top_domains)}`")
    lines.append("- Candidate relations from similar questions:")
    for rel in top_relations:
        example_text = "; ".join(f"`{q}`" for q in relation_examples.get(rel, [])[:2])
        lines.extend(
            [
                f"  - `{rel}` (seen in {relation_counts[rel]} similar question(s))",
                f"    Example questions: {example_text or '-'}",
            ]
        )
    if top_constraint_entities:
        lines.extend(
            [
                "",
                "- Possible second-entity / title constraints from similar questions:",
            ]
        )
        for entity in top_constraint_entities:
            example_text = "; ".join(f"`{q}`" for q in constraint_entity_examples.get(entity, [])[:2])
            lines.extend(
                [
                    f"  - `{entity}` (seen in {constraint_entity_counts[entity]} similar question(s))",
                    f"    Example questions: {example_text or '-'}",
                ]
            )
        lines.append("  - Only use these when the CURRENT question clearly contains or verifies a matching second entity / title / franchise.")
    lines.extend(
        [
            "",
            "- If a skill relation is absent from the CURRENT schema, ignore it.",
            "- Use the remaining valid relations as `related` / `maybe_related` candidates only.",
        ]
    )
    return "\n".join(lines)


def build_classified_discovery_hint(cards: Sequence[CaseSkillCard], *, max_directions: int = 5, max_cases_per_direction: int = 3) -> str:
    """Build a classified skill hint for Phase 1 (Discovery).

    Groups top-10 skill cards by their primary core_relation, producing
    a "Direction N (relation): Case1, Case2..." summary.  Answer strategy
    is deliberately excluded — that is injected later at Stage 5 only
    for the direction matching the selected action space.

    Returns an empty string when *cards* is empty.
    """
    if not cards:
        return ""

    # --- Group cards by primary core_relation ---
    direction_map: Dict[str, List[CaseSkillCard]] = {}
    for card in cards:
        key = card.core_relations[0] if card.core_relations else "_unknown_"
        direction_map.setdefault(key, []).append(card)

    lines: List[str] = [
        "[SKILL DIRECTION REFERENCE — Solved cases grouped by answer direction]",
        "Below are similar solved questions grouped by their core answer relation.",
        "Use these as directional priors for your planning. Do NOT copy answers directly.",
        "",
    ]

    for idx, (relation, group) in enumerate(
        sorted(direction_map.items(), key=lambda kv: -len(kv[1])), start=1
    ):
        if idx > max_directions:
            break
        lines.append(f"Direction {idx} (`{relation}`) — {len(group)} similar case(s):")
        for card in group[:max_cases_per_direction]:
            q_excerpt = card.question[:80] + ("…" if len(card.question) > 80 else "")
            lines.append(f"  Case `{card.case_id}`: \"{q_excerpt}\"")
            # Show action-space experience if available (no answer strategy)
            if card.action_space_experience:
                lines.append(f"    → {card.action_space_experience[:120]}")
        lines.append("")

    lines.extend([
        "NOTE: These are directional hints only. Verify relations via explore_schema before planning.",
        "At Stage 5, the answer strategy for the direction matching your selected action will be provided.",
    ])
    return "\n".join(lines)


def _skill_selection_summary(card: CaseSkillCard) -> Dict[str, Any]:
    strategy = card.answer_strategy or {}
    return {
        "case_id": card.case_id,
        "question": card.question,
        "question_type": card.question_type,
        "domains": card.core_relation_domains or list(card.retrieval_fields.get("domains", [])),
        "core_relation": card.core_relations[0] if card.core_relations else "",
        "intent_clarification": (card.intent_clarification or "").strip(),
        "action_space_experience": (card.action_space_experience or "").strip(),
        "final_selection_experience": list(card.final_selection_experience or []),
        "action_space_mode": strategy.get("action_space_mode", ""),
        "answering_tendency": strategy.get("answering_tendency", ""),
        "answer_count": strategy.get("answer_count", ""),
        "temporal_scope": strategy.get("temporal_scope", ""),
        "selection_rule": strategy.get("selection_rule", ""),
    }


async def select_stage_skill_ids(
    *,
    target_question: str,
    cards: Sequence[CaseSkillCard],
    max_skills: int = 2,
) -> tuple[List[str], str]:
    if not cards:
        return [], ""

    candidate_summaries = [_skill_selection_summary(card) for card in cards]
    prompt_payload = {
        "task": (
            "Given a target KGQA question and a shortlist of case skills, choose the small set of skills that should "
            "be loaded as references. Prefer skills with similar question pattern, answer-bearing relation, "
            "primary action-space semantics, and final-selection experience."
        ),
        "target_question": target_question,
        "max_skills": max_skills,
        "candidates": candidate_summaries,
        "output_schema": {
            "selected_skill_ids": ["case_id"],
            "reason": "string",
        },
    }
    text = await _call_llm(
        [
            {"role": "system", "content": "You choose reusable KGQA skills from a shortlist. Return JSON only."},
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False, indent=2)},
        ]
    )
    payload = _extract_json_object(text) or {}
    valid_ids = {card.case_id for card in cards}

    def _normalize(items: Any, limit: int) -> List[str]:
        values = [str(item) for item in (items or [])]
        result: List[str] = []
        for item in values:
            if item in valid_ids and item not in result:
                result.append(item)
            if len(result) >= limit:
                break
        return result

    selected_ids = _normalize(payload.get("selected_skill_ids"), max_skills)
    if not selected_ids:
        selected_ids = [card.case_id for card in cards[:max_skills]]
    return selected_ids, str(payload.get("reason", "")).strip()


def build_reasoning_stage_hint(
    cards: Sequence[CaseSkillCard],
    target_question: str = "",
    *,
    include_action_space: bool = False,
) -> str:
    if not cards:
        return ""
    mode = _skill_reasoning_injection_mode()
    experience_lines = _unique_nonempty_lines(
        (item for card in cards for item in (card.final_selection_experience or [])),
        limit=10,
    )
    lines = [
        "[RETRIEVED SKILL EXPERIENCE: FINAL SELECTION]",
        "Concrete selection experiences from similar solved questions:",
        "",
    ]
    if include_action_space:
        action_space_examples = _unique_nonempty_lines(
            (card.action_space_experience for card in cards),
            limit=3,
        )
    else:
        action_space_examples = []
    if action_space_examples:
        lines.append("- Action-space reminders from similar questions:")
        for item in action_space_examples:
            lines.append(f"  - {item}")
        lines.append("")
    if mode == "per_skill":
        lines.append("- Selected skill experiences:")
        for card in cards[:3]:
            lines.append(f"  - From `{card.question}`:")
            items = _unique_nonempty_lines(card.final_selection_experience or [], limit=3)
            if not items:
                lines.append("    - Inspect graph-visible evidence before collapsing candidates.")
            else:
                for item in items:
                    lines.append(f"    - {item}")
    else:
        lines.append("- Final-selection experience from similar questions:")
        if experience_lines:
            for item in experience_lines:
                lines.append(f"  - {item}")
        else:
            lines.extend(
                [
                    "  - Inspect graph-visible evidence inside the chosen primary action space before collapsing candidates.",
                    "  - Do not force a single answer when current evidence does not distinguish the surviving candidates.",
                ]
            )
    lines.extend(
        [
            "",
            "- If graph-visible evidence does not distinguish candidates, do not force a single-answer collapse.",
        ]
    )
    return "\n".join(lines)


def build_action_stage_hint(cards: Sequence[CaseSkillCard]) -> str:
    """Build action-stage guidance from skill cards.

    Injects at stages 2-3 (action selection) to steer the model away from
    common action errors observed in training data.
    Gated by KGQA_ENABLE_ACTION_STAGE_HINTS env var.
    """
    if not cards:
        return ""

    # Action-space experience: what worked in similar cases
    action_lines = _unique_nonempty_lines(
        (card.action_space_experience for card in cards),
        limit=5,
    )

    # Pitfalls from skills whose question pattern resembles the target
    pitfall_lines = _unique_nonempty_lines(
        (item for card in cards for item in (card.common_pitfalls or [])),
        limit=5,
    )

    # Constraint guidance (temporal, ordinal, superlative)
    constraint_lines = _unique_nonempty_lines(
        (item for card in cards for item in (card.constraint_guidance or [])),
        limit=4,
    )

    # Temporal scope hints
    temporal_hints = []
    for card in cards:
        ts = card.temporal_scope
        if ts and ts.type and ts.type != "none":
            ref = f" (reference={ts.reference})" if ts.reference else ""
            temporal_hints.append(f"Temporal scope: {ts.type}{ref}")

    temporal_lines = _unique_nonempty_lines(
        iter(temporal_hints),
        limit=3,
    )

    if not (action_lines or pitfall_lines or constraint_lines or temporal_lines):
        return ""

    lines = [
        "[RETRIEVED SKILL EXPERIENCE: ACTION SELECTION]",
        "Below are action-selection experiences from similar solved questions.",
        "Use them as soft priors to avoid common action errors.",
        "",
    ]

    if action_lines:
        lines.append("- Action-space experience:")
        for item in action_lines:
            lines.append(f"  - {item}")
        lines.append("")

    if pitfall_lines:
        lines.append("- Common pitfalls to avoid:")
        for item in pitfall_lines:
            lines.append(f"  - {item}")
        lines.append("")

    if constraint_lines:
        lines.append("- Constraint guidance:")
        for item in constraint_lines:
            lines.append(f"  - {item}")
        lines.append("")

    if temporal_lines:
        lines.append("- Temporal awareness:")
        for item in temporal_lines:
            lines.append(f"  - {item}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Negative skill support (feature-gated)
# ---------------------------------------------------------------------------

_NEGATIVE_SKILLS_CACHE: Dict[str, List[NegativeSkillCard]] = {}


def _negative_skill_injection_stages() -> List[str]:
    """Return stages where negative skills should be injected.

    Controlled by KGQA_NEGATIVE_SKILL_INJECTION env var.
    Values: comma-separated list of stages, e.g. "plan,action".
    Default: "" (disabled).
    """
    raw = os.getenv("KGQA_NEGATIVE_SKILL_INJECTION", "").strip().lower()
    if not raw:
        return []
    stages = [s.strip() for s in raw.split(",") if s.strip()]
    return [s for s in stages if s in {"plan", "action"}]


def load_negative_skills(negative_skills_root: Path) -> List[NegativeSkillCard]:
    """Load all negative skill cards from a directory.

    Caches per directory path.
    """
    cache_key = str(negative_skills_root.resolve())
    if cache_key in _NEGATIVE_SKILLS_CACHE:
        return _NEGATIVE_SKILLS_CACHE[cache_key]

    cards: List[NegativeSkillCard] = []
    if not negative_skills_root.exists():
        return cards

    for json_path in sorted(negative_skills_root.glob("neg_*.json")):
        try:
            data = read_json(json_path)
            card = NegativeSkillCard.from_negative_skill_file(data)
            if card.source_case_id and card.question:
                cards.append(card)
        except Exception:
            continue

    _NEGATIVE_SKILLS_CACHE[cache_key] = cards
    return cards


def match_negative_skills(
    *,
    target_question: str,
    negative_cards: List[NegativeSkillCard],
    top_k: int = 5,
) -> List[NegativeSkillCard]:
    """Match negative skills by question similarity (lexical)."""
    if not negative_cards:
        return []
    scored: List[tuple[float, NegativeSkillCard]] = []
    for card in negative_cards:
        score = _lexical_score(target_question, card.question)
        if score > 0:
            scored.append((score, card))
    scored.sort(key=lambda x: -x[0])
    return [card for _, card in scored[:top_k]]


def build_negative_plan_hint(negative_cards: Sequence[NegativeSkillCard]) -> str:
    """Build negative experience hint for plan stage.

    Frames negative skills as experience references, not prohibitions:
    "In similar cases, X approach failed, Y approach worked."
    """
    if not negative_cards:
        return ""

    injection_stages = _negative_skill_injection_stages()
    if "plan" not in injection_stages:
        return ""

    lines = [
        "[NEGATIVE EXPERIENCE: PLAN WARNINGS]",
        "In similar questions, the following plan choices led to wrong answers.",
        "Use this as experience reference, not absolute rules.",
        "",
    ]

    for i, card in enumerate(negative_cards[:5], 1):
        lines.append(f"Example {i} (Q: \"{card.question[:60]}...\"):")
        if card.wrong_plan_relations:
            wrong_rels = ", ".join(card.wrong_plan_relations[:4])
            lines.append(f"  - Wrong plan relations: {wrong_rels}")
        if card.correct_plan_relations:
            correct_rels = ", ".join(card.correct_plan_relations[:4])
            lines.append(f"  - Correct plan relations: {correct_rels}")
        if card.error_pattern:
            lines.append(f"  - What went wrong: {card.error_pattern[:120]}")
        if card.correct_approach:
            lines.append(f"  - Better approach: {card.correct_approach[:120]}")
        lines.append("")

    return "\n".join(lines)


def build_negative_action_hint(negative_cards: Sequence[NegativeSkillCard]) -> str:
    """Build negative experience hint for action stage."""
    if not negative_cards:
        return ""

    injection_stages = _negative_skill_injection_stages()
    if "action" not in injection_stages:
        return ""

    # Only include cards with actionable action-stage information
    actionable = [
        card for card in negative_cards
        if card.wrong_action_relation or card.wrong_plan_relations
    ]
    if not actionable:
        return ""

    lines = [
        "[NEGATIVE EXPERIENCE: ACTION WARNINGS]",
        "In similar questions, the following action paths led to wrong results.",
        "",
    ]

    for i, card in enumerate(actionable[:5], 1):
        parts = [f"Example {i} (Q: \"{card.question[:60]}...\"):"]
        if card.wrong_action_relation:
            parts.append(f"  - Failed action relation: {card.wrong_action_relation}")
        if card.correct_action_relation:
            parts.append(f"  - Better action relation: {card.correct_action_relation}")
        if card.wrong_plan_relations:
            # Even plan-level info helps action stage avoid wrong paths
            wrong_rels = ", ".join(card.wrong_plan_relations[:3])
            parts.append(f"  - These plan relations produced wrong candidates: {wrong_rels}")
        if card.correct_plan_relations:
            correct_rels = ", ".join(card.correct_plan_relations[:3])
            parts.append(f"  - These plan relations found correct answers: {correct_rels}")
        lines.extend(parts)
        lines.append("")

    return "\n".join(lines)


async def build_retrieved_skill_bundle(
    *,
    question_text: str,
    data_path: Path,
    skills_root: Path,
    exclude_case_id: str | None = None,
    query_case_id: str | None = None,
    query_data_path: Path | None = None,
    top_k: int = 3,
    use_llm: bool = True,
    negative_skills_root: Path | None = None,
    query_domains: set[str] | None = None,
) -> RetrievedSkillBundle:
    target_question = extract_question_surface(question_text)
    dataset_lookup = load_dataset_lookup(data_path)
    audit_mode = _skill_audit_mode()
    audit_keep_k = _skill_audit_keep_k()

    # Determine shortlist retrieval width based on audit mode.
    audit_active = audit_mode in {"always", "conflict_only"}
    shortlist_limit = max(top_k * 5, 20)  # Wider pool for post-plan relation filtering

    shortlist_case_ids, retrieval_reason = await select_similar_case_ids(
        target_question=target_question,
        dataset_lookup=dataset_lookup,
        data_path=data_path,
        cache_root=skills_root,
        exclude_case_id=exclude_case_id,
        query_case_id=query_case_id,
        query_data_path=query_data_path,
        top_k=shortlist_limit,
    )
    shortlist_cards: List[CaseSkillCard] = []
    for case_id in shortlist_case_ids:
        shortlist_cards.append(
            await ensure_case_skill_card(
                case_id=case_id,
                dataset_lookup=dataset_lookup,
                skills_root=skills_root,
                use_llm=use_llm,
            )
        )

    # --- Domain pre-filtering ---
    # Remove skills from unrelated domains before audit/selection.
    # This reduces cross-domain noise (e.g. TV questions getting medical skill directions).
    domain_filtered = False
    if query_domains:
        filtered_pairs = [
            (cid, card)
            for cid, card in zip(shortlist_case_ids, shortlist_cards)
            if set(card.core_relation_domains or []) & query_domains
        ]
        if filtered_pairs:
            # At least one card matches — use filtered pool
            shortlist_case_ids = [cid for cid, _ in filtered_pairs]
            shortlist_cards = [card for _, card in filtered_pairs]
            domain_filtered = True
        # If ALL cards were filtered out, keep original unfiltered pool (graceful fallback)

    # --- Skill audit stage ---
    audit_candidate_ids: List[str] = []
    audit_kept_ids: List[str] = []
    audit_dropped_ids: List[str] = []
    audit_reason: str = ""
    audit_conflict_detected: bool = False
    audit_trigger_reason: str = ""

    if audit_active and shortlist_cards:
        # Materialise initial top_k candidates from the wider shortlist.
        candidate_ids = shortlist_case_ids[:top_k]
        candidate_cards = shortlist_cards[:top_k]

        if audit_mode == "always":
            # Existing behaviour: always run the LLM selector to prune.
            audit_trigger_reason = "mode_always"
            audit_candidate_ids = list(candidate_ids)
            kept_ids, audit_reason = await select_stage_skill_ids(
                target_question=target_question,
                cards=candidate_cards,
                max_skills=audit_keep_k,
            )
            audit_kept_ids = list(kept_ids)
            audit_dropped_ids = [cid for cid in candidate_ids if cid not in set(kept_ids)]
            kept_set = set(kept_ids)
            selected_cards = [c for c in candidate_cards if c.case_id in kept_set]
            selected_case_ids = [c.case_id for c in selected_cards]

        elif audit_mode == "conflict_only":
            # Step 1: lightweight conflict check on initial top_k.
            is_conflicting, conflict_reason = await detect_skill_conflict(
                target_question=target_question,
                cards=candidate_cards,
            )
            audit_conflict_detected = is_conflicting

            if not is_conflicting:
                # No conflict → keep original top_k cards unchanged.
                audit_trigger_reason = "no_conflict"
                selected_case_ids = list(candidate_ids)
                selected_cards = list(candidate_cards)
            else:
                # Conflict detected → escalate to full audit on wider pool.
                audit_trigger_reason = f"conflict_detected:{conflict_reason}"
                wider_candidate_ids = shortlist_case_ids[:max(top_k * 3, 10)]
                wider_candidate_cards = shortlist_cards[:len(wider_candidate_ids)]
                audit_candidate_ids = list(wider_candidate_ids)
                kept_ids, audit_reason = await select_stage_skill_ids(
                    target_question=target_question,
                    cards=wider_candidate_cards,
                    max_skills=audit_keep_k,
                )
                audit_kept_ids = list(kept_ids)
                audit_dropped_ids = [cid for cid in wider_candidate_ids if cid not in set(kept_ids)]
                kept_set = set(kept_ids)
                selected_cards = [c for c in wider_candidate_cards if c.case_id in kept_set]
                selected_case_ids = [c.case_id for c in selected_cards]
        else:
            selected_case_ids = shortlist_case_ids[:top_k]
            selected_cards = shortlist_cards[:top_k]
    else:
        selected_case_ids = shortlist_case_ids[:top_k]
        selected_cards = shortlist_cards[:top_k]

    note_parts = [
        retrieval_reason,
        f"shortlist={len(shortlist_case_ids)}",
        f"selected_top_k={len(selected_case_ids)}",
    ]
    if domain_filtered:
        note_parts.append("domain_filtered=yes")
    if audit_active:
        note_parts.append(f"audit_mode={audit_mode}")
        note_parts.append(f"audit_candidates={len(audit_candidate_ids)}")
        note_parts.append(f"audit_kept={len(audit_kept_ids)}")
        if audit_conflict_detected:
            note_parts.append("audit_conflict=yes")

    # --- Negative skill loading (feature-gated) ---
    matched_negative: List[NegativeSkillCard] = []
    neg_plan_hint = ""
    neg_action_hint = ""

    if negative_skills_root is not None and _negative_skill_injection_stages():
        all_negative = load_negative_skills(negative_skills_root)
        if all_negative:
            matched_negative = match_negative_skills(
                target_question=target_question,
                negative_cards=all_negative,
                top_k=5,
            )
            if matched_negative:
                neg_plan_hint = build_negative_plan_hint(matched_negative)
                neg_action_hint = build_negative_action_hint(matched_negative)
                note_parts.append(f"negative_matched={len(matched_negative)}")

    return RetrievedSkillBundle(
        target_question=target_question,
        retrieved_case_ids=selected_case_ids,
        retrieved_cards=selected_cards,
        shortlisted_case_ids=shortlist_case_ids,
        shortlisted_cards=shortlist_cards,
        selected_case_ids=selected_case_ids,
        relation_stage_hint=build_relation_stage_hint(selected_cards),
        reasoning_stage_hint=build_reasoning_stage_hint(selected_cards, target_question),
        retrieval_note=" | ".join(part for part in note_parts if part),
        audit_candidate_ids=audit_candidate_ids,
        audit_kept_ids=audit_kept_ids,
        audit_dropped_ids=audit_dropped_ids,
        audit_reason=audit_reason,
        audit_mode=audit_mode,
        audit_conflict_detected=audit_conflict_detected,
        audit_trigger_reason=audit_trigger_reason,
        retrieved_negative_cards=matched_negative,
        negative_plan_hint=neg_plan_hint,
        negative_action_hint=neg_action_hint,
    )


def _extract_relation_family(relation: str) -> str:
    """Extract the domain.type prefix (first 2 segments) from a dotted relation.

    E.g. ``people.person.nationality`` -> ``people.person``.
    Returns the full relation unchanged if it has fewer than 2 segments.
    """
    parts = relation.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return relation


def _extract_relation_tokens(relation: str) -> set:
    """Extract semantic tokens from the relation suffix (after domain.type prefix).

    E.g. ``location.location.partially_containedby`` -> ``{partially, containedby}``.
    This captures finer-grained overlap than family-level matching.
    """
    parts = relation.split(".")
    if len(parts) <= 2:
        return {relation}
    # Split the suffix segments by underscore to get individual tokens
    suffix = "_".join(parts[2:])
    tokens = set(suffix.split("_"))
    return {t for t in tokens if t}


def _compute_token_overlap(card_relations: Sequence[str], plan_tokens: set) -> int:
    """Count how many card core_relations have at least one token overlap with plan."""
    overlap_count = 0
    for rel in card_relations:
        rel_tokens = _extract_relation_tokens(rel)
        if rel_tokens & plan_tokens:
            overlap_count += 1
    return overlap_count


def filter_skills_post_plan(
    cards: Sequence[CaseSkillCard],
    plan_relations: Sequence[str],
    *,
    max_cards: int = 3,
) -> List[CaseSkillCard]:
    """Filter skill cards by relation token overlap with the plan.

    Token-level matching: splits relation suffixes into tokens
    (e.g. ``partially_containedby`` -> ``{partially, containedby}``)
    and checks for overlap with plan relation tokens.

    This is finer-grained than family matching:
    - ``partially_containedby`` overlaps ``containedby`` ✓
    - ``adjoin_s`` does NOT overlap ``containedby`` ✗
    """
    if not cards:
        return []
    if not plan_relations:
        return list(cards[:max_cards])

    # Build token set from plan relations
    plan_tokens: set = set()
    for rel in plan_relations:
        plan_tokens |= _extract_relation_tokens(rel)

    scored_cards: List[tuple[int, CaseSkillCard]] = []
    for card in cards:
        if not card.core_relations:
            # No core_relations -- keep as potentially generic useful (0 overlap).
            scored_cards.append((0, card))
            continue
        overlap = _compute_token_overlap(card.core_relations, plan_tokens)
        scored_cards.append((overlap, card))

    # Keep cards with any token overlap, plus zero-relation cards.
    kept = [
        (overlap, card)
        for overlap, card in scored_cards
        if overlap > 0 or not card.core_relations
    ]

    # Order by overlap count descending (most overlap first), stable for ties.
    kept.sort(key=lambda item: -item[0])

    return [card for _, card in kept[:max_cards]]


def prewarm_precomputed_embedding_indexes(
    *,
    train_data_path: Path,
    query_data_path: Path | None = None,
) -> None:
    """Load precomputed question embedding indexes into process cache once."""
    train_lookup = load_dataset_lookup(train_data_path)
    _load_best_precomputed_embedding_index(
        data_path=train_data_path,
        dataset_lookup=train_lookup,
    )
    if query_data_path is not None:
        query_lookup = load_dataset_lookup(query_data_path)
        _load_best_precomputed_embedding_index(
            data_path=query_data_path,
            dataset_lookup=query_lookup,
        )


if __name__ == "__main__":
    # --- Quick verification of filter_skills_post_plan ---
    from .schemas import CaseSkillCard

    card_overlap = CaseSkillCard(
        case_id="overlap_1",
        question="What nationality is X?",
        question_type="person_to_country.nationality",
        retrieval_fields={},
        core_relation_domains=["people"],
        core_relations=["people.person.nationality", "people.person.place_of_birth"],
        constraint_guidance=[],
        answer_strategy={},
    )
    card_no_overlap = CaseSkillCard(
        case_id="no_overlap_1",
        question="What team does X play for?",
        question_type="person_to_team.affiliation",
        retrieval_fields={},
        core_relation_domains=["sports"],
        core_relations=["sports.pro_athlete.team"],
        constraint_guidance=[],
        answer_strategy={},
    )
    card_no_relations = CaseSkillCard(
        case_id="generic_1",
        question="Who is X?",
        question_type="generic",
        retrieval_fields={},
        core_relation_domains=[],
        core_relations=[],
        constraint_guidance=[],
        answer_strategy={},
    )
    card_partial_overlap = CaseSkillCard(
        case_id="partial_1",
        question="Where was X born?",
        question_type="person_to_location.birthplace",
        retrieval_fields={},
        core_relation_domains=["people"],
        core_relations=["people.person.place_of_birth"],
        constraint_guidance=[],
        answer_strategy={},
    )

    plan_relations = ["people.person.nationality", "location.country.currency"]

    result = filter_skills_post_plan(
        [card_overlap, card_no_overlap, card_no_relations, card_partial_overlap],
        plan_relations,
        max_cards=4,
    )
    result_ids = [c.case_id for c in result]

    # 1) Overlapping cards are kept
    assert "overlap_1" in result_ids, f"overlap_1 should be kept, got {result_ids}"
    # 2) Partial overlap cards are kept
    assert "partial_1" in result_ids, f"partial_1 should be kept, got {result_ids}"
    # 3) No-overlap cards are filtered out
    assert "no_overlap_1" not in result_ids, f"no_overlap_1 should be dropped, got {result_ids}"
    # 4) Cards with no core_relations are kept
    assert "generic_1" in result_ids, f"generic_1 should be kept, got {result_ids}"
    # 5) Ordering: overlap_1 (2 family overlaps) before partial_1 (1 family overlap)
    idx_overlap = result_ids.index("overlap_1")
    idx_partial = result_ids.index("partial_1")
    assert idx_overlap < idx_partial, (
        f"overlap_1 (2 overlaps) should come before partial_1 (1 overlap), got {result_ids}"
    )

    print("filter_skills_post_plan: all assertions passed.")
