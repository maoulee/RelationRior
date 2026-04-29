#!/usr/bin/env python3
"""Quick integration test: chain decomposition → GTE retrieval → graph traversal."""
from __future__ import annotations
import asyncio, json, pickle, re, time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import aiohttp

try:
    import graph_tool as gt
    _HAS_GT = True
except ImportError:
    _HAS_GT = False


ROOT = Path(__file__).resolve().parents[1]
LLM_API_URL = "http://localhost:8000/v1/chat/completions"
LLM_MODEL = "qwen35-9b-local"
GTE_API_URL = "http://localhost:8003"
GTE_TASK_DESC = "Given a knowledge graph question, retrieve relevant graph relations that answer the question"
REASON_STYLE = "default"  # "default" or "check"
DEFAULT_PILOT = ROOT / "reports/stage_pipeline_test/find_check_plan_pilot_10cases/results.json"
DEFAULT_CWQ = ROOT / "data/cwq_processed/test_literal_and_language_fixed.pkl"
MASK_WRONG_TYPE = ROOT / "data/cwq_processed/mask_wrong_type_ids.json"
DEFAULT_OUTPUT = ROOT / "reports/stage_pipeline_test/chain_decompose_test"

DECOMP_PROMPT = '''Decompose the question into natural language sub-questions.

Break the original question into an ordered sequence of sub-questions. Each sub-question must be a complete, natural sentence that a human would ask.

For each sub-question, label its type:
- find: retrieves new information by following a relation in the knowledge graph.
- verify: checks a filter constraint on results already found (temporal: "before 1998", "most recent"; superlative: "largest", "biggest"; geographic: "bordering a specified place"; numeric: equals a value; intersection: must satisfy both A and B).

Rules:
1. START from the most specific named entity in the question — the one with the fewest possible neighbors. Prefer unique names (people, events, titles) over countries, regions, or groups.
2. Use 1 to 4 ordered steps. At most ONE verify step, placed at the end.
3. If the question is a straightforward chain of lookups with no filter constraint, all steps are find.
4. Each sub-question must be a complete natural language sentence, not a keyword phrase.
5. For each step, also provide a compact relation_query using domain nouns and verbs for retrieval.
6. Endpoint rule: only the LAST step may carry an endpoint with a fixed entity explicitly from the question. Otherwise use none.
7. Never output placeholder endpoints such as "[Country Name]", "team name", or bracketed templates.
8. Do not enumerate or name entities not present in the question.
9. Do not output chain-of-thought, hidden reasoning, explanations, examples, or alternative plans.

Output format:
Anchor: [entity name] (entity_query: [search term for entity retrieval])
Answer_type: [free-form noun phrase describing what the answer IS, e.g. person, country, government_type, language, sport, event, year, monetary_value, percentage, etc.]
1. "Who was the Governor of Arizona in 2009?" (type: find; relation_query: governor of state; endpoint: none)
2. "Did that governor hold a governmental position before 1998?" (type: verify; relation_query: tenure start date; endpoint: none)

Return only the decomposition in the exact format above.
'''


def is_cvt_like(name: str) -> bool:
    """Detect CVT (Compound Value Type) nodes.
    Matches m.xxx / g.xxx pattern OR entities in non_text with no readable name."""
    if not name or len(name) < 2:
        return False
    if re.match(r"^[mg]\.[A-Za-z0-9_]+$", name):
        return True
    return False


def expand_cvt_leaves(ents, rels, h_ids, r_ids, t_ids):
    """Auto-expand CVT leaf nodes (degree ≤ 1) by finding additional edges
    from other triples in the subgraph that share the same CVT node.
    Returns potentially augmented (ents, rels, h_ids, r_ids, t_ids)."""
    from collections import Counter
    node_degree = Counter()
    for i in range(len(h_ids)):
        node_degree[h_ids[i]] += 1
        node_degree[t_ids[i]] += 1

    # Find CVT nodes with degree ≤ 1 (leaf/dead-end)
    cvt_leaves = []
    for idx, name in enumerate(ents):
        if is_cvt_like(name) and node_degree.get(idx, 0) <= 1:
            cvt_leaves.append(idx)

    if not cvt_leaves:
        return ents, rels, h_ids, r_ids, t_ids

    # For each CVT leaf, search the subgraph for any triples where it appears
    # that weren't included (e.g., via shared intermediate nodes)
    # Since we only have the subgraph, we can only find edges already present
    # but potentially missed due to indexing
    # No-op for now: full expansion requires KG API access
    # Flag count for debugging
    return ents, rels, h_ids, r_ids, t_ids


def rel_to_text(rel: str) -> str:
    """Return original dot-notation format for LLM display: 'people.person.religion'"""
    return rel


def rel_to_text_short(rel: str) -> str:
    """Short format (last 2 segments) for GTE retrieval: 'person religion'"""
    parts = rel.split(".")
    return " ".join(p.replace("_", " ") for p in parts[-2:]) if len(parts) >= 2 else rel.replace("_", " ")


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9%.' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def get_entity_contexts(entity_names, h_ids, r_ids, t_ids, ents, rels):
    """For each candidate entity, extract surrounding relation context from subgraph.
    Similar to check_entities' [Context: ...] annotation."""
    # Build name→idx lookup
    name_to_indices = {}
    for i, e in enumerate(ents):
        name_to_indices.setdefault(e, []).append(i)

    contexts = {}
    for cand in entity_names:
        indices = name_to_indices.get(cand, [])
        if not indices:
            continue
        idx_set = set(indices)
        outgoing = []
        incoming = []
        for i in range(len(h_ids)):
            if h_ids[i] in idx_set:
                rel_text = rel_to_text(rels[r_ids[i]]) if 0 <= r_ids[i] < len(rels) else "?"
                t_name = ents[t_ids[i]] if 0 <= t_ids[i] < len(ents) else "?"
                if not is_cvt_like(t_name) and t_name != cand:
                    outgoing.append(f"{rel_text}→{t_name}")
            if t_ids[i] in idx_set:
                rel_text = rel_to_text(rels[r_ids[i]]) if 0 <= r_ids[i] < len(rels) else "?"
                h_name = ents[h_ids[i]] if 0 <= h_ids[i] < len(ents) else "?"
                if not is_cvt_like(h_name) and h_name != cand:
                    incoming.append(f"{h_name}→{rel_text}")
        parts = outgoing[:2] + incoming[:1]
        if parts:
            contexts[cand] = "; ".join(parts)
    return contexts


async def llm_resolve_entity(session, question, query, candidates_with_ctx):
    """LLM selects the correct entity from GTE top-k candidates using surrounding relation context.
    candidates_with_ctx: list of (name, context_str) tuples.
    Returns selected entity name or None."""
    if not candidates_with_ctx:
        return None
    if len(candidates_with_ctx) == 1:
        return candidates_with_ctx[0][0]

    cand_lines = []
    for i, (name, ctx) in enumerate(candidates_with_ctx, 1):
        cand_lines.append(f"  {i}. {name} [{ctx}]" if ctx else f"  {i}. {name}")

    prompt = f"""Search query: {query}

Candidate entities (with relation context from knowledge graph):
{chr(10).join(cand_lines)}

Which candidate best matches the search query? Use the relation context to identify what each entity actually IS (a person, a location, a schema type, etc). Pick the specific entity, not generic types or schema entries.

<analysis>Brief reasoning about which candidate matches the query</analysis>
<selected>entity name</selected>"""

    for _ in range(2):
        raw = await call_llm(session, [
            {"role": "system", "content": "Select the correct entity from candidates. Output <analysis> and <selected> XML tags."},
            {"role": "user", "content": prompt},
        ], max_tokens=300)
        sel = extract_xml_tag(raw, "selected")
        if sel:
            sel = sel.strip().strip('"').strip("'")
            # Match to candidate names
            for name, _ in candidates_with_ctx:
                if normalize(sel) == normalize(name):
                    return name
            # Fuzzy match
            for name, _ in candidates_with_ctx:
                if normalize(sel) in normalize(name) or normalize(name) in normalize(sel):
                    return name
    # Fallback to GTE top-1
    return candidates_with_ctx[0][0]


def candidate_hit(cands: List[str], targets: List[str]) -> bool:
    norm_cands = [normalize(c) for c in cands if c.strip()]
    if not norm_cands:
        return False
    for t in targets:
        nt = normalize(t)
        for c in norm_cands:
            if c == nt or nt in c or c in nt:
                return True
    # Fuzzy fallback: catch near-matches like "Connor" vs "Conner"
    # Only for entities ≥ 8 chars to avoid false positives on short names
    for t in targets:
        nt = normalize(t)
        if len(nt) < 8:
            continue
        for c in norm_cands:
            if len(c) < 8:
                continue
            if SequenceMatcher(None, c, nt).ratio() >= 0.92:
                return True
    return False


def strict_candidate_hit(cands: List[str], targets: List[str]) -> bool:
    """Strict matching: exact or substring with min length 4."""
    norm_cands = [normalize(c) for c in cands]
    for t in targets:
        nt = normalize(t)
        if len(nt) < 2:
            continue
        for c in norm_cands:
            if len(c) < 2:
                continue
            if c == nt:
                return True
            shorter = min(len(c), len(nt))
            if shorter >= 4 and (nt in c or c in nt):
                return True
    return False


def compute_match_stats(predicted: List[str], gold: List[str]) -> Dict[str, float]:
    """Compute P/R/F1 between predicted and gold answer sets.

    Returns: {'precision': float, 'recall': float, 'f1': float,
              'matched_gold': int, 'matched_pred': int, 'n_gold': int, 'n_pred': int}
    Uses the same matching logic as candidate_hit (normalize + substring + fuzzy).
    """
    if not predicted or not gold:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'matched_gold': 0, 'matched_pred': 0, 'n_gold': len(gold), 'n_pred': len(predicted)}

    norm_pred = [normalize(c) for c in predicted if c.strip()]
    norm_gold = [normalize(t) for t in gold if t.strip()]
    if not norm_pred or not norm_gold:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'matched_gold': 0, 'matched_pred': 0, 'n_gold': len(gold), 'n_pred': len(predicted)}

    def _matches(c: str, t: str) -> bool:
        if c == t or t in c or c in t:
            return True
        if len(c) >= 8 and len(t) >= 8 and SequenceMatcher(None, c, t).ratio() >= 0.92:
            return True
        return False

    # Recall: how many gold items matched by at least one prediction
    matched_gold = 0
    for t in norm_gold:
        if any(_matches(c, t) for c in norm_pred):
            matched_gold += 1

    # Precision: how many predictions matched at least one gold item
    matched_pred = 0
    for c in norm_pred:
        if any(_matches(c, t) for t in norm_gold):
            matched_pred += 1

    precision = matched_pred / len(norm_pred) if norm_pred else 0.0
    recall = matched_gold / len(norm_gold) if norm_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1,
            'matched_gold': matched_gold, 'matched_pred': matched_pred,
            'n_gold': len(norm_gold), 'n_pred': len(norm_pred)}


class _BatchCoalescer:
    """Coalesces concurrent call_llm requests into batch API calls.

    When multiple call_llm calls arrive within a short collection window,
    they are merged into a single /v1/chat/completions/batch request.
    This avoids GPU contention between independent LLM requests.
    """
    _BATCH_WINDOW = 0.05  # 50ms collection window
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._pending = []       # list of (messages, max_tokens, future)
        self._timer = None
        self._lock = asyncio.Lock()

    async def submit(self, session, messages, max_tokens=500, retries=3):
        """Submit a request. Returns response string."""
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        async with self._lock:
            self._pending.append((session, messages, max_tokens, retries, fut))
            if self._timer is None or self._timer.done():
                self._timer = asyncio.ensure_future(self._flush_after_window())
        return await fut

    async def _flush_after_window(self):
        await asyncio.sleep(self._BATCH_WINDOW)
        await self._flush()

    async def _flush(self):
        async with self._lock:
            batch = self._pending[:]
            self._pending.clear()

        if not batch:
            return

        # Group by (session, max_tokens) — same params can share a batch
        groups = {}
        for session, messages, max_tokens, retries, fut in batch:
            key = (id(session), max_tokens)
            groups.setdefault(key, []).append((session, messages, max_tokens, retries, fut))

        for key, items in groups.items():
            session = items[0][0]
            max_tokens = items[0][2]
            retries = items[0][3]

            if len(items) == 1:
                # Single request — use normal endpoint
                _, messages, _, _, fut = items[0]
                try:
                    result = await self._call_single(session, messages, max_tokens, retries)
                    if not fut.done():
                        fut.set_result(result)
                except Exception:
                    try:
                        result = await self._call_single(session, messages, max_tokens, retries)
                        if not fut.done():
                            fut.set_result(result)
                    except Exception:
                        if not fut.done():
                            fut.set_result("")
            else:
                # Multiple requests — use batch endpoint
                messages_list = [m for _, m, _, _, _ in items]
                futs = [f for _, _, _, _, f in items]
                try:
                    results = await self._call_batch(session, messages_list, max_tokens, retries)
                    for f, r in zip(futs, results):
                        if not f.done():
                            f.set_result(r)
                except Exception:
                    # Fallback: individual calls
                    for _, messages, _, _, fut in items:
                        if not fut.done():
                            try:
                                r = await self._call_single(session, messages, max_tokens, retries)
                                fut.set_result(r)
                            except Exception:
                                fut.set_result("")

    @staticmethod
    async def _call_single(session, messages, max_tokens, retries):
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "top_p": 0.8,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        for attempt in range(retries):
            try:
                async with session.post(LLM_API_URL, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    data = await resp.json()
                return data["choices"][0]["message"]["content"]
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt < retries - 1:
                    await asyncio.sleep(2)
                else:
                    raise

    @staticmethod
    async def _call_batch(session, messages_list, max_tokens, retries):
        payload = {
            "model": LLM_MODEL,
            "messages": messages_list,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "top_p": 0.8,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        batch_url = LLM_API_URL.replace("/chat/completions", "/chat/completions/batch")
        for attempt in range(retries):
            try:
                async with session.post(batch_url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    data = await resp.json()
                results = [None] * len(messages_list)
                for choice in data.get("choices", []):
                    idx = choice["index"]
                    if 0 <= idx < len(messages_list):
                        results[idx] = choice["message"]["content"]
                # Fill any missing results with individual calls
                for i, r in enumerate(results):
                    if r is None:
                        results[i] = await _BatchCoalescer._call_single(
                            session, messages_list[i], max_tokens, retries)
                return results
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt < retries - 1:
                    await asyncio.sleep(2)
                else:
                    raise


async def call_llm(session: aiohttp.ClientSession, messages: list, max_tokens: int = 500, retries: int = 3) -> str:
    """LLM call with automatic request coalescing for batch optimization."""
    return await _BatchCoalescer.get().submit(session, messages, max_tokens, retries)


# ══════════════════════════════════════════════════════════════════════════════
# Stage-based batch execution infrastructure
# ══════════════════════════════════════════════════════════════════════════════

_BATCH_URL = LLM_API_URL.replace("/chat/completions", "/chat/completions/batch")


_TRUNCATE_RETRY_MSG = {"role": "user", "content": "Your previous response was cut off due to length. Answer concisely in under 200 words — avoid repetition and explanation."}

async def batch_call_llm(session, prompts: List[List[Dict[str, str]]],
                         max_tokens: int = 500) -> List[Optional[str]]:
    """Send N conversations as one batch to the vLLM batch endpoint.
    Returns list of response strings aligned with input prompts.
    Auto-retries truncated responses (finish_reason=length) with a concise instruction."""
    if not prompts:
        return []
    if len(prompts) == 1:
        try:
            return [await _call_single_direct(session, prompts[0], max_tokens)]
        except Exception:
            return [None]

    payload = {
        "model": LLM_MODEL,
        "messages": prompts,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "top_p": 0.8,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for attempt in range(3):
        try:
            async with session.post(_BATCH_URL, json=payload,
                                    timeout=aiohttp.ClientTimeout(total=180)) as resp:
                data = await resp.json()
            results: List[Optional[str]] = [None] * len(prompts)
            truncated: List[int] = []
            for choice in data.get("choices", []):
                idx = choice["index"]
                if 0 <= idx < len(prompts):
                    results[idx] = choice["message"]["content"]
                    if choice.get("finish_reason") == "length":
                        truncated.append(idx)
            # Retry truncated responses with concise instruction
            for idx in truncated:
                try:
                    retry_msgs = prompts[idx] + [_TRUNCATE_RETRY_MSG]
                    results[idx] = await _call_single_direct(session, retry_msgs, max_tokens)
                except Exception:
                    pass
            # Fill None with individual fallback
            for i, r in enumerate(results):
                if r is None:
                    try:
                        results[i] = await _call_single_direct(session, prompts[i], max_tokens)
                    except Exception:
                        results[i] = ""
            return results
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt < 2:
                await asyncio.sleep(2)
            else:
                return [""] * len(prompts)
    return [""] * len(prompts)


async def _call_single_direct(session, messages, max_tokens=500):
    """Direct single LLM call (no coalescing). Used by batch_call_llm for single-item."""
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "top_p": 0.8,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with session.post(LLM_API_URL, json=payload,
                            timeout=aiohttp.ClientTimeout(total=60)) as resp:
        data = await resp.json()
    return data["choices"][0]["message"]["content"]


@dataclass
class CaseState:
    """Holds all intermediate state for one case between pipeline stages."""
    # Identity
    case_id: str
    case_num: int
    sample: Dict[str, Any]
    pilot_row: Dict[str, Any]

    # Input data (set at initialization)
    question: str = ""
    gt_answers: List[str] = field(default_factory=list)
    ents: List[str] = field(default_factory=list)
    rels: List[str] = field(default_factory=list)
    h_ids: List[int] = field(default_factory=list)
    r_ids: List[int] = field(default_factory=list)
    t_ids: List[int] = field(default_factory=list)
    rel_texts: List[str] = field(default_factory=list)
    ent_candidates: List[str] = field(default_factory=list)

    # Stage 0: NER
    ner_scored: List[Dict] = field(default_factory=list)
    ner_top_ents: List[Tuple[str, float]] = field(default_factory=list)
    ner_name_to_ids_expanded: Dict[str, List[int]] = field(default_factory=dict)

    # Stage 1: Decomposition
    decomp_raw: Optional[str] = None
    decomp_question: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    answer_type: Optional[str] = None
    use_ner: bool = True
    anchor_forbidden: Optional[str] = None

    # Stage 1.5: Decomposition reflection
    decomp_retry: bool = False
    decomp_reflect_raw: Optional[str] = None
    decomp_retry_reason: Optional[str] = None

    # Stage 2: Entity resolution
    anchor_idx: Optional[int] = None
    anchor_name: Optional[str] = None
    breakpoints: Dict[int, int] = field(default_factory=dict)
    entity_retrieval_details: List[Dict] = field(default_factory=list)

    # Stage 3: GTE relation retrieval
    step_candidates: Dict[int, List[Tuple[int, str, float]]] = field(default_factory=dict)
    gte_per_step: Dict[int, Dict[int, Tuple[str, float]]] = field(default_factory=dict)
    relation_retrieval_details: List[Dict] = field(default_factory=list)

    # Stage 4: Relation pruning
    step_relations: List[set] = field(default_factory=list)
    prune_debug: Dict = field(default_factory=dict)

    # Stage 5: Graph traversal
    paths: List[Dict] = field(default_factory=list)
    max_depth: int = 0
    max_cov: int = 0
    answer_candidates: List[str] = field(default_factory=list)
    gt_hit: bool = False
    gt_hit_strict: bool = False
    gt_f1: float = 0.0
    all_subgraph_nodes: set = field(default_factory=set)

    # Stage 5/6: Frontier traversal + safety net
    layer_diagnostics: List[Dict] = field(default_factory=list)
    planning_attempts: List[Dict] = field(default_factory=list)
    needs_direct_answer: bool = False

    # Stage 7: Path selection
    logical_paths: List[Dict] = field(default_factory=list)
    selected_paths: List[int] = field(default_factory=list)
    attempt_log: List[Dict] = field(default_factory=list)

    # Stage 8: Answer reasoning
    llm_answer: Optional[str] = None
    llm_hit: bool = False
    llm_f1: float = 0.0
    llm_precision: float = 0.0
    llm_recall: float = 0.0
    llm_reasoning_prompt: Optional[str] = None
    llm_reasoning_full: Optional[str] = None
    num_triples: int = 0

    # Control
    active: bool = True
    error: Optional[str] = None
    stage_times: Dict[str, float] = field(default_factory=dict)

    # Stage 1a: Entity analysis output
    _1a_prompt: Optional[str] = None  # actual input prompt sent for 1a
    _1a_raw: Optional[str] = None
    _1a_anchor: Optional[str] = None
    _1a_endpoints: Optional[str] = None
    _1a_answer_type: Optional[str] = None
    _1a_rewritten: Optional[str] = None
    _1a_interpretation: Optional[str] = None
    decomp_prompt_formatted: Optional[str] = None  # actual formatted prompt sent to LLM

    # Internal temp fields for cross-stage data
    _pending_endpoints: List[Dict] = field(default_factory=list)
    _pending_anchor_eq: Optional[str] = None
    _pending_anchor_eq_name: Optional[str] = None
    _prev_attempt_score: tuple = (-1, -1, -1)


async def gte_retrieve(session, query, candidates, candidate_texts=None, top_k=10):
    payload = {"query": query, "candidates": candidates, "candidate_texts": candidate_texts, "top_k": top_k}
    async with session.post(f"{GTE_API_URL}/retrieve", json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
        data = await resp.json()
    return data.get("results", [])


def score_causal_tier(hit_set, bridge_length=0):
    """Score path by decomposition layer coverage using lexicographic tuple.

    Returns (hit_count, max_layer_hit, -bridge_length) for comparison.
    Higher = better. Layer dedup: each layer counted once regardless of
    how many of its relations match.

    3-step: {R1,R2,R3} > {R1,R3}={R2,R3} > {R1,R2} > {R3} > {R2} > {R1}
      (3,2,*)       (2,2,*)  (2,2,*)    (2,1,*)   (1,2,*)  (1,1,*)  (1,0,*)
    """
    if not hit_set:
        return (0, -1, 0)
    return (len(hit_set), max(hit_set), -bridge_length)


def extract_xml_tag(text, tag):
    """Extract content between <tag>...</tag>, returns None if not found."""
    if not text:
        return None
    match = re.search(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    return match.group(1).strip() if match else None


async def llm_prune_all_relations(session, question, all_steps, step_candidates):
    """Single LLM call to prune relations for ALL steps at once.

    Args:
        all_steps: list of parsed step dicts
        step_candidates: dict mapping step_num -> list of (idx, rel_name, score)

    Returns:
        dict mapping step_num -> set of selected relation indices,
        and debug dict with full prompt/response
    """
    # Build the prompt with all steps' candidates
    chain_lines = []
    step_blocks = []
    for s in all_steps:
        sn = s["step"]
        ep_str = f" -> endpoint: {s['endpoint']}" if s.get("endpoint") else ""
        chain_lines.append(f"  Step {sn}: {s['question']}{ep_str}")

        cands = step_candidates.get(sn, [])
        if not cands:
            step_blocks.append(f"Step {sn}: {s['question']}\n  Purpose: {s.get('definition', '')}\n  Candidates: (none)")
            continue

        cand_lines = [f"    {i}. {name}"
                      for i, (idx, name, score) in enumerate(cands, 1)]
        step_blocks.append(
            f"Step {sn}: {s['question']}\n"
            f"  Purpose: {s.get('definition', '')}\n"
            f"  Candidates:\n" + "\n".join(cand_lines)
        )

    chain_text = "\n".join(chain_lines)
    blocks_text = "\n\n".join(step_blocks)

    prompt = f"""Analyze and select knowledge graph relations for each step of this reasoning chain.

Question: {question}

Reasoning chain:
{chain_text}

Step-by-step candidates:
{blocks_text}

Rules:
1. Each step connects FROM previous output TO next — select bridge relations
2. Select 2-4 relevant relations per step. Pick the best candidates that match the step's purpose.
3. When uncertain about a relation's relevance, INCLUDE it — missing a key relation is far worse
   than having an extra irrelevant one.
4. Ignore unrelated attributes (currency, codes when asking about geography)
5. If no relations fit a step, output empty list
6. ORDER matters: rank by relevance to the step (most relevant first)

Output format:
<analysis>
One sentence per step: what it needs and which relations fit.
</analysis>
<selected>
step_1: [1, 3]
step_2: [2, 5]
</selected>"""

    system = "You are a knowledge graph relation selector for multi-step QA. Analyze the full chain, then select relevant relations per step. Output <analysis> and <selected> XML tags."

    # Try up to 3 times (1 initial + 2 retries)
    for attempt in range(3):
        raw = await call_llm(session, [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ], max_tokens=2000)

        selected_yaml = extract_xml_tag(raw, "selected")
        if selected_yaml:
            break
    else:
        # All attempts failed to produce <selected> tag
        selected_yaml = None

    # Parse YAML-style step_N: [1, 3, 5]
    result = {}
    if selected_yaml:
        for line in selected_yaml.split('\n'):
            line = line.strip()
            m = re.match(r'step_(\d+)\s*:\s*\[(.*?)\]', line)
            if m:
                sn = int(m.group(1))
                nums = [int(x.strip()) for x in m.group(2).split(',') if x.strip().isdigit()]
                cands = step_candidates.get(sn, [])
                selected_indices = set()
                for n in nums:
                    if 1 <= n <= len(cands):
                        selected_indices.add(cands[n - 1][0])
                result[sn] = selected_indices

    # Fallback: for any step missing from result, use top-3 GTE
    for s in all_steps:
        sn = s["step"]
        if sn not in result or not result[sn]:
            cands = step_candidates.get(sn, [])
            result[sn] = set(idx for idx, _, _ in cands[:3])

    debug = {
        "prompt": prompt,
        "response": raw,
        "parsed_yaml": selected_yaml,
    }
    return result, debug


async def llm_reselect_single_step_relation(session, question, step, step_candidates, current_indices):
    """Reselect relations for one failed step, avoiding the current failed choice set."""
    cands = step_candidates.get(step["step"], [])
    if not cands:
        return set()

    cand_lines = []
    current_pos = set()
    for i, (idx, name, score) in enumerate(cands, 1):
        marker = " [CURRENT]" if idx in current_indices else ""
        if idx in current_indices:
            current_pos.add(i)
        cand_lines.append(f"  {i}. {name}{marker}")

    prompt = f"""The current relation choice for one reasoning step appears to be wrong or too noisy.

Question: {question}
Failed step: {step['question']}
Step purpose: {step.get('definition', '')}

Candidate relations:
{chr(10).join(cand_lines)}

Select 1 to 3 BETTER alternative relations for this step.

Rules:
- Prefer relations that directly express the step semantics.
- Avoid the currently marked failed choices if better alternatives exist.
- Do not select generic or weakly related relations just because they are broad.

Output format:
<analysis>One short sentence.</analysis>
<selected>comma-separated candidate numbers only</selected>"""

    raw = await call_llm(session, [
        {"role": "system", "content": "You reselect better knowledge graph relations for a single failed reasoning step. Output <analysis> and <selected>."},
        {"role": "user", "content": prompt},
    ], max_tokens=400)

    selected = set()
    sel_text = extract_xml_tag(raw, "selected") or ""
    for m in re.finditer(r"\d+", sel_text):
        n = int(m.group())
        if 1 <= n <= len(cands):
            idx = cands[n - 1][0]
            selected.add(idx)

    if not selected:
        for idx, _, _ in cands:
            if idx not in current_indices:
                selected.add(idx)
            if len(selected) >= 3:
                break
    if not selected:
        selected = set(idx for idx, _, _ in cands[:3])
    return selected


def parse_decomposition(raw: str) -> Tuple[Optional[str], Optional[str], List[Dict[str, Any]], Optional[str]]:
    """Parse LLM decomposition: extract anchor name, entity_query, structured steps, and answer_type.

    Supports both new format (with type: find/verify, quoted questions) and old format.
    New:  1. "Who was the Governor?" (type: find; relation_query: ...; endpoint: ...)
    Old:  1. Who was the Governor? (relation_query: ...; endpoint: ...)
    """
    anchor_name = None
    anchor_entity_query = None
    steps = []
    answer_type = None
    for line in raw.strip().split("\n"):
        line = line.strip()
        # Extract anchor line: Anchor: name (entity_query: ...)
        m_anchor = re.match(r"^Anchor:\s*(.+?)\s*\(entity_query:\s*(.+?)\)\s*$", line, re.IGNORECASE)
        if m_anchor:
            anchor_name = m_anchor.group(1).strip()
            anchor_entity_query = m_anchor.group(2).strip()
            continue
        # Fallback: anchor without entity_query
        m_anchor2 = re.match(r"^Anchor:\s*(.+)$", line, re.IGNORECASE)
        if m_anchor2 and not anchor_name:
            anchor_name = m_anchor2.group(1).strip()
            anchor_entity_query = anchor_name
            continue
        # Extract answer type: Answer_type: [type]
        m_at = re.match(r"^Answer_type:\s*(.+)$", line, re.IGNORECASE)
        if m_at and not answer_type:
            answer_type = m_at.group(1).strip()
            continue

        # New format: 1. "question" (type: find/verify; relation_query: ...; endpoint: ...)
        m_new = re.match(
            r'^(\d+)\.\s*["""\u201c](.+?)["""\u201d]\s*\(type:\s*(find|verify);\s*relation_query:\s*(.+?);\s*endpoint:\s*(.+?)\)\s*$',
            line)
        if m_new:
            endpoint_raw = m_new.group(5).strip()
            endpoint, endpoint_query = _parse_endpoint(endpoint_raw)
            steps.append({
                "step": int(m_new.group(1)),
                "question": m_new.group(2).strip(),
                "type": m_new.group(3).strip(),
                "relation_query": m_new.group(4).strip(),
                "endpoint": endpoint,
                "endpoint_query": endpoint_query,
            })
            continue

        # Old format: 1. question (relation_query: ...; endpoint: ... or none)
        m = re.match(r"^(\d+)\.\s+(.+?)\s*\(relation_query:\s*(.+?);\s*endpoint:\s*(.+?)\)\s*$", line)
        if not m:
            continue
        endpoint_raw = m.group(4).strip()
        endpoint, endpoint_query = _parse_endpoint(endpoint_raw)
        steps.append({
            "step": int(m.group(1)),
            "question": m.group(2).strip(),
            "type": "find",
            "relation_query": m.group(3).strip(),
            "endpoint": endpoint,
            "endpoint_query": endpoint_query,
        })
    return anchor_name, anchor_entity_query, steps, answer_type


def _parse_endpoint(endpoint_raw: str):
    """Parse endpoint string: 'entity (entity_query: term)' or 'none'."""
    if endpoint_raw.lower() == "none":
        return None, None
    m_ep = re.match(r"(.+?)\s*\(entity_query:\s*(.+?)\)", endpoint_raw)
    if m_ep:
        return m_ep.group(1).strip(), m_ep.group(2).strip()
    return endpoint_raw, endpoint_raw


def parse_chain(text):
    """Parse CHAIN_PROMPT output into structured dict with anchor, hops, endpoints.

    Returns: {'anchor': str, 'hops': [{'relation': str, 'keyword': str, 'definition': str, 'endpoint': str}],
              'endpoint_entities': [{'entity': str, 'hop': int}], 'reasoning': str, 'raw': str}
    """
    text = text.strip()
    text = re.sub(r'^(Answer|答案)[：:]\s*', '', text)

    reasoning = ''
    m = re.search(r'Reasoning:\s*(.+?)(?=\n\s*Chain:|\nChain:)', text, re.DOTALL)
    if m:
        reasoning = m.group(1).strip()

    # Extract answer_type (XML tag or legacy format)
    answer_type = None
    at_xml = re.search(r'<answer_type>(.*?)</answer_type>', text, re.DOTALL)
    if at_xml:
        answer_type = at_xml.group(1).strip()
    else:
        at_match = re.search(r'Answer_type:\s*(.+)', text)
        if at_match:
            answer_type = at_match.group(1).strip()

    cm = re.search(r'Chain:\s*\n?(.*?)(?=\n\s*(Analysis:|Endpoints:)|\nAnalysis:|\nEndpoints:|$)', text, re.DOTALL)
    if not cm:
        return None
    chain_lines = [l.strip() for l in cm.group(1).split('\n') if '-(' in l]
    if not chain_lines:
        return None
    chain_text = chain_lines[-1]

    am = re.match(r'^(.+?)\s*-\(', chain_text)
    if not am:
        return None
    anchor = am.group(1).strip()
    rest = chain_text[am.end() - 1:]

    hops = []
    for m in re.finditer(r'\(([^)]+)\)\s*(?:\.inv\s*)?->\s*(.+?)(?=\s*-\(|$)', rest):
        node = m.group(2).strip()
        if not node or node.lower() in ('unknown', 'the', 'a'):
            node = 'node'
        hops.append({'relation': m.group(1).strip(), 'endpoint': node, 'keyword': '', 'definition': '', 'subquestion': ''})

    if not hops:
        return None

    endpoint_entities = []
    ep_match = re.search(r'Endpoints:\s*(.+)', text)
    if ep_match:
        ep_text = ep_match.group(1).strip()
        if ep_text.lower() != 'none':
            for em in re.finditer(r'\[([^\]]+)\]\s*(?:at\s*hop\s*(\d+))?', ep_text):
                ep_name = em.group(1).strip()
                hop_num = int(em.group(2)) if em.group(2) else None
                endpoint_entities.append({'entity': ep_name, 'hop': hop_num})

    anm = re.search(r'Analysis:\s*\n(.+)', text, re.DOTALL)
    if anm:
        kw = re.findall(r'-\s*Keyword:\s*(.+)', anm.group(1))
        df = re.findall(r'-\s*Definition:\s*(.+)', anm.group(1))
        sq = re.findall(r'-\s*Sub-question:\s*(.+)', anm.group(1))
        for i, hop in enumerate(hops):
            if i < len(kw): hop['keyword'] = kw[i].strip()
            if i < len(df): hop['definition'] = df[i].strip()
            if i < len(sq): hop['subquestion'] = sq[i].strip()

    return {'anchor': anchor, 'hops': hops, 'reasoning': reasoning,
            'endpoint_entities': endpoint_entities, 'answer_type': answer_type, 'raw': text}


def expand_node(node_idx, rel_indices, h_ids, r_ids, t_ids, reverse=False):
    children = []
    for i in range(len(h_ids)):
        if reverse:
            if t_ids[i] == node_idx and r_ids[i] in rel_indices:
                children.append((h_ids[i], r_ids[i]))
        else:
            if h_ids[i] == node_idx and r_ids[i] in rel_indices:
                children.append((t_ids[i], r_ids[i]))
    return children


def expand_through_cvt(node_idx, h_ids, r_ids, t_ids, entity_list):
    name = entity_list[node_idx] if 0 <= node_idx < len(entity_list) else ""
    if not is_cvt_like(name):
        return []
    children, seen = [], set()
    for i in range(len(h_ids)):
        if h_ids[i] == node_idx and t_ids[i] not in seen:
            children.append((t_ids[i], r_ids[i])); seen.add(t_ids[i])
        if t_ids[i] == node_idx and h_ids[i] not in seen:
            children.append((h_ids[i], r_ids[i])); seen.add(h_ids[i])
    return children


def _path_hits_breakpoints(path, breakpoint_nodes, h_ids, r_ids, t_ids, entity_list):
    """Return True if a path lands on, or terminates in a CVT adjacent to, any breakpoint."""
    if not path or not breakpoint_nodes:
        return False
    nodes = path.get("nodes", [])
    if not nodes:
        return False

    last = nodes[-1]
    if last in breakpoint_nodes:
        return True

    last_name = entity_list[last] if 0 <= last < len(entity_list) else ""
    if is_cvt_like(last_name):
        for nb_idx, _ in expand_through_cvt(last, h_ids, r_ids, t_ids, entity_list):
            if nb_idx in breakpoint_nodes:
                return True
    return False


def prefer_breakpoint_hit_paths(paths, breakpoints, h_ids, r_ids, t_ids, entity_list):
    """Prefer paths that hit explicit endpoint breakpoints, but never drop recall if none do."""
    if not paths or not breakpoints:
        return paths

    breakpoint_nodes = {bp for bp in breakpoints.values() if bp is not None}
    if not breakpoint_nodes:
        return paths

    hit_paths = [
        p for p in paths
        if _path_hits_breakpoints(p, breakpoint_nodes, h_ids, r_ids, t_ids, entity_list)
    ]
    return hit_paths if hit_paths else paths


def chain_expand(anchor_idx, step_relations, h_ids, r_ids, t_ids, entity_list):
    """Ordered chain expansion with forward validation: step 1 → step 2 → ...
    After expanding step K, validate that resulting nodes have step K+1 edges.
    Prunes structurally incomplete paths — if a tail node can't continue the
    chain, the triple is removed. Last step has no constraint (all results kept).
    Returns ALL paths, not just deepest.
    """
    paths = [{"nodes": [anchor_idx], "relations": [], "depth": 0}]

    for step_idx, rel_indices in enumerate(step_relations):
        if not rel_indices:
            continue
        # Look ahead: find next non-empty step for forward validation
        next_rel_indices = set()
        for nxt in range(step_idx + 1, len(step_relations)):
            if step_relations[nxt]:
                next_rel_indices = step_relations[nxt]
                break

        new_paths = []
        for path in paths:
            current = path["nodes"][-1]
            fwd = expand_node(current, rel_indices, h_ids, r_ids, t_ids)
            rev = expand_node(current, rel_indices, h_ids, r_ids, t_ids, reverse=True)
            all_children = fwd + rev
            if all_children:
                seen = set(path["nodes"])
                for child_idx, rel_idx in all_children:
                    if child_idx in seen:
                        continue
                    # Forward validation: skip child if it can't continue the chain
                    if next_rel_indices and not _has_step_edges(
                            child_idx, next_rel_indices, h_ids, r_ids, t_ids, entity_list):
                        continue
                    new_path = {"nodes": path["nodes"] + [child_idx], "relations": path["relations"] + [rel_idx], "depth": path["depth"] + 1}
                    child_name = entity_list[child_idx] if 0 <= child_idx < len(entity_list) else ""
                    if is_cvt_like(child_name):
                        cvt_children = expand_through_cvt(child_idx, h_ids, r_ids, t_ids, entity_list)
                        new_seen = set(new_path["nodes"])
                        for cvt_idx, cvt_rel in cvt_children:
                            if cvt_idx in new_seen:
                                continue
                            new_paths.append({"nodes": new_path["nodes"] + [cvt_idx], "relations": new_path["relations"] + [cvt_rel], "depth": new_path["depth"] + 1})
                    else:
                        new_paths.append(new_path)
            else:
                # Dead end: keep path as-is for this step
                new_paths.append(path)
        paths = new_paths if new_paths else paths

    if not paths:
        return [], 0
    max_depth = max(p["depth"] for p in paths)
    return paths, max_depth


def _has_step_edges(node_idx, rel_indices, h_ids, r_ids, t_ids, entity_list):
    """Check if node has edges matching rel_indices within 2 hops.
    Direct edges first, then through CVT intermediary:
      node → [any rel] → CVT → [target rel]
    This handles cases like child_labor_percent where the relation
    bridges through a CVT node rather than connecting directly.
    """
    # Direct check
    fwd = expand_node(node_idx, rel_indices, h_ids, r_ids, t_ids)
    rev = expand_node(node_idx, rel_indices, h_ids, r_ids, t_ids, reverse=True)
    if fwd or rev:
        return True
    # 2-hop: node → CVT → [target rel]
    seen = {node_idx}
    for i in range(len(h_ids)):
        if h_ids[i] == node_idx:
            neighbor = t_ids[i]
        elif t_ids[i] == node_idx:
            neighbor = h_ids[i]
        else:
            continue
        if neighbor in seen:
            continue
        neighbor_name = entity_list[neighbor] if 0 <= neighbor < len(entity_list) else ""
        if not is_cvt_like(neighbor_name):
            continue
        seen.add(neighbor)
        fwd2 = expand_node(neighbor, rel_indices, h_ids, r_ids, t_ids)
        rev2 = expand_node(neighbor, rel_indices, h_ids, r_ids, t_ids, reverse=True)
        if fwd2 or rev2:
            return True
    return False




def chain_expand_v2(anchor_idx, step_relations, h_ids, r_ids, t_ids, entity_list,
                    max_hops=4, beam_width=80, per_branch_width=5):
    """Relation-anchored multi-hop expansion: anchor on relations, not hop count.

    For step_relations = [R1_set, R2_set, ..., Rn_set]:
    1. MAIN: Search all paths within max_hops where the LAST hop's relation is in Rn_set.
    2. BACKTRACK: Check if earlier relations (R1..Rn-1) appear in order in the path.
    3. SCORE: Full match (all relations in order) > partial match > single relation.
    4. FALLBACK: If no Rn paths found, try Rn-1, then Rn-2, ..., down to R1.

    CVT nodes pass through without counting toward hop limit.

    Returns (paths, max_depth, max_coverage_tier) where paths are dicts with keys:
      nodes, relations, depth, covered_steps, matched_relations, coverage_tier
    """
    n_steps = len(step_relations)
    if n_steps == 0:
        return [], 0, 0

    # Build adjacency list (undirected)
    adj = {}
    for i in range(len(h_ids)):
        h, r, t = h_ids[i], r_ids[i], t_ids[i]
        adj.setdefault(h, []).append((t, r))
        adj.setdefault(t, []).append((h, r))

    # Build rel_to_step mapping
    rel_to_step = {}
    all_target_rels = set()
    for step_idx, rel_set in enumerate(step_relations):
        for rel in rel_set:
            rel_to_step.setdefault(rel, set()).add(step_idx)
            all_target_rels.add(rel)

    # Coverage tier computation:
    # Given a path's matched steps, compute tier score.
    # For [R1, R2]: R1+R2=3, R2_only=2, R1_only=1
    # For [R1, R2, R3]: R1+R2+R3=7, R1+R3=5, R2+R3=6, R3_only=4, R1+R2=3, R2_only=2, R1_only=1
    # General: priority by last-step presence, then by earlier steps, ordered correctly.
    def compute_tier(matched_steps_set):
        """Higher is better. Must include the target step for main strategy."""
        if not matched_steps_set:
            return 0
        # Score: each matched step contributes 2^(position). Order matters implicitly
        # because we check subset ordering separately.
        score = 0
        for s in matched_steps_set:
            score += (1 << s)
        return score

    def check_order(path_rels, target_rels_ordered):
        """Check which target relation sets appear in the path in correct order.

        Returns set of step indices that are matched in order.
        """
        if not path_rels:
            return set()

        matched = set()
        search_start = 0
        for step_idx, rel_set in enumerate(target_rels_ordered):
            if not rel_set:
                continue
            # Find first occurrence of any rel in rel_set at or after search_start
            for pos in range(search_start, len(path_rels)):
                if path_rels[pos] in rel_set:
                    matched.add(step_idx)
                    search_start = pos + 1
                    break
        return matched

    def _beam_prune(paths, limit):
        if len(paths) <= limit:
            return paths
        # Group by branch signature (last node + covered steps pattern) to maintain diversity
        groups = {}
        for p in paths:
            sig = (p["nodes"][-1], tuple(sorted(p.get("covered_steps", set()))))
            groups.setdefault(sig, []).append(p)
        result = []
        overflow = []
        for sig, group in groups.items():
            group.sort(key=lambda x: (x.get("coverage_tier", 0), -x.get("depth", 0)), reverse=True)
            take = min(len(group), per_branch_width)
            result.extend(group[:take])
            overflow.extend(group[take:])
        if len(result) < limit and overflow:
            overflow.sort(key=lambda x: (x.get("coverage_tier", 0), -x.get("depth", 0)), reverse=True)
            result.extend(overflow[:limit - len(result)])
        if len(result) > limit:
            result.sort(key=lambda x: (x.get("coverage_tier", 0), -x.get("depth", 0)), reverse=True)
            result = result[:limit]
        return result

    def _search_from_anchor(target_step_indices):
        """BFS/DFS hybrid from anchor, looking for paths ending with target step relations.

        target_step_indices: list of step indices whose relations are acceptable as the last hop.
        Returns list of path dicts.
        """
        # Collect target relations for this search
        target_rels = set()
        for si in target_step_indices:
            target_rels |= step_relations[si]

        if not target_rels:
            return []

        # BFS with beam
        # State: list of path dicts
        initial_path = {
            "nodes": [anchor_idx],
            "relations": [],
            "depth": 0,
            "real_hops": 0,  # hops excluding CVT passthrough
            "covered_steps": set(),
            "matched_relations": set(),
            "coverage_tier": 0,
            "last_hop_is_target": False,
        }
        active = [initial_path]
        completed = []  # paths that ended with a target relation

        for _ in range(max_hops * 3):  # enough iterations for CVT passthroughs
            if not active:
                break

            new_active = []
            for path in active:
                current = path["nodes"][-1]
                current_name = entity_list[current] if 0 <= current < len(entity_list) else ""
                is_at_cvt = is_cvt_like(current_name)

                all_neighbors = adj.get(current, [])
                if not all_neighbors:
                    continue

                seen = set(path["nodes"])

                # 3-tier edge filtering: pool relations first, non-pool strictly limited
                target_edges = []   # matches current search target (e.g. R_n)
                pool_edges = []     # in any step's relation set
                other_edges = []    # not in any step — noise
                for neighbor, rel in all_neighbors:
                    if neighbor in seen:
                        continue
                    if rel in target_rels:
                        target_edges.append((neighbor, rel))
                    elif rel in all_target_rels:
                        pool_edges.append((neighbor, rel))
                    else:
                        other_edges.append((neighbor, rel))

                # Priority: target > pool > limited fallback
                if target_edges or pool_edges:
                    edges = target_edges + pool_edges
                else:
                    edges = other_edges[:5]

                for neighbor, rel in edges:
                    new_nodes = path["nodes"] + [neighbor]
                    new_rels = path["relations"] + [rel]
                    new_real_hops = path["real_hops"]

                    # CVT passthrough: don't count toward hop limit
                    neighbor_name = entity_list[neighbor] if 0 <= neighbor < len(entity_list) else ""
                    if not is_cvt_like(current_name) or not is_at_cvt:
                        new_real_hops += 1

                    # Check hop limit
                    if new_real_hops > max_hops:
                        continue

                    # Check if this hop uses a target relation
                    rel_step_matches = rel_to_step.get(rel, set())
                    is_target_hop = rel in target_rels

                    new_covered = set(path["covered_steps"])
                    new_matched = set(path["matched_relations"])
                    if rel_step_matches:
                        new_covered |= rel_step_matches
                        new_matched.add(rel)

                    new_path = {
                        "nodes": new_nodes,
                        "relations": new_rels,
                        "depth": path["depth"] + 1,
                        "real_hops": new_real_hops,
                        "covered_steps": new_covered,
                        "matched_relations": new_matched,
                        "coverage_tier": 0,
                        "last_hop_is_target": is_target_hop,
                    }

                    if is_target_hop:
                        # This path ends with a target relation — compute coverage
                        ordered_match = check_order(new_rels, step_relations)
                        new_path["covered_steps"] = ordered_match
                        new_path["coverage_tier"] = compute_tier(ordered_match)
                        completed.append(new_path)
                        # Continue expanding from here too (might find longer matches)
                        if new_real_hops < max_hops:
                            new_active.append(new_path)
                    else:
                        # Not a target hop, keep searching
                        new_active.append(new_path)

            # Beam prune active paths
            active = _beam_prune(new_active, beam_width)

        # Cap completed paths: sort by coverage tier, keep top beam_width
        if len(completed) > beam_width:
            completed.sort(key=lambda p: (p.get("coverage_tier", 0), -p.get("depth", 0)), reverse=True)
            completed = completed[:beam_width]
        return completed

    # ---- Main strategy: search for paths ending with R_n ----
    result_paths = []

    # Try from last step backwards (fallback cascade)
    for target_depth in range(n_steps - 1, -1, -1):
        target_indices = [target_depth]
        found = _search_from_anchor(target_indices)

        if found:
            # Deduplicate by (nodes tuple, relations tuple)
            seen_keys = set()
            for p in result_paths:
                seen_keys.add((tuple(p["nodes"]), tuple(p["relations"])))

            for p in found:
                key = (tuple(p["nodes"]), tuple(p["relations"]))
                if key not in seen_keys:
                    seen_keys.add(key)
                    result_paths.append(p)

            # If we found paths for the deepest target step, we can still look for earlier ones
            # but only if the main target was the last step (not a fallback)
            if target_depth == n_steps - 1:
                # Also search for earlier steps as supplementary (these are lower tier)
                for supplementary_depth in range(n_steps - 2, -1, -1):
                    supp_found = _search_from_anchor([supplementary_depth])
                    for p in supp_found:
                        key = (tuple(p["nodes"]), tuple(p["relations"]))
                        if key not in seen_keys:
                            seen_keys.add(key)
                            result_paths.append(p)
                break  # Main strategy succeeded, don't fallback further
        # If target_depth < n_steps-1, this is a fallback — accept and stop

    if not result_paths:
        return [], 0, 0

    # Sort by coverage tier (desc), then depth (desc for deeper=more info), then fewer nodes
    result_paths.sort(key=lambda p: (p.get("coverage_tier", 0), -p.get("depth", 0)), reverse=True)

    # Final beam prune on total results
    if len(result_paths) > beam_width * 2:
        result_paths = _beam_prune(result_paths, beam_width * 2)

    max_depth = max(p.get("depth", 0) for p in result_paths)
    max_tier = max(p.get("coverage_tier", 0) for p in result_paths)

    return result_paths, max_depth, max_tier


def _coverage_rank(path):
    """Causal tier ranking: layer coverage count > max layer hit > shorter bridge."""
    covered = path.get("covered_steps", frozenset())
    depth = path.get("depth", 0)
    return score_causal_tier(covered, bridge_length=depth)


def _merge_paths(fwd_path, bwd_path, entity_list):
    fwd_nodes = fwd_path["nodes"]
    bwd_nodes = bwd_path["nodes"]
    meeting_node = fwd_nodes[-1]
    if meeting_node != bwd_nodes[-1]:
        return None
    if set(fwd_nodes[:-1]) & set(bwd_nodes[:-1]):
        return None
    return {
        "nodes": fwd_nodes + list(reversed(bwd_nodes[:-1])),
        "relations": fwd_path["relations"] + list(reversed(bwd_path["relations"])),
        "depth": fwd_path["depth"] + bwd_path["depth"],
        "covered_steps": frozenset(set(fwd_path.get("covered_steps", frozenset())) | set(bwd_path.get("covered_steps", frozenset()))),
        "matched_relations": frozenset(set(fwd_path.get("matched_relations", frozenset())) | set(bwd_path.get("matched_relations", frozenset()))),
    }



def bidirectional_expand(anchor_idx, target_idx, step_relations, h_ids, r_ids, t_ids, entity_list,
                         max_hops=5, beam_width=50, per_branch_width=6):
    """Relation-prior-guided bidirectional BFS with step-aware scoring.

    Forward from anchor: preferentially expands via step-aligned relations (step 0 at hop 0, etc.)
    Backward from endpoint: preferentially expands via reversed step relations (step N-1 at hop 0, etc.)
    Three-tier edge priority: guided (step-aligned) > in-pool (other steps) > fallback (all edges).
    Post-hoc ranking by step coverage count.
    """
    n_steps = len(step_relations)
    # Build rel_to_steps mapping and pooled relation set
    rel_to_steps: Dict[int, set] = {}
    relation_pool: set = set()
    for step_idx, rel_set in enumerate(step_relations):
        for rel in rel_set:
            rel_to_steps.setdefault(rel, set()).add(step_idx)
            relation_pool.add(rel)

    # Build undirected adjacency
    adj: Dict[int, List[tuple]] = {}
    for i in range(len(h_ids)):
        h, r, t = h_ids[i], r_ids[i], t_ids[i]
        adj.setdefault(h, []).append((t, r))
        adj.setdefault(t, []).append((h, r))

    def _make_path(nodes, relations, depth, covered=frozenset(), matched=frozenset()):
        return {"nodes": nodes, "relations": relations, "depth": depth,
                "covered_steps": covered, "matched_relations": matched}

    def _extend_path(path, neighbor, rel):
        r_steps = rel_to_steps.get(rel, set())
        new_covered = frozenset(set(path["covered_steps"]) | r_steps)
        new_matched = frozenset(set(path["matched_relations"]) | ({rel} if r_steps else set()))
        return _make_path(
            path["nodes"] + [neighbor], path["relations"] + [rel], path["depth"] + 1,
            new_covered, new_matched)

    def _branch_sig(path):
        """Branch signature: (step_idx, first_matched_rel) per covered step + last node."""
        covered = path.get("covered_steps", frozenset())
        sig = []
        for step_idx in sorted(covered):
            for r in sorted(path.get("matched_relations", frozenset())):
                if step_idx in rel_to_steps.get(r, set()):
                    sig.append((step_idx, r))
                    break
        sig.append(path["nodes"][-1])
        return tuple(sig)

    def _beam_prune(paths, limit):
        if len(paths) <= limit:
            return paths
        groups: Dict[tuple, list] = {}
        for p in paths:
            sig = _branch_sig(p)
            groups.setdefault(sig, []).append(p)
        result = []
        remaining = []
        for sig, group in groups.items():
            group.sort(key=_coverage_rank, reverse=True)
            result.extend(group[:per_branch_width])
            remaining.extend(group[per_branch_width:])
        if len(result) < limit and remaining:
            remaining.sort(key=_coverage_rank, reverse=True)
            result.extend(remaining[:limit - len(result)])
        if len(result) > limit:
            result.sort(key=_coverage_rank, reverse=True)
            result = result[:limit]
        return result

    def _get_edges(node_idx, hop, is_forward):
        """Get edges for expansion with 3-tier priority: guided > in-pool > fallback."""
        all_edges = adj.get(node_idx, [])
        if not all_edges:
            return []
        # Determine expected step for this hop
        if is_forward:
            expected_step = hop % n_steps if n_steps > 0 else -1
        else:
            expected_step = (n_steps - 1 - hop % n_steps) if n_steps > 0 else -1

        expected_rels = step_relations[expected_step] if 0 <= expected_step < n_steps else set()

        guided = []    # Matches expected step relation
        in_pool = []   # In some step but not expected
        fallback = []  # Not in any step

        for neighbor, rel in all_edges:
            if rel in expected_rels:
                guided.append((neighbor, rel))
            elif rel in relation_pool:
                in_pool.append((neighbor, rel))
            else:
                fallback.append((neighbor, rel))

        # Priority: guided first, then in-pool (limited), then fallback (very limited)
        if guided:
            return guided + in_pool[:5]
        elif in_pool:
            return in_pool + fallback[:5]
        else:
            return fallback[:10]

    # Initialize frontiers
    init_fwd = _make_path([anchor_idx], [], 0)
    init_bwd = _make_path([target_idx], [], 0)
    fwd_frontier: Dict[int, List[Dict]] = {anchor_idx: [init_fwd]}
    bwd_frontier: Dict[int, List[Dict]] = {target_idx: [init_bwd]}
    fwd_path_map: Dict[int, List[Dict]] = {anchor_idx: [init_fwd]}
    bwd_path_map: Dict[int, List[Dict]] = {target_idx: [init_bwd]}
    fwd_visited: Dict[int, int] = {anchor_idx: 0}
    bwd_visited: Dict[int, int] = {target_idx: 0}
    meeting_paths = []

    for hop in range(max_hops):
        if len(fwd_frontier) <= len(bwd_frontier):
            # Forward expansion with guided edges
            new_frontier: Dict[int, List[Dict]] = {}
            for node_idx, paths in fwd_frontier.items():
                edges = _get_edges(node_idx, hop, is_forward=True)
                for neighbor, rel in edges:
                    if neighbor in fwd_visited and fwd_visited[neighbor] < hop + 1:
                        continue
                    fwd_visited.setdefault(neighbor, hop + 1)
                    for path in paths:
                        if neighbor in set(path["nodes"]):
                            continue
                        new_path = _extend_path(path, neighbor, rel)
                        if neighbor in bwd_visited:
                            for bwd_path in bwd_path_map.get(neighbor, []):
                                merged = _merge_paths(new_path, bwd_path, entity_list)
                                if merged:
                                    meeting_paths.append(merged)
                        new_frontier.setdefault(neighbor, []).append(new_path)
            for node in new_frontier:
                new_frontier[node] = _beam_prune(new_frontier[node], per_branch_width * 3)
            fwd_frontier = new_frontier
            for node, node_paths in new_frontier.items():
                fwd_path_map.setdefault(node, []).extend(node_paths)
                fwd_path_map[node] = _beam_prune(fwd_path_map[node], beam_width)
        else:
            # Backward expansion with guided edges
            new_frontier: Dict[int, List[Dict]] = {}
            for node_idx, paths in bwd_frontier.items():
                edges = _get_edges(node_idx, hop, is_forward=False)
                for neighbor, rel in edges:
                    if neighbor in bwd_visited and bwd_visited[neighbor] < hop + 1:
                        continue
                    bwd_visited.setdefault(neighbor, hop + 1)
                    for path in paths:
                        if neighbor in set(path["nodes"]):
                            continue
                        new_path = _extend_path(path, neighbor, rel)
                        if neighbor in fwd_visited:
                            for fwd_path in fwd_path_map.get(neighbor, []):
                                merged = _merge_paths(fwd_path, new_path, entity_list)
                                if merged:
                                    meeting_paths.append(merged)
                        new_frontier.setdefault(neighbor, []).append(new_path)
            for node in new_frontier:
                new_frontier[node] = _beam_prune(new_frontier[node], per_branch_width * 3)
            bwd_frontier = new_frontier
            for node, node_paths in new_frontier.items():
                bwd_path_map.setdefault(node, []).extend(node_paths)
                bwd_path_map[node] = _beam_prune(bwd_path_map[node], beam_width)

    if not meeting_paths:
        return [], 0, 0

    # Deduplicate and rank
    seen = set()
    unique = []
    for p in meeting_paths:
        key = (tuple(p["nodes"]), tuple(p["relations"]))
        if key not in seen:
            seen.add(key)
            unique.append(p)

    unique.sort(key=_coverage_rank, reverse=True)
    max_cov = len(unique[0].get("covered_steps", frozenset())) if unique else 0
    max_depth = max(p.get("depth", 0) for p in unique)

    return unique, max_depth, max_cov



def relation_prior_expand(anchor_idx, step_relations, h_ids, r_ids, t_ids, entity_list,
                          explicit_targets=None, max_hops=3, beam_width=80, per_branch_width=5):
    """Forward layer-by-layer relation-prior expansion.

    New behavior:
    1. Start from current entity frontier (initially the anchor).
    2. For layer i, search all paths within max_hops whose LAST hop relation is in R_i.
    3. Use the endpoints of those matched paths as the start frontier for the next layer.
    4. If a layer has no hit, skip it and continue from the current frontier.
    5. If explicit endpoint targets exist, connect the final frontier to those targets
       via a shortest path search within max_hops.

    This removes the backward-target template and avoids the repeated-relation
    penetration issue seen in bidirectional matching such as r1 -> r1 collapse.

    Performance optimizations (v2):
    - Paths stored as tuples (nodes, rels, depth, real_hops, covered, matched)
      instead of dicts, avoiding dict creation overhead in the hot inner loop.
    - CVT status pre-computed once as a boolean list.
    - Adjacency neighbor lists stored as tuples for faster iteration.
    - BFS in _connect_to_targets uses collections.deque.
    - Reduced frozenset churn: only create new frozensets when coverage changes.
    - _prune_paths uses frozenset directly as hash key instead of sorted tuple.
    """
    n_steps = len(step_relations)
    if n_steps == 0:
        return [], 0, 0

    # ── Build adjacency ──────────────────────────────────────
    adj: Dict[int, tuple] = {}
    for i in range(len(h_ids)):
        h, r, t = h_ids[i], r_ids[i], t_ids[i]
        if h in adj:
            adj[h] = adj[h] + ((t, r),)
        else:
            adj[h] = ((t, r),)
        if t in adj:
            adj[t] = adj[t] + ((h, r),)
        else:
            adj[t] = ((h, r),)
    adj_empty = ()

    # ── Pre-compute CVT mask (avoids re.match per hop) ──────
    is_cvt = [is_cvt_like(name) for name in entity_list]
    n_ents = len(entity_list)

    # ── Build reverse mapping ────────────────────────────────
    rel_to_step: Dict[int, set] = {}
    for si, rs in enumerate(step_relations):
        for r in rs:
            rel_to_step.setdefault(r, set()).add(si)
    all_layer_rels = set(rel_to_step.keys())

    # ── Helpers ───────────────────────────────────────────────
    def _real_hop_inc(curr_idx, next_idx):
        """All hops count equally. CVT expansion is deferred to triple generation."""
        return 1

    def _coverage_rank_fast(path):
        """Path is tuple: (nodes, rels, depth, real_hops, covered_steps, matched_rels)."""
        covered = path[4]
        depth = path[2]
        if not covered:
            return (0, -1, 0)
        return (len(covered), max(covered), -depth)

    def _prune_paths(paths, limit):
        if len(paths) <= limit:
            return paths
        # Group by (endpoint, covered_steps) — frozenset is directly hashable
        grouped: Dict[tuple, list] = {}
        for p in paths:
            sig = (p[0][-1], p[4])  # (nodes[-1], covered_steps)
            if sig in grouped:
                grouped[sig].append(p)
            else:
                grouped[sig] = [p]
        result = []
        overflow = []
        for group in grouped.values():
            group.sort(key=_coverage_rank_fast, reverse=True)
            result.extend(group[:per_branch_width])
            overflow.extend(group[per_branch_width:])
        if len(result) < limit and overflow:
            overflow.sort(key=_coverage_rank_fast, reverse=True)
            result.extend(overflow[: limit - len(result)])
        if len(result) > limit:
            result.sort(key=_coverage_rank_fast, reverse=True)
            result = result[:limit]
        return result

    def _search_terminal_relation_paths(start_paths, target_rels, layer_idx):
        """From start_paths, search local segments that terminate at the FIRST hit of target_rels.

        Rules:
        - bridge hops may not use any selected layer relation
        - once a target relation is hit, the segment ends immediately
        - same-layer relations cannot chain within one segment
        """
        if not target_rels:
            return []
        active = list(start_paths)
        matched = []
        seen_matched = set()
        layer_idx_frozen = frozenset({layer_idx})

        for _ in range(max_hops * 3):
            if not active:
                break
            new_active = []
            for path in active:
                nodes, rels, depth, real_hops, covered, matched_rels = path
                current = nodes[-1]
                neighbors = adj.get(current, adj_empty)
                for neighbor, rel in neighbors:
                    if neighbor in nodes:
                        continue
                    # ── Inverse-pair loop detection: same rel twice = trivial cycle ──
                    if rels and rels[-1] == rel:
                        continue
                    inc = _real_hop_inc(current, neighbor)
                    new_real_hops = real_hops + inc
                    if new_real_hops > max_hops:
                        continue

                    is_target_rel = rel in target_rels
                    is_any_layer_rel = rel in all_layer_rels

                    if not is_target_rel and is_any_layer_rel:
                        continue

                    new_nodes = nodes + (neighbor,)
                    new_rels = rels + (rel,)
                    new_depth = depth + 1

                    rel_steps = rel_to_step.get(rel)
                    if rel_steps:
                        new_covered = covered | rel_steps
                        new_matched = matched_rels | frozenset({rel})
                    else:
                        new_covered = covered
                        new_matched = matched_rels

                    new_path = (new_nodes, new_rels, new_depth, new_real_hops, new_covered, new_matched)

                    if is_target_rel:
                        final_covered = new_covered | layer_idx_frozen
                        final_path = (new_nodes, new_rels, new_depth, new_real_hops, final_covered, new_matched)
                        key = (new_nodes, new_rels)
                        if key not in seen_matched:
                            seen_matched.add(key)
                            matched.append(final_path)
                    else:
                        new_active.append(new_path)
            active = _prune_paths(new_active, beam_width)
        return _prune_paths(matched, beam_width)

    def _connect_to_targets(paths, targets):
        """Attach final frontier endpoints to explicit targets by shortest unconstrained path."""
        if not targets or not paths:
            return paths
        targets_set = set(targets) - {anchor_idx, None}
        if not targets_set:
            return paths

        connected = []
        seen = set()
        for base in paths:
            start = base[0][-1]  # nodes[-1]
            queue = deque([(start, (start,), (), 0)])
            local_seen = {(start, 0)}
            best = []
            while queue:
                node, nodes_seq, rel_seq, rhops = queue.popleft()
                if node in targets_set and node != start:
                    merged = (
                        base[0] + nodes_seq[1:],
                        base[1] + rel_seq,
                        base[2] + len(rel_seq),
                        base[3] + rhops,
                        base[4],
                        base[5],
                    )
                    best.append(merged)
                    continue
                for neighbor, rel in adj.get(node, adj_empty):
                    if neighbor in nodes_seq:
                        continue
                    inc = _real_hop_inc(node, neighbor)
                    new_hops = rhops + inc
                    if new_hops > max_hops:
                        continue
                    state_key = (neighbor, new_hops)
                    if state_key in local_seen:
                        continue
                    local_seen.add(state_key)
                    queue.append((neighbor, nodes_seq + (neighbor,), rel_seq + (rel,), new_hops))
            best.sort(key=_coverage_rank_fast, reverse=True)
            for p in best[:per_branch_width]:
                key = (p[0], p[1])
                if key not in seen:
                    seen.add(key)
                    connected.append(p)
        return _prune_paths(connected, beam_width) if connected else paths

    # ── Main expansion logic ─────────────────────────────────
    # Internal path format: (nodes_tuple, rels_tuple, depth, real_hops, covered_steps, matched_rels)
    frontier_paths = [((anchor_idx,), (), 0, 0, frozenset(), frozenset())]
    all_result_paths = []
    matched_layer_indices = []
    nonempty_layers = [i for i, rs in enumerate(step_relations) if rs]

    for layer_idx, target_rels in enumerate(step_relations):
        if not target_rels:
            continue
        matched = _search_terminal_relation_paths(frontier_paths, target_rels, layer_idx)
        if not matched:
            continue
        frontier_paths = matched
        all_result_paths = matched
        matched_layer_indices.append(layer_idx)

    # Minimal repair: only repair the final non-empty layer if it was missed.
    if nonempty_layers and matched_layer_indices:
        last_nonempty_idx = nonempty_layers[-1]
        if last_nonempty_idx not in matched_layer_indices and frontier_paths:
            repaired = _search_terminal_relation_paths(frontier_paths, step_relations[last_nonempty_idx], last_nonempty_idx)
            if repaired:
                merged = list(all_result_paths or []) + repaired
                merged.sort(key=_coverage_rank_fast, reverse=True)
                all_result_paths = _prune_paths(merged, beam_width)

    if explicit_targets:
        all_result_paths = _connect_to_targets(all_result_paths or frontier_paths, explicit_targets)
    elif not all_result_paths:
        all_result_paths = frontier_paths

    if not all_result_paths:
        return [], 0, 0

    # Dedup + post-hoc cycle filter
    dedup = []
    seen = set()
    for p in all_result_paths:
        key = (p[0], p[1])
        if key not in seen:
            seen.add(key)
            dedup.append(p)

    # Remove paths with cycles (repeated nodes)
    acyclic = [p for p in dedup if len(set(p[0])) == len(p[0])]
    if acyclic:
        dedup = acyclic

    dedup.sort(key=_coverage_rank_fast, reverse=True)
    dedup = _prune_paths(dedup, beam_width)

    # Convert internal tuple format back to dict format for API compatibility
    result_dicts = []
    for p in dedup:
        result_dicts.append({
            "nodes": list(p[0]),
            "relations": list(p[1]),
            "depth": p[2],
            "real_hops": p[3],
            "covered_steps": p[4],
            "matched_relations": p[5],
        })

    max_cov = max((len(p[4]) for p in dedup), default=0)
    max_depth = max((p[2] for p in dedup), default=0)
    return result_dicts, max_depth, max_cov

def diagnose_layers(anchor_idx, step_relations, h_ids, r_ids, t_ids, entity_list,
                    max_hops=3, beam_width=80, per_branch_width=5):
    """Diagnose each layer via a single-pass unified BFS from the anchor.

    One beam search traversal tracks TWO things per path simultaneously:
    - frontier_depth (int): maximum contiguous layer coverage (0,1,...,k).
      Advances when the NEXT expected layer's target relation is matched.
    - anchor_layers (frozenset of ints): ALL layers whose target relations
      were reached, regardless of order.

    After the single BFS:
    - anchor_hit[i] = any path has layer i in its anchor_layers
    - frontier_hit[i] = any path has frontier_depth >= i

    This replaces the original 2N separate BFS calls (frontier + anchor per
    layer) with one unified traversal, using CVT vertex properties for O(1)
    hop increments and tuple-based paths for speed.
    """
    if anchor_idx is None:
        return []

    if not _HAS_GT:
        raise ImportError("graph_tool is required for diagnose_layers")

    # Build undirected graph_tool graph with CVT vertex property
    graph = gt.Graph(directed=False)
    graph.add_vertex(len(entity_list))
    ep_rel = graph.new_edge_property("int")
    graph.edge_properties["relation_id"] = ep_rel

    # Mark CVT nodes at construction time (O(1) check later instead of regex)
    _cvt_re = re.compile(r"^[mg]\.[A-Za-z0-9_]+$")
    vp_cvt = graph.new_vertex_property("bool")
    for i, name in enumerate(entity_list):
        vp_cvt[i] = bool(_cvt_re.match(name)) if name else False

    # Add edges
    for i in range(len(h_ids)):
        edge = graph.add_edge(h_ids[i], t_ids[i])
        ep_rel[edge] = r_ids[i]

    # Cached neighbor lookups via graph_tool C++ backend
    _nb_cache = {}

    def _get_neighbors(node_idx):
        if node_idx in _nb_cache:
            return _nb_cache[node_idx]
        v = graph.vertex(node_idx)
        nbs = []
        for e in v.out_edges():
            s, t = int(e.source()), int(e.target())
            nb = t if s == node_idx else s
            nbs.append((nb, int(ep_rel[e])))
        result = tuple(nbs)
        _nb_cache[node_idx] = result
        return result

    # rel -> set of layer indices mapping
    rel_to_layers = {}
    for si, rs in enumerate(step_relations):
        for r in rs:
            rel_to_layers.setdefault(r, set()).add(si)

    # Precompute step_relations as list of sets for fast membership test
    step_rel_sets = [set(rs) for rs in step_relations]
    n_layers = len(step_relations)
    max_iterations = max_hops * 3 * max(n_layers, 1)

    def _hop_inc(curr, nxt):
        return 0 if (vp_cvt[curr] or vp_cvt[nxt]) else 1

    def _prune(paths, limit):
        if len(paths) <= limit:
            return paths
        grouped = {}
        for p in paths:
            sig = (p[0][-1], p[4])
            grouped.setdefault(sig, []).append(p)
        result = []
        overflow = []
        for group in grouped.values():
            group.sort(key=lambda p: p[4], reverse=True)
            result.extend(group[:per_branch_width])
            overflow.extend(group[per_branch_width:])
        if len(result) < limit and overflow:
            overflow.sort(key=lambda p: p[4], reverse=True)
            result.extend(overflow[:limit - len(result)])
        return result[:limit]

    # Path tuple: (nodes, rels, depth, hops, frontier_depth, anchor_layers)
    active = [((anchor_idx,), (), 0, 0, -1, frozenset())]

    # Collect per-layer hit info during traversal
    frontier_hits = defaultdict(list)
    anchor_hits = defaultdict(list)

    for _ in range(max_iterations):
        if not active:
            break
        new_active = []
        for nodes, rels, dep, hops, f_depth, a_layers in active:
            current = nodes[-1]
            for nb, rel in _get_neighbors(current):
                if nb in nodes:
                    continue
                new_hops = hops + _hop_inc(current, nb)
                if new_hops > max_hops:
                    continue

                # Frontier: advance contiguous coverage
                new_f = f_depth
                next_expected = f_depth + 1
                if next_expected < n_layers and rel in step_rel_sets[next_expected]:
                    new_f = next_expected

                # Anchor: record all layers matched by this relation
                matched = rel_to_layers.get(rel, set())
                new_a = a_layers | frozenset(matched)

                new_path = (nodes + (nb,), rels + (rel,), dep + 1, new_hops, new_f, new_a)
                new_active.append(new_path)

                # Record hits
                for li in matched:
                    anchor_hits[li].append(new_path)
                if new_f > f_depth:
                    for li in range(f_depth + 1, new_f + 1):
                        frontier_hits[li].append(new_path)

        active = _prune(new_active, beam_width)

    # Build diagnostics
    diagnostics = []
    for li in range(n_layers):
        diagnostics.append({
            "layer_idx": li,
            "frontier_hit": bool(frontier_hits.get(li)),
            "anchor_hit": bool(anchor_hits.get(li)),
            "frontier_count": len(frontier_hits.get(li, [])),
            "anchor_count": len(anchor_hits.get(li, [])),
        })
    return diagnostics


def compress_paths(paths, ents, rels_list, anchor_idx, breakpoint_indices):
    """Compress raw paths into logical patterns with role-aware node identification.

    Pattern format: Anchor --[rel_chain]--> <node> ... --> [Candidate] --> Endpoint
    - Anchor: starting entity name
    - <node>: abstracted bridge entities (not shown by name)
    - [Candidate]: potential answer entities
    - Endpoint: constraint entity (if exists)
    Merges paths with same relation chain pattern, collecting unique candidates.
    """
    if not paths or anchor_idx is None:
        return []

    anchor_name = ents[anchor_idx] if 0 <= anchor_idx < len(ents) else "?"
    patterns: Dict[tuple, dict] = {}

    def _has_non_cvt_loop(path):
        seen_nodes = set()
        for node_idx in path.get("nodes", []):
            name = ents[node_idx] if 0 <= node_idx < len(ents) else ""
            if is_cvt_like(name):
                continue
            if node_idx in seen_nodes:
                return True
            seen_nodes.add(node_idx)
        return False

    def _raw_path_to_readable(path):
        """Render one witness path with abstract node labels.

        Keep relation order and path shape, but hide concrete intermediate entities.
        This lets the model judge semantic fit of the path rather than overfitting
        to visible candidate names.
        """
        nodes = path["nodes"]
        rels = path["relations"]
        if not nodes:
            return anchor_name
        # If the path already reaches a non-CVT entity and then only trails into
        # CVT/value nodes, hide that trailing CVT tail from the logical path.
        # We only keep a CVT tail visible when the core relation itself lands on CVT
        # (i.e. no intermediate non-CVT endpoint has already been reached).
        cutoff_len = len(nodes)
        non_cvt_positions = []
        for i, node_idx in enumerate(nodes):
            name = ents[node_idx] if 0 <= node_idx < len(ents) else ""
            if not is_cvt_like(name):
                non_cvt_positions.append(i)
        if len(non_cvt_positions) >= 2:
            last_non_cvt_pos = non_cvt_positions[-1]
            trailing_count = len(nodes) - 1 - last_non_cvt_pos
            if trailing_count > 1:
                # Keep 1 trailing hop for relation visibility
                cutoff_len = last_non_cvt_pos + 2
        nodes = nodes[:cutoff_len]
        rels = rels[: max(0, cutoff_len - 1)]
        parts = []
        node_labels = {}
        next_label_id = 1
        for i, node_idx in enumerate(nodes):
            name = ents[node_idx] if 0 <= node_idx < len(ents) else "?"
            if i == 0:
                if node_idx == anchor_idx:
                    parts.append(anchor_name)
                else:
                    parts.append("node0")
                continue
            prev_rel_idx = rels[i - 1] if i - 1 < len(rels) else None
            rel_text = rel_to_text(rels_list[prev_rel_idx]) if prev_rel_idx is not None and prev_rel_idx < len(rels_list) else "?"
            if node_idx in node_labels:
                node_text = node_labels[node_idx]
            elif node_idx in breakpoint_indices and node_idx != anchor_idx:
                node_text = f"node{next_label_id}[endpoint]"
                node_labels[node_idx] = node_text
                next_label_id += 1
            else:
                node_text = f"node{next_label_id}"
                node_labels[node_idx] = node_text
                next_label_id += 1
            parts.append(f"--[{rel_text}]--> {node_text}")
        return " ".join(parts)

    for path in paths:
        if _has_non_cvt_loop(path):
            continue
        # Extract non-CVT nodes with positions
        sig_nodes = []
        for i, node_idx in enumerate(path["nodes"]):
            name = ents[node_idx] if 0 <= node_idx < len(ents) else ""
            if not is_cvt_like(name):
                sig_nodes.append((i, node_idx, name))

        # Build relation chain
        rel_chain = []
        if len(sig_nodes) >= 2:
            # Normal: relation chain between consecutive non-CVT nodes
            for j in range(1, len(sig_nodes)):
                prev_pos = sig_nodes[j - 1][0]
                curr_pos = sig_nodes[j][0]
                rel_indices_between = path["relations"][prev_pos:curr_pos]
                rel_texts = [rel_to_text(rels_list[ri]) for ri in rel_indices_between if ri < len(rels_list)]
                rel_chain.append(" -> ".join(rel_texts) if rel_texts else "?")
            # Append trailing relations from last non-CVT to CVT/value tail
            last_non_cvt_pos = sig_nodes[-1][0]
            trailing_rels = path["relations"][last_non_cvt_pos:]
            if trailing_rels:
                rel_texts = [rel_to_text(rels_list[ri]) for ri in trailing_rels if ri < len(rels_list)]
                rel_chain.append(" -> ".join(rel_texts) if rel_texts else "?")
        elif path["relations"]:
            # Path ends at CVT: use full relation chain as pattern
            rel_chain = [rel_to_text(rels_list[ri]) for ri in path["relations"] if ri < len(rels_list)]
        else:
            continue

        # Identify endpoint
        endpoint_name = None
        for _, node_idx, name in sig_nodes:
            if node_idx in breakpoint_indices and node_idx != anchor_idx:
                endpoint_name = name
                break

        # Identify candidates: ALL non-anchor, non-endpoint, non-CVT nodes
        # The answer is often a bridge entity being verified by the last relation,
        # not the last node before the endpoint.
        candidates = set()
        bp_set = breakpoint_indices if breakpoint_indices else set()
        for _, node_idx, name in sig_nodes:
            if node_idx != anchor_idx and node_idx not in bp_set and not is_cvt_like(name):
                candidates.add(name)

        # Pattern key: relation chain + endpoint
        key = (tuple(rel_chain), endpoint_name)

        # Compute causal tier for this raw path
        covered = path.get("covered_steps", frozenset())
        tier = score_causal_tier(covered, bridge_length=path.get("depth", 0))

        if key not in patterns:
            patterns[key] = {
                "rel_chain": rel_chain,
                "endpoint": endpoint_name,
                "candidates": set(),
                "raw_paths": [],
                "best_tier": (0, -1, 0),
                "best_raw_path": path,
                "best_depth": path.get("depth", 0),
            }
        patterns[key]["candidates"].update(candidates)
        if tier > patterns[key]["best_tier"]:
            patterns[key]["best_tier"] = tier
            patterns[key]["best_raw_path"] = path
            patterns[key]["best_depth"] = path.get("depth", 0)
        elif tier == patterns[key]["best_tier"] and path.get("depth", 0) < patterns[key].get("best_depth", 10**9):
            patterns[key]["best_raw_path"] = path
            patterns[key]["best_depth"] = path.get("depth", 0)
        if len(patterns[key]["raw_paths"]) < 200:
            patterns[key]["raw_paths"].append(path)

    # Convert to sorted list — rank only by the best causal tier of the pattern
    result = []
    for key, group in sorted(patterns.items(), key=lambda x: x[1]["best_tier"], reverse=True):
        cands = sorted(group["candidates"])[:20]
        # Use one witness path to preserve bridge structure for display.
        readable = _raw_path_to_readable(group["best_raw_path"])

        result.append({
            "rel_chain": group["rel_chain"],
            "endpoint": group["endpoint"],
            "candidates": cands,
            "best_tier": group["best_tier"],
            "readable": readable,
            "raw_paths": group["raw_paths"],
            "best_raw_path": group["best_raw_path"],
        })
    return result


def expand_to_triples(paths, ents, rels_list):
    """Expand raw paths to triples, bridging through CVT intermediate nodes.

    When a triple endpoint is a CVT node (m.xxx / g.xxx), we bridge forward
    through consecutive CVT nodes to the next real entity, merging relation
    names with " > ". This preserves connectivity while hiding gibberish.
    """
    triples = []
    seen = set()
    for path in paths:
        nodes = path["nodes"]
        rels = path["relations"]
        # Build raw (h_name, r_text, t_name) triples
        raw = []
        for i in range(min(len(rels), len(nodes) - 1)):
            h_idx, r_idx, t_idx = nodes[i], rels[i], nodes[i + 1]
            h_name = ents[h_idx] if 0 <= h_idx < len(ents) else "?"
            t_name = ents[t_idx] if 0 <= t_idx < len(ents) else "?"
            r_text = rel_to_text(rels_list[r_idx]) if 0 <= r_idx < len(rels_list) else "?"
            raw.append((h_name, r_text, t_name))
        # Bridge through CVT nodes
        i = 0
        while i < len(raw):
            h, r, t = raw[i]
            if is_cvt_like(h):
                i += 1
                continue
            if is_cvt_like(t):
                merged_rels = [r]
                j = i
                while j < len(raw) - 1 and is_cvt_like(raw[j][2]):
                    j += 1
                    merged_rels.append(raw[j][1])
                final_t = raw[j][2]
                if not is_cvt_like(final_t):
                    merged_r = " > ".join(merged_rels)
                    sig = (normalize(h), normalize(merged_r), normalize(final_t))
                    if sig not in seen:
                        seen.add(sig)
                        triples.append((h, merged_r, final_t))
                    i = j + 1
                else:
                    # Preserve tail relation even if it ends at a CVT/value node.
                    sig = (normalize(h), normalize(r), normalize(final_t))
                    if sig not in seen:
                        seen.add(sig)
                        triples.append((h, r, final_t))
                    i += 1
            else:
                sig = (normalize(h), normalize(r), normalize(t))
                if sig not in seen:
                    seen.add(sig)
                    triples.append((h, r, t))
                i += 1
    return triples


def _extract_relation_segments_from_path(path, ents):
    """Collapse a raw path into relation segments.

    Keep:
    - non-CVT -> non-CVT segments
    - direct non-CVT -> ... -> CVT tail segment only when no later non-CVT entity
      has already been reached

    This preserves direct value constraints such as:
    country -> statistical_region.child_labor_percent -> g.xxx
    but removes tails such as:
    person -> religion -> Judaism -> membership -> CVT
    where the core relation has already reached a real entity.
    """
    sig_positions = []
    for i, node_idx in enumerate(path.get("nodes", [])):
        name = ents[node_idx] if 0 <= node_idx < len(ents) else ""
        if not is_cvt_like(name):
            sig_positions.append(i)
    segments = []
    for j in range(1, len(sig_positions)):
        prev_pos = sig_positions[j - 1]
        curr_pos = sig_positions[j]
        rel_seq = path.get("relations", [])[prev_pos:curr_pos]
        if rel_seq:
            segments.append(list(rel_seq))
    # Preserve the tail segment when the endpoint is a CVT node.
    # Endpoint CVTs need attribute expansion (rate, date, etc.).
    # Middle CVTs are just path intermediaries — their direction is already determined.
    nodes = path.get("nodes", [])
    rels = path.get("relations", [])
    if len(sig_positions) >= 1:
        last_node_pos = len(nodes) - 1
        last_node_name = ents[nodes[last_node_pos]] if 0 <= last_node_pos < len(nodes) else ""
        if is_cvt_like(last_node_name) and sig_positions[-1] < last_node_pos:
            tail_rel_seq = rels[sig_positions[-1]:]
            if tail_rel_seq:
                segments.append(list(tail_rel_seq))
    return segments


class PatternEvidence:
    """Triples and metadata for one relation pattern, used in Stage 8 reasoning."""
    __slots__ = ("label", "readable", "candidates", "triples", "tree_data")

    def __init__(self, label, readable, candidates, triples, tree_data=None):
        self.label = label
        self.readable = readable
        self.candidates = candidates
        self.triples = triples
        self.tree_data = tree_data  # (step_ent_names, step_edge_names, cvt_attrs, cvt_to_named)


def _is_noisy_path_relation(rel_name):
    rel_name = rel_name or ""
    noisy_prefixes = (
        "type.",
        "common.",
        "freebase.",
        "kg.",
        "user.",
        "base.ontologies.",
    )
    noisy_shorts = {
        "type",
        "types",
        "instance",
        "instances",
        "notable_types",
        "topic_equivalent_webpage",
        "webpage",
        "mid",
        "guid",
        "key",
        "keys",
        "permission",
        "is_reviewed",
    }
    return rel_name.startswith(noisy_prefixes) or rel_name.rsplit(".", 1)[-1] in noisy_shorts


def _path_relation_names(path, rels_list):
    return [
        rel_to_text(rels_list[rel_idx]) if 0 <= rel_idx < len(rels_list) else "?"
        for rel_idx in path.get("relations", [])
    ]


def _logical_path_needs_endpoint_rescue(lp, rels_list, min_depth=4):
    best = lp.get("best_raw_path") or {}
    rel_names = _path_relation_names(best, rels_list)
    return best.get("depth", len(best.get("relations", []))) >= min_depth or any(
        _is_noisy_path_relation(rel_name) for rel_name in rel_names
    )


def build_endpoint_rescue_patterns(paths, selected_patterns, ents, rels_list,
                                   anchor_idx, breakpoint_indices,
                                   max_paths=4):
    """Build endpoint-constrained rescue patterns for noisy/overlong selections.

    This is a Stage 8 evidence supplement: when selected logical paths reach an
    endpoint through noisy or overlong chains, expose direct anchor-to-endpoint
    raw paths ranked by step coverage first, then noise and depth.
    """
    if not paths or not breakpoint_indices:
        return []
    if not all(_logical_path_needs_endpoint_rescue(lp, rels_list) for lp in selected_patterns):
        return []

    endpoint_set = {idx for idx in breakpoint_indices if idx is not None and idx != anchor_idx}
    if not endpoint_set:
        return []

    rescue_candidates = []
    for path in paths:
        nodes = path.get("nodes", [])
        rels = path.get("relations", [])
        if not nodes or not rels:
            continue
        hit_pos = None
        for pos, node_idx in enumerate(nodes):
            if node_idx in endpoint_set:
                hit_pos = pos
                break
        if hit_pos is None or hit_pos == 0:
            continue
        trimmed = dict(path)
        trimmed["nodes"] = nodes[:hit_pos + 1]
        trimmed["relations"] = rels[:hit_pos]
        trimmed["depth"] = len(trimmed["relations"])
        rel_names = _path_relation_names(trimmed, rels_list)
        noisy_count = sum(1 for rel_name in rel_names if _is_noisy_path_relation(rel_name))
        covered_count = len(trimmed.get("covered_steps", frozenset()))
        rescue_candidates.append((-covered_count, noisy_count, trimmed["depth"], trimmed))

    if not rescue_candidates:
        return []

    rescue_candidates.sort(key=lambda item: item[:3])
    support = []
    seen = set()
    for _, _, _, path in rescue_candidates:
        sig = (tuple(path.get("nodes", [])), tuple(path.get("relations", [])))
        if sig in seen:
            continue
        seen.add(sig)
        support.append(path)
        if len(support) >= max_paths:
            break
    if not support:
        return []

    best = support[0]
    endpoint_name = ents[best["nodes"][-1]] if 0 <= best["nodes"][-1] < len(ents) else None
    readable_parts = []
    for i, node_idx in enumerate(best.get("nodes", [])):
        if i == 0:
            readable_parts.append(ents[node_idx] if 0 <= node_idx < len(ents) else "?")
            continue
        rel_idx = best["relations"][i - 1]
        rel_text = rel_to_text(rels_list[rel_idx]) if 0 <= rel_idx < len(rels_list) else "?"
        node_text = f"node{i}"
        if node_idx in endpoint_set:
            node_text += "[endpoint]"
        readable_parts.append(f"--[{rel_text}]--> {node_text}")

    candidates = []
    for path in support:
        for node_idx in path.get("nodes", []):
            if node_idx == anchor_idx or node_idx in endpoint_set:
                continue
            if 0 <= node_idx < len(ents):
                name = ents[node_idx]
                if not is_cvt_like(name):
                    candidates.append(name)
    deduped_candidates = []
    seen_candidates = set()
    for name in candidates:
        key = normalize(name)
        if key in seen_candidates:
            continue
        seen_candidates.add(key)
        deduped_candidates.append(name)

    return [{
        "rel_chain": [],
        "endpoint": endpoint_name,
        "candidates": deduped_candidates[:20],
        "best_tier": (len(best.get("covered_steps", frozenset())), -best.get("depth", 0), 0),
        "readable": "ENDPOINT RESCUE: " + " ".join(readable_parts),
        "raw_paths": support,
        "best_raw_path": best,
    }]



# ── Relation Preflight Probe (Stage 4.5) ──

RELATION_PREFLIGHT_LEVEL0_FILTER = False
RELATION_PREFLIGHT_MAX_FRONTIER = 80
RELATION_PREFLIGHT_LARGE_FANOUT = 50


def _anchor_reachability_filter(anchor_idx, step_relations, adj):
    """Level 0: BFS from anchor, collect relations reachable within hop budget.

    Step i budget = i + 2 hops (step 0: 2, step 1: 3, step 2: 4, ...).
    Returns: (filtered_step_relations, debug_info)
    """
    n_steps = len(step_relations)
    max_budget = n_steps + 1

    # BFS from anchor, collect reachable relations cumulatively at each hop
    reachable_rel_by_hop = {}
    visited = {anchor_idx}
    frontier = {anchor_idx}
    all_reachable_rels = set()

    for hop in range(1, max_budget + 1):
        hop_rels = set()
        next_frontier = set()
        for node in frontier:
            for nb, rel in adj.get(node, []):
                hop_rels.add(rel)
                if nb not in visited:
                    visited.add(nb)
                    next_frontier.add(nb)
        all_reachable_rels |= hop_rels
        reachable_rel_by_hop[hop] = set(all_reachable_rels)
        frontier = next_frontier

    # Filter each step's relations against its hop budget
    filtered = []
    removed_debug = []
    for si, rs in enumerate(step_relations):
        budget = si + 2
        reachable = reachable_rel_by_hop.get(budget, set())
        kept = [r for r in rs if r in reachable]
        removed = [r for r in rs if r not in reachable]
        filtered.append(kept)
        if removed:
            removed_debug.append({"step": si, "budget": budget, "removed": removed})
    return filtered, removed_debug


def relation_preflight_probe(anchor_idx, step_relations, steps,
                             h_ids, r_ids, t_ids, ents, rels_list,
                             max_frontier=RELATION_PREFLIGHT_MAX_FRONTIER):
    """Frontier-conditioned relation preflight.

    Level 0: anchor reachability pre-filter (BFS with step-dependent hop budget).
    Level 1/2: diagnostic metrics from current frontier.
    """
    if anchor_idx is None or not step_relations:
        return step_relations, {"enabled": False, "reason": "missing_anchor_or_relations"}

    adj = {}
    for h_idx, r_idx, t_idx in zip(h_ids, r_ids, t_ids):
        adj.setdefault(h_idx, []).append((t_idx, r_idx))
        adj.setdefault(t_idx, []).append((h_idx, r_idx))

    # ── Level 0: anchor reachability pre-filter ──
    l0_filtered, l0_removed = _anchor_reachability_filter(anchor_idx, step_relations, adj)
    if RELATION_PREFLIGHT_LEVEL0_FILTER:
        step_relations = l0_filtered

    def _rel_name(rel_idx):
        return rel_to_text(rels_list[rel_idx]) if 0 <= rel_idx < len(rels_list) else "?"

    def _sample(items):
        return sorted(items)[:max_frontier]

    frontier = {anchor_idx}
    committed = {anchor_idx}
    filtered_layers = []
    debug_steps = []

    for step_idx, rels_for_step in enumerate(step_relations):
        rels_ordered = list(rels_for_step)
        next_rels = set(step_relations[step_idx + 1]) if step_idx + 1 < len(step_relations) else set()
        rel_reports = []
        reachable_rels = []
        next_frontier_by_rel = {}

        for rel_idx in rels_ordered:
            hits = set()
            loop_hits = 0
            for node_idx in frontier:
                for nb_idx, edge_rel in adj.get(node_idx, []):
                    if edge_rel != rel_idx:
                        continue
                    if nb_idx in committed:
                        loop_hits += 1
                        continue
                    hits.add(nb_idx)

            cvt_hits = {idx for idx in hits if 0 <= idx < len(ents) and is_cvt_like(ents[idx])}
            normal_hits = hits - cvt_hits
            sidecar_next_hits = 0
            direct_next_hits = 0
            if next_rels:
                for hit_idx in _sample(hits):
                    for nb_idx, edge_rel in adj.get(hit_idx, []):
                        if edge_rel in next_rels:
                            direct_next_hits += 1
                            break
                    if hit_idx in cvt_hits:
                        for nb_idx, edge_rel in adj.get(hit_idx, []):
                            if edge_rel in next_rels:
                                sidecar_next_hits += 1
                                break

            rel_text = _rel_name(rel_idx)
            reachable = bool(hits)
            next_reachable = (not next_rels) or direct_next_hits > 0 or sidecar_next_hits > 0
            noisy = _is_noisy_path_relation(rel_text)
            fanout = len(hits)
            status = "PASS"
            reason = "reachable from current frontier"
            if not reachable:
                status = "BLOCK"
                reason = "unreachable from current frontier"
            elif fanout > RELATION_PREFLIGHT_LARGE_FANOUT:
                status = "WARN"
                reason = "large fanout"
            elif noisy:
                status = "WARN"
                reason = "noisy relation"
            elif next_rels and not next_reachable:
                status = "WARN"
                reason = "no next-step reachability in local probe"

            if reachable:
                reachable_rels.append(rel_idx)
                next_frontier_by_rel[rel_idx] = hits

            rel_reports.append({
                "rel_idx": rel_idx,
                "rel": rel_text,
                "status": status,
                "reason": reason,
                "hit_count": fanout,
                "normal_hit_count": len(normal_hits),
                "cvt_hit_count": len(cvt_hits),
                "loop_hits": loop_hits,
                "direct_next_hits": direct_next_hits,
                "sidecar_next_hits": sidecar_next_hits,
                "next_reachable": next_reachable,
                "noisy": noisy,
            })

        # Level 1 filter: only keep reachable relations (if L0 already ran, this
        # catches relations that passed L0 but are unreachable from *current* frontier)
        if RELATION_PREFLIGHT_LEVEL0_FILTER and reachable_rels:
            accepted = [rel_idx for rel_idx in rels_ordered if rel_idx in set(reachable_rels)]
            fallback_keep_original = False
        else:
            accepted = rels_ordered
            fallback_keep_original = bool(rels_ordered and not reachable_rels)

        next_frontier = set()
        for rel_idx in accepted:
            next_frontier.update(next_frontier_by_rel.get(rel_idx, set()))
        if next_frontier:
            frontier = set(_sample(next_frontier))
            committed.update(frontier)

        filtered_layers.append(accepted)
        debug_steps.append({
            "step": steps[step_idx].get("step", step_idx + 1) if step_idx < len(steps) else step_idx + 1,
            "frontier_size": len(frontier),
            "input_relations": rels_ordered,
            "accepted_relations": accepted,
            "fallback_keep_original": fallback_keep_original,
            "relation_reports": rel_reports,
        })

    return filtered_layers, {
        "enabled": True,
        "mode": "level0_multihop_filter_level1_diagnostic",
        "level0_removed": l0_removed,
        "level0_active": RELATION_PREFLIGHT_LEVEL0_FILTER,
        "steps": debug_steps,
    }


def build_pattern_evidence_triples(selected_patterns, ents, rels_list, h_ids, r_ids, t_ids,
                                   anchor_idx, max_grouped_lines=120):
    """Build bounded pattern evidence from the witness path plus sibling raw paths.

    The Stage 7 logical path is grouped from many raw paths, but a single witness
    path is often too narrow for Stage 8 reasoning. Here we keep the witness path
    and then add a bounded number of sibling raw paths that introduce distinct
    non-CVT terminal entities, so the evidence remains representative without
    exploding into full pattern-level expansion.
    """
    node_edges = {}
    for i in range(len(h_ids)):
        h, r, t = h_ids[i], r_ids[i], t_ids[i]
        node_edges.setdefault(h, []).append((h, r, t))
        node_edges.setdefault(t, []).append((h, r, t))

    anchor_name = ents[anchor_idx] if 0 <= anchor_idx < len(ents) else ""

    def _make_adder(triples_list, seen_set):
        def _add(h_idx, r_idx, t_idx):
            h_name = ents[h_idx] if 0 <= h_idx < len(ents) else "?"
            t_name = ents[t_idx] if 0 <= t_idx < len(ents) else "?"
            r_text = rel_to_text(rels_list[r_idx]) if 0 <= r_idx < len(rels_list) else "?"
            if not h_name.isascii() or not t_name.isascii():
                return
            if len(normalize(h_name)) < 2 or len(normalize(t_name)) < 2:
                return
            sig = (normalize(h_name), normalize(r_text), normalize(t_name))
            if sig not in seen_set:
                seen_set.add(sig)
                triples_list.append((h_name, r_text, t_name))
        return _add

    def _meta_rel_priority(rel_name):
        rel_name = rel_name or ""
        short = rel_name.rsplit(".", 1)[-1]
        noisy_prefixes = (
            "type.",
            "common.",
            "freebase.",
            "kg.",
            "user.",
            "base.ontologies.",
        )
        noisy_shorts = {
            "type",
            "types",
            "instance",
            "instances",
            "notable_types",
            "topic_equivalent_webpage",
            "webpage",
            "mid",
            "guid",
            "key",
            "keys",
            "permission",
        }
        return rel_name.startswith(noisy_prefixes) or short in noisy_shorts

    def _expand_endpoint_cvt(cvt_idx, add, witness_nodes):
        scored = []
        witness_node_set = set(witness_nodes)
        for h_idx, r_idx, t_idx in node_edges.get(cvt_idx, []):
            other_idx = t_idx if h_idx == cvt_idx else h_idx
            other_name = ents[other_idx] if 0 <= other_idx < len(ents) else ""
            if not other_name or is_cvt_like(other_name):
                continue
            rel_name = rels_list[r_idx] if 0 <= r_idx < len(rels_list) else ""
            score = (
                0 if other_idx in witness_node_set else 1,
                1 if _meta_rel_priority(rel_name) else 0,
                0 if other_name.isascii() else 1,
                len(normalize(other_name)) < 2,
                rel_to_text(rel_name) if rel_name else "",
                other_name,
            )
            scored.append((score, h_idx, r_idx, t_idx))
        for _, h_idx, r_idx, t_idx in sorted(scored):
            add(h_idx, r_idx, t_idx)

    def _expand_sibling_cvts(path_nodes, path_rels, add):
        """For each CVT in the path, find and expand ALL sibling CVTs reachable
        via the same (head_entity, relation) pair.

        Key: must add the parent edge (head, rel, sibling_cvt) BEFORE expanding
        the sibling CVT's attributes, so the formatter sees it as pattern evidence.
        """
        expanded_rels = set()
        for i, node_idx in enumerate(path_nodes):
            if i == 0:
                continue
            node_name = ents[node_idx] if 0 <= node_idx < len(ents) else ""
            if not is_cvt_like(node_name):
                continue
            prev_idx = path_nodes[i - 1]
            rel_idx = path_rels[i - 1] if i - 1 < len(path_rels) else None
            if rel_idx is None:
                continue
            sig = (prev_idx, rel_idx)
            if sig in expanded_rels:
                continue
            expanded_rels.add(sig)
            for edge_h, edge_r, edge_t in node_edges.get(prev_idx, []):
                if edge_h != prev_idx or edge_r != rel_idx:
                    continue
                t_name = ents[edge_t] if 0 <= edge_t < len(ents) else ""
                if not is_cvt_like(t_name):
                    continue
                if edge_t != node_idx:
                    # Add parent edge first so formatter recognizes sibling
                    add(prev_idx, rel_idx, edge_t)
                    _expand_endpoint_cvt(edge_t, add, path_nodes)

    def _path_sig(path):
        return (tuple(path.get("nodes", [])), tuple(path.get("relations", [])))

    def _support_diversity_key(path):
        nodes = path.get("nodes", [])
        non_cvt_nodes = []
        cvt_nodes = []
        for node_idx in nodes:
            if 0 <= node_idx < len(ents):
                name = ents[node_idx]
                if is_cvt_like(name):
                    cvt_nodes.append(normalize(name))
                else:
                    non_cvt_nodes.append(normalize(name))
        return (tuple(non_cvt_nodes), tuple(cvt_nodes))

    def _select_support_paths(lp, max_paths=24):
        witness = lp.get("best_raw_path")
        if not witness:
            return []

        support = [witness]
        seen_path_sigs = {_path_sig(witness)}
        seen_diversity = {_support_diversity_key(witness)}

        for rp in lp.get("raw_paths", []):
            sig = _path_sig(rp)
            if sig in seen_path_sigs:
                continue
            diversity_key = _support_diversity_key(rp)
            if diversity_key in seen_diversity:
                continue
            support.append(rp)
            seen_path_sigs.add(sig)
            seen_diversity.add(diversity_key)
            if len(support) >= max_paths:
                break
        return support

    result = {}

    for pat_idx, lp in enumerate(selected_patterns):
        label = f"P{pat_idx + 1}"
        witness = lp.get("best_raw_path")
        if not witness:
            continue

        pat_triples = []
        pat_seen = set()
        add = _make_adder(pat_triples, pat_seen)

        support_paths = _select_support_paths(lp)
        expanded_cvts = set()
        for sp in support_paths:
            sp_nodes = sp.get("nodes", [])
            sp_rels = sp.get("relations", [])
            for i in range(min(len(sp_rels), len(sp_nodes) - 1)):
                add(sp_nodes[i], sp_rels[i], sp_nodes[i + 1])
            for node_idx in sp_nodes:
                node_name = ents[node_idx] if 0 <= node_idx < len(ents) else ""
                if is_cvt_like(node_name) and node_idx not in expanded_cvts:
                    _expand_endpoint_cvt(node_idx, add, sp_nodes)
                    expanded_cvts.add(node_idx)
            _expand_sibling_cvts(sp_nodes, sp_rels, add)

        witness_nodes = witness.get("nodes", [])

        cand_list = sorted(lp.get("candidates", []))[:20]

        sig_nodes = []
        for node_idx in witness_nodes:
            if 0 <= node_idx < len(ents):
                name = ents[node_idx]
                if not is_cvt_like(name):
                    sig_nodes.append((node_idx, name))

        step_ent_names = []
        if sig_nodes:
            step_ent_names = [[name] for _, name in sig_nodes]
        elif anchor_name:
            step_ent_names = [[anchor_name]]

        step_edge_names = []
        if len(sig_nodes) >= 2:
            for i in range(len(sig_nodes) - 1):
                from_name = sig_nodes[i][1]
                to_name = sig_nodes[i + 1][1]
                step_edge_names.append({from_name: {to_name}})

        # Resolve CVT attributes for inline display
        cvt_attrs = {}
        cvt_to_named = {}  # CVT -> named entity mapping
        for h_name, r_text, t_name in pat_triples:
            if is_cvt_like(h_name):
                attr_name = r_text.rsplit('.', 1)[-1] if '.' in r_text else r_text
                cvt_attrs.setdefault(h_name, []).append(f"{attr_name}={t_name}")
            # Track CVT -> named entity mappings (for TAIL-side CVT resolution)
            if is_cvt_like(t_name) and not is_cvt_like(h_name):
                cvt_to_named[t_name] = h_name

        if not pat_triples:
            continue

        def _cvt_attr_display(cvt_idx, limit=20):
            # Group attributes by prefix for cleaner display
            prefix_groups = {}
            for h_idx, r_idx, t_idx in node_edges.get(cvt_idx, []):
                other_idx = t_idx if h_idx == cvt_idx else h_idx
                other_name = ents[other_idx] if 0 <= other_idx < len(ents) else ""
                if not other_name or is_cvt_like(other_name):
                    continue
                rel_name = rels_list[r_idx] if 0 <= r_idx < len(rels_list) else ""
                short = rel_to_text(rel_name).rsplit(".", 1)[-1] if rel_name else "?"
                if t_idx == cvt_idx and h_idx != cvt_idx:
                    short = f"{short}.inv"
                # Extract prefix (e.g. "government.government_position_held")
                parts = rel_name.rsplit(".", 1) if rel_name else ["?"]
                prefix = parts[0] if len(parts) > 1 else ""
                prefix_groups.setdefault(prefix, []).append((short, other_name))

            attrs = []
            for prefix, items in prefix_groups.items():
                for short, val in items:
                    attrs.append(f"{short}={val}")
            return attrs[:limit]

        def _node_display(node_idx, expand_full=False):
            name = ents[node_idx] if 0 <= node_idx < len(ents) else "?"
            if is_cvt_like(name):
                path_attrs = cvt_attrs.get(name, [])
                if expand_full:
                    graph_attrs = _cvt_attr_display(node_idx)
                    seen = set(path_attrs)
                    merged = list(path_attrs)
                    for a in graph_attrs:
                        if a not in seen:
                            merged.append(a)
                            seen.add(a)
                    if merged:
                        return f"{name}: [" + ", ".join(merged[:20]) + "]"
                else:
                    if path_attrs:
                        return f"{name}: [" + ", ".join(path_attrs[:20]) + "]"
                return f"{name}: []"
            return name

        tree_paths = []
        tree_seen = set()
        for sp in support_paths:
            sp_nodes = sp.get("nodes", [])
            sp_rels = sp.get("relations", [])
            if not sp_nodes or not sp_rels:
                continue
            n = len(sp_nodes)
            display_nodes = [_node_display(idx, expand_full=(pos >= n - 2)) for pos, idx in enumerate(sp_nodes)]
            display_rels = [
                rel_to_text(rels_list[rel_idx]) if 0 <= rel_idx < len(rels_list) else "?"
                for rel_idx in sp_rels[: max(0, len(display_nodes) - 1)]
            ]
            sig = (tuple(display_nodes), tuple(display_rels))
            if sig in tree_seen:
                continue
            tree_seen.add(sig)
            tree_paths.append({"nodes": display_nodes, "relations": display_rels})

        result[label] = PatternEvidence(
            label=label,
            readable=lp.get("readable", ""),
            candidates=cand_list,
            triples=pat_triples,
            tree_data={"paths": tree_paths},
        )

    return result


def format_subgraph_with_cvt(triples, max_lines=80, max_tails=12):
    """Format triples for LLM reasoning with inline CVT resolution.

    Normal triples: grouped by (head, relation) as before.
    CVT nodes are resolved inline:
      Entity → relation → [attr1=val1, attr2=val2]
    Orphan CVTs (no parent entity) fall back to bare block display.
    """
    def _is_noisy_cvt_attr(rel_name):
        rel_name = rel_name or ""
        short = rel_name.rsplit(".", 1)[-1]
        noisy_prefixes = (
            "type.",
            "common.",
            "freebase.",
            "kg.",
            "user.",
            "base.ontologies.",
        )
        noisy_shorts = {
            "type",
            "types",
            "instance",
            "instances",
            "notable_types",
            "topic_equivalent_webpage",
            "webpage",
            "mid",
            "guid",
            "key",
            "keys",
            "permission",
        }
        return rel_name.startswith(noisy_prefixes) or short in noisy_shorts

    def _sort_attrs(attrs):
        return sorted(
            attrs,
            key=lambda x: (
                1 if _is_noisy_cvt_attr(x[0]) else 0,
                1 if is_cvt_like(x[1]) else 0,
                0 if x[1].isascii() else 1,
                len(normalize(x[1])) < 2,
                x[0],
                x[1],
            ),
        )

    clean = []
    cvt_attrs = {}  # cvt_name -> [(short_rel, value)]
    cvt_parents = {}  # cvt_name -> [(parent_entity, parent_rel)]

    for h, r, t in triples:
        h_cvt = is_cvt_like(h)
        t_cvt = is_cvt_like(t)

        if not h_cvt and not t_cvt:
            clean.append((h, r, t))
        elif h_cvt and t_cvt:
            continue
        elif h_cvt:
            short = r.split(".")[-1]
            cvt_attrs.setdefault(h, []).append((short, t))
        elif t_cvt:
            cvt_parents.setdefault(t, []).append((h, r))

    # Resolve CVT nodes inline: merge parent edge + CVT attributes
    resolved_cvt = set()
    cvt_inline = []  # (parent, rel, attrs_str) for inline display
    for cvt_name, parents in cvt_parents.items():
        attrs = list(cvt_attrs.get(cvt_name, []))
        if len(parents) > 1:
            for extra_parent, extra_rel in parents[1:]:
                attrs.append((extra_rel.split(".")[-1] + ".inv", extra_parent))
        attrs = _sort_attrs(attrs)
        if parents and attrs:
            attr_str = ", ".join(f"{short}={val}" for short, val in attrs[:20])
            parent, rel = parents[0]
            cvt_inline.append((parent, rel, attr_str))
            resolved_cvt.add(cvt_name)
        elif parents:
            # Path CVT with no expanded attrs — keep visible as unresolved reference
            parent, rel = parents[0]
            cvt_inline.append((parent, rel, f"CVT:{cvt_name}"))
            resolved_cvt.add(cvt_name)

    lines = []

    # Part 1: Normal triples (existing grouped format)
    if clean:
        grouped_lines = format_grouped_triples(clean, max_lines=max_lines, max_tails=max_tails)
        lines.extend(grouped_lines)

    # Part 2: Inline-resolved CVT edges
    if cvt_inline:
        if lines:
            lines.append("")
        # Group by (parent, rel) for compression
        inline_groups = {}
        for parent, rel, attr_str in cvt_inline:
            inline_groups.setdefault((parent, rel), []).append(attr_str)
        for (parent, rel), attr_strs in inline_groups.items():
            if len(lines) >= max_lines:
                break
            if len(attr_strs) == 1:
                lines.append(f"({parent}, {rel}, [{attr_strs[0]}])")
            else:
                shown = attr_strs[:max_tails]
                suffix = f", ...(+{len(attr_strs) - len(shown)})" if len(attr_strs) > len(shown) else ""
                lines.append(f"({parent}, {rel}, [{'][  ['.join(shown)}{suffix}])")

    # Part 3: Orphan CVTs (no parent connection) — bare block display
    orphan_cvts = {k: v for k, v in cvt_attrs.items() if k not in resolved_cvt}
    if orphan_cvts:
        if lines:
            lines.append("")
        for cvt_name, attrs in orphan_cvts.items():
            if len(lines) >= max_lines:
                break
            attrs = _sort_attrs(attrs)
            lines.append(f"[{cvt_name}]")
            for short_rel, val in attrs:
                if len(lines) >= max_lines:
                    break
                lines.append(f"  {short_rel} → {val}")

    return "\n".join(lines)


def format_pattern_evidence(pattern_evidence, max_total_lines=120, max_per_pattern=50, max_tails=12):
    """Format pattern-grouped evidence as entity tree for LLM reasoning.

    Each pattern shows a tree of entities at each node position, grouped by
    the first-level entity. CVT nodes are resolved to attribute displays.
    """
    lines = []

    for label, pe in pattern_evidence.items():
        if not pe.triples:
            continue

        # Header with readable path
        lines.append(f"=== PATTERN {label}: {pe.readable} ===")

        if pe.candidates:
            lines.append(f"  Candidates: {', '.join(pe.candidates[:8])}")
        tree_lines = _render_path_tree(pe.tree_data, max_lines=max_per_pattern)
        if tree_lines:
            lines.append("  Evidence tree:")
            lines.extend(f"  {tl}" for tl in tree_lines)
        else:
            cvt_text = format_subgraph_with_cvt(pe.triples, max_lines=max_per_pattern, max_tails=50)
            for tl in cvt_text.split('\n'):
                lines.append(f"  {tl}")

        lines.append("")

    return "\n".join(lines)


def _render_path_tree(tree_data, max_lines=50, max_children=12):
    """Render support raw paths as a compact YAML-like trie.

    This keeps the relation pattern aligned with the actual materialized paths:
    each child is nested under the parent entity/CVT that really leads to it.
    """
    if not isinstance(tree_data, dict):
        return []
    paths = tree_data.get("paths") or []
    if not paths:
        return []

    root = {"name": None, "edges": {}}
    for path in paths:
        nodes = path.get("nodes") or []
        rels = path.get("relations") or []
        if not nodes:
            continue
        if root["name"] is None:
            root["name"] = nodes[0]
        cur = root
        for i, rel_name in enumerate(rels):
            if i + 1 >= len(nodes):
                break
            child_name = nodes[i + 1]
            rel_children = cur["edges"].setdefault(rel_name, {})
            cur = rel_children.setdefault(child_name, {"name": child_name, "edges": {}})

    if root["name"] is None:
        return []

    lines = [f"node0: {root['name']}"]

    def _is_cvt_display(name):
        return (
            isinstance(name, str)
            and (name.startswith("m.") or name.startswith("g.") or name.startswith("CVT:"))
        )

    def _parse_cvt_display(name):
        if not _is_cvt_display(name):
            return name, []
        if ": [" not in name or not name.endswith("]"):
            return name, []
        cvt_id, attrs_raw = name.split(": [", 1)
        attrs_raw = attrs_raw[:-1]
        if not attrs_raw:
            return cvt_id, []
        return cvt_id, [a.strip() for a in attrs_raw.split(", ") if a.strip()]

    def _extract_inv_entities(cvt_display_name):
        """Extract .inv attribute values from a CVT display string as separate named entities."""
        _, attrs = _parse_cvt_display(cvt_display_name)
        inv_entities = []
        for attr in attrs:
            if ".inv=" in attr:
                rel_short, entity = attr.split(".inv=", 1)
                inv_entities.append((rel_short, entity.strip()))
        return inv_entities

    def _format_cvt_display(cvt_id, attrs):
        return f"{cvt_id}: [" + ", ".join(attrs) + "]"

    def _compress_cvt_displays(child_names):
        if len(child_names) < 2 or not all(_is_cvt_display(name) for name in child_names):
            return [], child_names
        parsed = [_parse_cvt_display(name) for name in child_names]
        attr_sets = [set(attrs) for _, attrs in parsed]
        shared = set.intersection(*attr_sets) if attr_sets else set()
        shared_attrs = [attr for attr in parsed[0][1] if attr in shared]
        if not shared_attrs:
            return [], child_names
        compressed = []
        for cvt_id, attrs in parsed:
            own_attrs = [attr for attr in attrs if attr not in shared]
            compressed.append(_format_cvt_display(cvt_id, own_attrs))
        return shared_attrs, compressed

    def _fold_cvt_leaf_edges(node):
        if not node.get("edges"):
            return ""
        parts = []
        for rel_name in sorted(node.get("edges", {})):
            children = node["edges"][rel_name]
            if not all(not child.get("edges") for child in children.values()):
                return ""
            child_names = sorted(children)
            shown = child_names[:max_children]
            suffix = f" | ... (+{len(child_names) - len(shown)})" if len(child_names) > len(shown) else ""
            short_rel = rel_name.rsplit(".", 1)[-1]
            parts.append(f"{short_rel}=[" + " | ".join(shown) + suffix + "]")
        return "; ".join(parts)

    def _render_node(node, depth, indent):
        if len(lines) >= max_lines:
            return
        for rel_name in sorted(node.get("edges", {})):
            if len(lines) >= max_lines:
                return
            children = node["edges"][rel_name]
            lines.append(" " * indent + f"{rel_name}:")

            if all(not child.get("edges") for child in children.values()):
                child_names = sorted(children)
                if any(_is_cvt_display(child_name) for child_name in child_names):
                    shown_names = child_names[:max_children]
                    shared_attrs, shown_names = _compress_cvt_displays(shown_names)
                    if shared_attrs and len(lines) < max_lines:
                        lines.append(
                            " " * (indent + 2)
                            + "shared: [" + ", ".join(shared_attrs) + "]"
                        )
                    for child_name in shown_names:
                        if len(lines) >= max_lines:
                            return
                        lines.append(" " * (indent + 2) + f"- node{depth + 1}: {child_name}")
                        for inv_rel, inv_ent in _extract_inv_entities(child_name):
                            if len(lines) < max_lines:
                                lines.append(" " * (indent + 4) + f"← also {inv_rel}: {inv_ent}")
                    omitted = len(child_names) - len(shown_names)
                    if omitted > 0 and len(lines) < max_lines:
                        lines.append(" " * (indent + 2) + f"- ... (+{omitted})")
                    continue
                shown_names = child_names[:max_children]
                suffix = f" | ... (+{len(child_names) - len(shown_names)})" if len(child_names) > len(shown_names) else ""
                lines.append(
                    " " * (indent + 2)
                    + f"node{depth + 1}: [" + " | ".join(shown_names) + suffix + "]"
                )
                continue

            child_items = sorted(children.items())
            shown_items = child_items[:max_children]
            compressed_by_name = {}
            cvt_names = [name for name, _ in shown_items if _is_cvt_display(name)]
            if cvt_names:
                shared_attrs, compressed_names = _compress_cvt_displays(cvt_names)
                compressed_by_name = dict(zip(cvt_names, compressed_names))
                if shared_attrs and len(lines) < max_lines:
                    lines.append(
                        " " * (indent + 2)
                        + "shared: [" + ", ".join(shared_attrs) + "]"
                    )
            for child_name, child in shown_items:
                if len(lines) >= max_lines:
                    return
                child_display = compressed_by_name.get(child_name, child_name)
                folded = _fold_cvt_leaf_edges(child) if _is_cvt_display(child_name) else ""
                suffix = f" ; {folded}" if folded else ""
                lines.append(" " * (indent + 2) + f"- node{depth + 1}: {child_display}{suffix}")
                for inv_rel, inv_ent in _extract_inv_entities(child_display):
                    if len(lines) < max_lines:
                        lines.append(" " * (indent + 4) + f"← also {inv_rel}: {inv_ent}")
                if not folded:
                    _render_node(child, depth + 1, indent + 4)
            omitted = len(child_items) - len(shown_items)
            if omitted > 0 and len(lines) < max_lines:
                lines.append(" " * (indent + 2) + f"- ... (+{omitted})")

    _render_node(root, 0, 2)
    return lines


def _render_entity_tree(step_ent_names, step_edge_names, cvt_attrs,
                        max_branches=15, max_paths_per_branch=10, cvt_to_named=None):
    """Render entity tree with node position labels.

    Output format:
      node1=EntityA
        node2=X → node3=Y
        node2=Z
      node1=EntityB
        node2=W → node3=V
    """
    if len(step_ent_names) < 2 or not step_edge_names:
        return []

    if cvt_to_named is None:
        cvt_to_named = {}

    lines = []
    step1_ents = step_ent_names[1][:max_branches]
    num_steps = len(step_edge_names)

    def _resolve(name):
        # First check if CVT has attributes (original logic)
        if is_cvt_like(name) and name in cvt_attrs and cvt_attrs[name]:
            return "[" + ", ".join(cvt_attrs[name][:20]) + "]"
        # Bug fix: Check if CVT maps to a named entity (TAIL-side CVT resolution)
        # This handles cases like religions where CVT m.0493b56 -> Shia Islam
        if is_cvt_like(name) and name in cvt_to_named:
            return cvt_to_named[name]
        return name

    def _enumerate(step, entity, limit):
        """Return list of path-tail lists from this entity onward."""
        if step >= num_steps:
            return [[]]
        children = step_edge_names[step].get(entity, set())
        if not children:
            return [[]]
        result = []
        for child in sorted(children):
            if len(result) >= limit:
                break
            node_label = f"node{step + 1}"
            sub_paths = _enumerate(step + 1, child, limit - len(result))
            for sp in sub_paths:
                result.append([f"{node_label}={_resolve(child)}"] + sp)
        return result

    for s1 in step1_ents:
        paths = _enumerate(1, s1, max_paths_per_branch)
        if not paths or paths == [[]]:
            lines.append(f"  node1={s1}")
        else:
            lines.append(f"  node1={s1}")
            for p in paths:
                if p:
                    lines.append("    " + " → ".join(p))

    return lines


def format_grouped_triples(triples, max_lines=80, max_tails=50):
    """Group triples by (head, relation) and (relation, tail) to reduce token usage."""
    # Forward grouping: (head, relation) -> list of tails
    fwd_groups = {}
    for h, r, t in triples:
        fwd_groups.setdefault((h, r), []).append(t)

    # Deduplicate forward groups
    fwd_deduped = {}
    for (h, r), tails in fwd_groups.items():
        uniq = []
        seen = set()
        for t in tails:
            nt = normalize(t)
            if nt not in seen:
                seen.add(nt)
                uniq.append(t)
        fwd_deduped[(h, r)] = uniq

    # Identify reverse-compressible groups: (relation, tail) -> list of heads
    # Only when multiple heads share same (relation, tail)
    rev_groups = {}
    for (h, r), tails in fwd_deduped.items():
        for t in tails:
            rev_groups.setdefault((r, normalize(t)), []).append(h)

    # Mark which (head, relation) pairs are absorbed into a reverse group
    reverse_absorbed = set()
    reverse_lines = {}
    for (r, nt), heads in rev_groups.items():
        # Deduplicate heads
        uniq_heads = []
        seen_h = set()
        for h in heads:
            nh = normalize(h)
            if nh not in seen_h:
                seen_h.add(nh)
                uniq_heads.append(h)
        if len(uniq_heads) >= 2:
            # Find the original tail name from the triples
            orig_tail = None
            for (h2, r2), tails2 in fwd_deduped.items():
                for t2 in tails2:
                    if r2 == r and normalize(t2) == nt:
                        orig_tail = t2
                        break
                if orig_tail:
                    break
            if orig_tail and len(uniq_heads) <= max_tails:
                reverse_lines[(r, nt)] = (uniq_heads, r, orig_tail)
                for h in uniq_heads:
                    reverse_absorbed.add((normalize(h), r))

    lines = []
    for (h, r), tails in fwd_deduped.items():
        # Skip if this head is absorbed into a reverse group AND all tails match
        nh = normalize(h)
        if (nh, r) in reverse_absorbed:
            # Check if ALL tails of this (h, r) are absorbed into reverse groups
            all_absorbed = all(
                (r, normalize(t)) in reverse_lines for t in tails
            )
            if all_absorbed:
                continue
            # Partially absorbed — show only non-absorbed tails
            remaining = [t for t in tails if (r, normalize(t)) not in reverse_lines]
            if not remaining:
                continue
            tails = remaining

        if len(tails) == 1:
            lines.append(f"({h}, {r}, {tails[0]})")
        else:
            shown = tails[:max_tails]
            suffix = f", ...(+{len(tails) - len(shown)})" if len(tails) > len(shown) else ""
            lines.append(f"({h}, {r}, [{', '.join(shown)}{suffix}])")
        if len(lines) >= max_lines:
            break

    # Append reverse-compressed lines
    for (r, nt), (heads, rel, tail) in sorted(reverse_lines.items()):
        if len(lines) >= max_lines:
            break
        if len(heads) == 2:
            lines.append(f"([{heads[0]}, {heads[1]}], {rel}, {tail})")
        else:
            shown = heads[:max_tails]
            suffix = f", ...(+{len(heads) - len(shown)})" if len(heads) > len(shown) else ""
            lines.append(f"([{', '.join(shown)}{suffix}], {rel}, {tail})")

    return lines


def collect_local_subgraph_triples(seed_node_indices, ents, rels_list, h_ids, r_ids, t_ids,
                                   max_per_seed=6, global_limit=120):
    """Collect a small 1-hop local subgraph around selected witness/candidate nodes.

    This is for final reasoning only. It exposes local evidence around bridge nodes
    and candidate entities that is not visible from the witness path alone.
    """
    triples = []
    seen = set()

    def _add_triple(h_idx, r_idx, t_idx):
        h_name = ents[h_idx] if 0 <= h_idx < len(ents) else "?"
        t_name = ents[t_idx] if 0 <= t_idx < len(ents) else "?"
        r_text = rel_to_text(rels_list[r_idx]) if 0 <= r_idx < len(rels_list) else "?"
        # Filter out pure non-Latin entities (multilingual noise from Freebase)
        if not re.search(r'[a-zA-Z]', h_name) or not re.search(r'[a-zA-Z]', t_name):
            return
        sig = (normalize(h_name), normalize(r_text), normalize(t_name))
        if sig not in seen:
            seen.add(sig)
            triples.append((h_name, r_text, t_name))

    for node_idx in seed_node_indices:
        count = 0
        if node_idx is None or not (0 <= node_idx < len(ents)):
            continue
        for i in range(len(h_ids)):
            if len(triples) >= global_limit or count >= max_per_seed:
                break
            if h_ids[i] == node_idx or t_ids[i] == node_idx:
                _add_triple(h_ids[i], r_ids[i], t_ids[i])
                count += 1
        if len(triples) >= global_limit:
            break
    return triples


# ---------------------------------------------------------------------------
# NER-based entity resolution (validated approach from test_full_pipeline.py)
# ---------------------------------------------------------------------------

# Stage 1a: Entity analysis + question interpretation + rewrite
ENTITY_ANALYSIS_PROMPT = """Entities from knowledge graph: {entities}

Question: {question}

Task: Analyze the given entities, interpret the question, and rewrite it as a declarative sentence.

IMPORTANT: You may ONLY use entities from the list above as anchor or endpoints.
Do NOT infer, guess, or invent entities not in the list. If only one entity is given, it is the anchor.
The answer is usually NOT one of the listed entities. Listed non-anchor entities are constraints unless the
question explicitly asks for that exact entity.

## Entity Analysis
For each entity, assess its suitability as a traversal starting point (anchor).
Consider: how many possible graph neighbors does this entity have? Lower ambiguity = better anchor.
The entity list contains exact names from entity recognition — trust them as-is, do NOT judge
their validity. Even unusual names are valid entities.

Rate each entity: HIGH / MEDIUM / LOW ambiguity as starting point.
- Specific names (people, events, unique titles) → LOW ambiguity → good anchor
- Generic types (Country, Person, College/University) → HIGH ambiguity → bad anchor
- Active subject of the sentence → better anchor than entities in prepositional phrases

## Anchor Selection
Pick the entity with LOWEST ambiguity as anchor.
Rule: the ACTIVE subject of the sentence is preferred over locative constraints.
("Who founded X" → X is anchor; "in Y" → Y is endpoint)
Rule: When choosing between a specific named entity (event, unique title, person) and a
generic entity (country, place, sport), ALWAYS prefer the specific named entity — it has
fewer graph neighbors and produces more focused traversal.
("2010 FIFA World Cup" > "Spain"; "I Am... World Tour" > "Beyoncé")

## Endpoint Selection (optional)
From the REMAINING entities in the list, pick those that CONSTRAINT the answer path:
- Locative constraints: "in [Place]", "near [Location]"
- Co-participants: entities that shape the answer alongside the anchor
Skip generic type words. If none qualify or only one entity exists, output "none".

## Question Interpretation
Before rewriting, explain what the question is actually asking in your own words.
Identify: What is the answer's role? (person who did X, place where Y happened, year when Z occurred, etc.)
This interpretation guides the answer type and rewrite.
Use the wh-phrase to determine Answer_type:
- "Which country..." → country
- "Which region/globe region..." → region
- "Who..." → person
- "Where..." → location/place
Do NOT set Answer_type to an endpoint constraint such as a time zone, country, person, or date mentioned after
"with", "in", "near", "bordering", "during", or similar constraint phrases.

## Question Rewrite
Based on your interpretation, rewrite the question as a declarative sentence.
The rewrite must preserve all constraints and clearly express what entity/value is sought.
Start with "The [answer type] that..." or "The [answer type] where/when/which...".
The rewrite must contain a placeholder for the sought answer, e.g. "[Answer]".
Do NOT rewrite the endpoint constraint as the answer.

## Output format
Wrap each field in XML tags. Example outputs:

Example 1:
Entities from knowledge graph: Eiffel Tower
Question: What is the capital of the country where the Eiffel Tower is located?

<reasoning>Eiffel Tower is a unique landmark with LOW ambiguity as anchor. It determines a specific country.</reasoning>
<anchor>Eiffel Tower</anchor>
<endpoints>none</endpoints>
<interpretation>The question asks for the capital city of the country containing the Eiffel Tower. The answer is a city name.</interpretation>
<answer_type>city</answer_type>
<rewritten>The city [Answer] that is the capital of the country where the Eiffel Tower is located.</rewritten>

Example 2:
Entities from knowledge graph: Lou Seal
Question: Lou Seal is the mascot for the team that last won the World Series when?

<reasoning>Lou Seal is a unique mascot name with LOW ambiguity. It identifies one specific team.</reasoning>
<anchor>Lou Seal</anchor>
<endpoints>none</endpoints>
<interpretation>The question asks for the year when the sports team associated with mascot Lou Seal last won the World Series. The answer is a year.</interpretation>
<answer_type>year</answer_type>
<rewritten>The year [Answer] when the team for which Lou Seal is the mascot last won the World Series.</rewritten>

Example 3:
Entities from knowledge graph: Afghan National Anthem
Question: The national anthem Afghan National Anthem is from the country which practices what religion?

<reasoning>Afghan National Anthem is a unique title with LOW ambiguity. Only one entity available so it is the anchor.</reasoning>
<anchor>Afghan National Anthem</anchor>
<endpoints>none</endpoints>
<interpretation>The question asks for the religion practiced in the country that has the Afghan National Anthem. The answer is a religion name.</interpretation>
<answer_type>religion</answer_type>
<rewritten>The religion [Answer] that is practiced in the country which has the Afghan National Anthem as its national anthem.</rewritten>

Example 4:
Entities from knowledge graph: Brad Stevens | 2008 NBA Finals
Question: What year did the basketball team coached by Brad Stevens win the 2008 NBA Finals?

<reasoning>Brad Stevens is a specific person with LOW ambiguity. 2008 NBA Finals is a specific event but the question asks about the team's win year, not the event itself. Brad Stevens as the coaching starting point is the better anchor.</reasoning>
<anchor>Brad Stevens</anchor>
<endpoints>2008 NBA Finals</endpoints>
<interpretation>The question asks for the year when the team coached by Brad Stevens won. The answer is a year.</interpretation>
<answer_type>year</answer_type>
<rewritten>The year [Answer] when the basketball team coached by Brad Stevens won.</rewritten>

Now analyze the given question and entities. Output:

<reasoning>[one sentence per entity analyzing ambiguity]</reasoning>
<anchor>[chosen entity from the list only]</anchor>
<endpoints>[Entity from the list only, or "none"]</endpoints>
<interpretation>[1-2 sentences explaining what the question asks, what role the answer plays]</interpretation>
<answer_type>[free-form noun phrase: person, country, government_type, language, sport, event, year, religion, college, movie, sports team, location, currency, etc.]</answer_type>
<rewritten>[declarative sentence based on interpretation]</rewritten>"""

# Stage 1b: Chain decomposition (given anchor, endpoints, rewritten question)
CHAIN_PROMPT = """Entities: {entities}
Anchor: {anchor}
Endpoints: {endpoints}
Answer type: {answer_type}
Interpretation: {interpretation}
Rewritten question: {rewritten}
Original question: {question}

Decompose this question into a chain of abstract relation hops.
Each hop describes WHAT kind of graph edge to look for — NOT which specific entity it leads to.
Think first, then write the chain. Do NOT write the chain before doing the per-hop analysis.

IMPORTANT RULES:
1. This is QUESTION DECOMPOSITION, not graph traversal. You do NOT know what entities exist in the graph.
2. Intermediate nodes MUST be written as literally "node". NEVER fill in guessed entity names.
3. Each hop = ONE atomic relation. Compound actions MUST be split into separate hops.
4. No verification hops for superlatives ("find the most recent" → post-processing, not a hop).
5. No circular chains.
6. Endpoints are entities that must be reached or constrain the path.
7. Count hops carefully: each "that/which/who/where" clause typically adds ONE hop. Do NOT merge clauses.
8. "Return the answer", "the result is", "verify it", or "implicit return" is NOT a hop.
9. A constraint entity should appear only at the hop where it is reached or checked.
10. For "X that/which contains/has an airport that serves Y", start from Y with:
    Y -(served by airport)-> node -(airport located in country/region)-> node.
11. For "country/region with [endpoint constraint]", first find the answer node, then add one hop from that
    answer node to the endpoint constraint. Do NOT make the endpoint itself the answer.

Examples:
  Question: "What is the capital of the country where the Eiffel Tower is located?"
  Entities: Eiffel Tower
  Anchor: Eiffel Tower
  Chain: Eiffel Tower -(located in country)-> node -(capital city)-> node  [2 hops]

  Question: "What language is spoken in the country where the leader was appointed to office?"
  Entities: Leader
  Anchor: Leader
  Chain: Leader -(government position held)-> node -(jurisdiction of office)-> node -(language spoken)-> node  [3 hops]

  Question: "What sport does the most popular team in the country containing Paris play?"
  Entities: Paris
  Anchor: Paris
  Chain: Paris -(contained by country)-> node -(popular sport)-> node  [2 hops, "most popular" is post-processing]

  Question: "What country bordering a given country contains an airport that serves a given city?"
  Entities: [city] | [border country]
  Anchor: [city]
  Endpoints: [border country]
  Chain: [city] -(served by airport)-> node -(airport located in country)-> node -(borders country)-> node  [3 hops; endpoint is reached only by the border hop]

  Question: "Which region with a specified time zone contains a given country?"
  Entities: [country] | [time zone]
  Anchor: [country]
  Endpoints: [time zone]
  Chain: [country] -(contained in region)-> node -(has time zone)-> node  [2 hops; answer is the region node, endpoint is the time zone constraint]

WRONG Chain: Leader -(appointed to role in country where language is spoken)-> node  ← merged 3 hops into 1!
WRONG Chain: [city] -(located in country)-> node -(borders endpoint)-> node -(located in country)-> node  ← final "return" hop is fake and creates a loop.
WRONG Chain: [country] -(has time zone)-> node  ← answers the endpoint constraint, not the requested region.

Output (exact format — think first, chain last):
Reasoning: [1-3 short sentences. Identify the answer node and which endpoint constraints must be reached.]

Analysis:
1. [action description]
   - Keyword: core concept word
   - Definition: A concise dictionary-style definition of the relation concept in 15 words or less. Describe what the relation MEANS naturally. Do NOT use graph terminology (no "link", "node", "entity", "edge"). Examples: "neighboring countries sharing a common border", "the sport an athlete plays professionally", "a country's top-level administrative regions", "the official song representing a nation's identity".
   - Sub-question: a complete natural language question for this specific hop, as if asking a person (e.g., "What country contains this department?", "Who is the governor of this state?", "What sport does this team play?").

Chain:
{anchor} -(action description)-> node ... -(final action)-> node"""


def _token_overlap(question: str, entity: str) -> float:
    """Ratio of entity tokens found in question."""
    q_tokens = set(normalize(question).split())
    e_tokens = set(normalize(entity).split())
    if not e_tokens:
        return 0.0
    return len(e_tokens & q_tokens) / len(e_tokens)


async def resolve_anchor_ner(session, question, entity_list, rel_list, h_ids, r_ids, t_ids):
    """NER + GTE + token overlap + relation overlap scoring for entity resolution."""
    from collections import defaultdict

    clean_ents, seen = [], set()
    for e in entity_list:
        if not e or is_cvt_like(e) or len(e) <= 1:
            continue
        if e not in seen:
            clean_ents.append(e)
            seen.add(e)

    rel_texts = [rel_to_text(r) for r in rel_list]

    # GTE entity retrieval
    q_ent_rows = await gte_retrieve(session, question, clean_ents, top_k=12)
    gte_ents = [(r["candidate"], r.get("score", 0)) for r in q_ent_rows
                if r.get("candidate") and r["candidate"] in entity_list]

    # Token overlap filter (min 0.5)
    filtered = [(e, s) for e, s in gte_ents if _token_overlap(question, e) >= 0.5]

    # Relation overlap scoring
    q_rel_rows = await gte_retrieve(session, question, rel_list,
                                    candidate_texts=rel_texts, top_k=5)
    q_top_rel_idx = {rel_list.index(r["candidate"]) for r in q_rel_rows
                     if r.get("candidate") and r["candidate"] in rel_list}

    name_to_ids = defaultdict(list)
    for i, name in enumerate(entity_list):
        name_to_ids[name].append(i)

    scored = []
    for ent, gte_score in filtered:
        ent_idx = set(name_to_ids.get(ent, []))
        ent_rels = set()
        for h, r, t in zip(h_ids, r_ids, t_ids):
            if h in ent_idx or t in ent_idx:
                ent_rels.add(r)
        overlap = len(ent_rels & q_top_rel_idx)
        scored.append({"entity": ent, "gte": round(gte_score, 4), "overlap": overlap})
    scored.sort(key=lambda x: (-x["overlap"], -x["gte"]))

    return scored, name_to_ids


def _parse_ner_steps(raw):
    """Parse steps from NER decomposition format."""
    steps = []
    for line in raw.split("\n"):
        m = re.match(r"^\d+\.\s+(.+?)\s*\(relation:\s*(.+?)\)\s*$", line.strip())
        if m:
            steps.append({
                "step": len(steps) + 1,
                "question": m.group(1).strip(),
                "relation_query": m.group(2).strip(),
                "endpoint": None,
                "endpoint_query": None,
            })
    return steps


def _parse_ner_start(raw):
    m = re.search(r"^Start:\s*(.+)$", raw, re.M)
    return m.group(1).strip() if m else None


def _parse_ner_end(raw):
    m = re.search(r"^End:\s*(.+)$", raw, re.M)
    end = m.group(1).strip() if m else "none"
    if end.lower() == "none":
        return None, None
    return end, end


def _collect_hr_frontier(anchor_idx, step_relations, h_ids, r_ids, t_ids,
                         path_nodes=None, target_nodes=None, paths=None):
    """Collect HR frontier from path-level (h+r) triples.

    For each hop (h, r, t) in existing paths, collect ALL t' from KG where
    (h, r, t') exists. This supplements path entities with siblings missed
    by relation_prior_expand's beam_width limits.
    """
    if not paths:
        return [], set()

    # Collect unique (h, r) pairs from all path hops
    hr_pairs = set()
    for path in paths:
        nodes = path.get("nodes", [])
        relations = path.get("relations", [])
        for i in range(min(len(relations), len(nodes) - 1)):
            hr_pairs.add((nodes[i], relations[i]))

    if not hr_pairs:
        return [], set()

    all_nodes = set()
    for i in range(len(h_ids)):
        if (h_ids[i], r_ids[i]) in hr_pairs and t_ids[i] != anchor_idx:
            all_nodes.add(t_ids[i])

    return [], all_nodes


def _extract_path_candidates(paths, anchor_idx, ents, h_ids, r_ids, t_ids):
    """Extract deduplicated candidates from path traversal nodes only (no HR frontier).
    Used for GT recall — measures whether graph traversal can reach the answer.
    """
    path_nodes = {anchor_idx}
    for path in paths:
        path_nodes.update(path.get("nodes", []))
    cands = []
    for node_idx in sorted(path_nodes):
        if node_idx == anchor_idx:
            continue
        name = ents[node_idx] if 0 <= node_idx < len(ents) else ""
        if is_cvt_like(name):
            for cvt_idx, _ in expand_through_cvt(node_idx, h_ids, r_ids, t_ids, ents):
                if cvt_idx != anchor_idx and 0 <= cvt_idx < len(ents) and not is_cvt_like(ents[cvt_idx]):
                    cands.append(ents[cvt_idx])
        else:
            cands.append(name)
    seen = set()
    unique = []
    for c in cands:
        nc = normalize(c)
        if len(nc) < 2 or not c.isascii():
            continue
        if nc not in seen:
            seen.add(nc)
            unique.append(c)
    return unique


async def run_case(session, sample, pilot_row):
    question = pilot_row["question"]
    gt_answers = pilot_row.get("gt", pilot_row.get("ground_truth", pilot_row.get("gt_answers", [])))
    ents = sample.get("text_entity_list", []) + sample.get("non_text_entity_list", [])
    rels = list(sample.get("relation_list", []))
    h_ids, r_ids, t_ids = sample.get("h_id_list", []), sample.get("r_id_list", []), sample.get("t_id_list", [])

    _stage_times = {}

    # ── NER entity resolution (BEFORE expand_cvt_leaves, using original data) ──
    _t0 = time.perf_counter()
    ner_scored, ner_name_to_ids = await resolve_anchor_ner(
        session, question, ents, rels, h_ids, r_ids, t_ids)
    ner_top_ents = []
    _seen_e = set()
    for s in ner_scored[:6]:
        if s["entity"] not in _seen_e:
            ner_top_ents.append((s["entity"], s["gte"]))
            _seen_e.add(s["entity"])
    _stage_times["ner_resolve"] = time.perf_counter() - _t0

    # Auto-expand CVT leaf nodes (degree ≤ 1)
    _t0 = time.perf_counter()
    ents, rels, h_ids, r_ids, t_ids = expand_cvt_leaves(ents, rels, h_ids, r_ids, t_ids)
    rel_texts = list(rels)
    _stage_times["cvt_expand"] = time.perf_counter() - _t0

    # Build entity candidates (non-CVT)
    ent_candidates = [e for e in ents if e and len(e) > 1 and not is_cvt_like(e)]

    # Re-map NER entity names to expanded entity list indices
    ner_name_to_ids_expanded = {}
    for i, name in enumerate(ents):
        ner_name_to_ids_expanded.setdefault(name, []).append(i)

    async def execute_planning(anchor_forbidden=None, step_relations_override=None,
                               retry_note=None, use_ner=True):
        entity_retrieval_details = []
        anchor_idx = None
        anchor_name = None

        # ── NER-based anchor resolution (with fallback) ────────────────
        ner_ok = False
        if use_ner and ner_top_ents:
            decomp_question = f"Q: {question} [NER mode]"
            _ents = sample.get("q_entity", [])
            ent_str = "\n".join(f"- {e}" for e in _ents)
            first_ent = _ents[0].strip() if _ents else "Entity"
            prompt_text = CHAIN_PROMPT.format(
                entities=ent_str, anchor=first_ent, endpoints="none",
                answer_type="other", interpretation="", rewritten=question,
                question=question)
            raw = await call_llm(session, [
                {"role": "user", "content": prompt_text},
            ], max_tokens=2000)

            parsed = parse_chain(raw)
            if parsed and parsed.get('hops'):
                start_name = parsed['anchor']
                # Convert hops to steps format
                steps = []
                for i, hop in enumerate(parsed['hops']):
                    ep = None
                    for ep_info in parsed.get('endpoint_entities', []):
                        if ep_info.get('hop') == i + 1:
                            ep = ep_info['entity']
                            break
                    steps.append({
                        "step": i + 1,
                        "question": hop['relation'],
                        "type": "find",
                        "relation_query": hop.get('keyword', hop['relation']),
                        "definition": hop.get('definition', ''),
                        "keyword": hop.get('keyword', ''),
                        "endpoint": ep,
                        "endpoint_query": ep,
                        "entity_query": None,
                    })

                # Resolve anchor from NER scored entities
                if start_name:
                    sn = normalize(start_name)
                    for s in ner_scored:
                        if normalize(s["entity"]) == sn:
                            anchor_idx = ner_name_to_ids_expanded.get(s["entity"], [None])[0]
                            anchor_name = s["entity"]
                            break
                    if anchor_idx is None:
                        _cands = []
                        for s in ner_scored:
                            en = normalize(s["entity"])
                            if sn in en or en in sn:
                                _cands.append(s)
                        if _cands:
                            _cands.sort(key=lambda s: -len(s["entity"]))
                            anchor_idx = ner_name_to_ids_expanded.get(_cands[0]["entity"], [None])[0]
                            anchor_name = _cands[0]["entity"]
                if anchor_idx is None and ner_scored:
                    anchor_name = ner_scored[0]["entity"]
                    anchor_idx = ner_name_to_ids_expanded.get(anchor_name, [None])[0]

                entity_retrieval_details.append({
                    "role": "anchor_ner",
                    "ner_top_ents": ner_top_ents[:6],
                    "selected": anchor_name,
                    "selected_idx": anchor_idx,
                })

                # Endpoint resolve from parse_chain endpoint_entities
                breakpoints = {}
                for ep_info in parsed.get('endpoint_entities', []):
                    ep_name = ep_info.get('entity')
                    ep_hop = ep_info.get('hop')
                    if ep_name:
                        ep_rows = await gte_retrieve(session, ep_name, ent_candidates, top_k=3)
                        ep_cands = [r.get("candidate", "") for r in ep_rows if r.get("candidate")]
                        ep_ctx = get_entity_contexts(ep_cands, h_ids, r_ids, t_ids, ents, rels)
                        ep_cands_with_ctx = [(n, ep_ctx.get(n, "")) for n in ep_cands]
                        best = await llm_resolve_entity(session, question, ep_name, ep_cands_with_ctx)
                        idx = ents.index(best) if best and best in ents else None
                        if idx is not None and ep_hop:
                            breakpoints[ep_hop] = idx
                        entity_retrieval_details.append({
                            "role": f"endpoint_step{ep_hop}",
                            "query": ep_name,
                            "selected": best,
                            "selected_idx": idx,
                            "llm_resolved": True,
                        })
                ner_ok = True
            else:
                # Fallback to old parser
                anchor_eq_name_ner, _, steps, _ = parse_decomposition(raw)
                start_name = _parse_ner_start(raw)
                if not start_name and anchor_eq_name_ner:
                    start_name = anchor_eq_name_ner
                if steps:
                    if start_name:
                        sn = normalize(start_name)
                        for s in ner_scored:
                            if normalize(s["entity"]) == sn:
                                anchor_idx = ner_name_to_ids_expanded.get(s["entity"], [None])[0]
                                anchor_name = s["entity"]
                                break
                    if anchor_idx is None and ner_scored:
                        anchor_name = ner_scored[0]["entity"]
                        anchor_idx = ner_name_to_ids_expanded.get(anchor_name, [None])[0]
                    for step in steps:
                        step["entity_query"] = None
                    breakpoints = {}
                    ner_ok = True

        # ── Original decomposition-based anchor resolution (fallback) ──
        if not ner_ok:
            decomp_question = f"Question: {question}"
            if retry_note:
                decomp_question += f"\n\nRetry instruction: {retry_note}"
            if anchor_forbidden:
                decomp_question += f"\nDo not use this previous anchor again: {anchor_forbidden}"

            raw = await call_llm(session, [
                {"role": "system", "content": DECOMP_PROMPT},
                {"role": "user", "content": decomp_question},
            ])
            anchor_eq_name, anchor_eq, steps, _ = parse_decomposition(raw)
            if not steps:
                return {"error": "decomposition failed", "raw": raw, "decomp_question": decomp_question}

            if anchor_eq:
                rows = await gte_retrieve(session, anchor_eq, ent_candidates, top_k=5)
                topk = [{"rank": i+1, "candidate": r.get("candidate", ""), "score": round(r.get("score", 0), 4)} for i, r in enumerate(rows)]
                anchor_cands = [r["candidate"] for r in topk if r["candidate"]]
                anchor_ctx = get_entity_contexts(anchor_cands, h_ids, r_ids, t_ids, ents, rels)
                anchor_cands_with_ctx = [(n, anchor_ctx.get(n, "")) for n in anchor_cands]
                selected = await llm_resolve_entity(session, question, anchor_eq, anchor_cands_with_ctx)
                anchor_idx = ents.index(selected) if selected and selected in ents else None
                anchor_name = selected or anchor_eq_name
                entity_retrieval_details.append({
                    "role": "anchor",
                    "query": anchor_eq,
                    "top_k": topk,
                    "selected": selected,
                    "selected_idx": anchor_idx,
                    "llm_resolved": True,
                })

            # Endpoint resolve for original format
            breakpoints = {}
            for step in steps:
                if step["endpoint"] and step.get("endpoint_query"):
                    ep_rows = await gte_retrieve(session, step["endpoint_query"], ent_candidates, top_k=3)
                    ep_topk = [{"rank": i+1, "candidate": r.get("candidate", ""), "score": round(r.get("score", 0), 4)} for i, r in enumerate(ep_rows)]
                    ep_cands = [r["candidate"] for r in ep_topk if r["candidate"]]
                    ep_ctx = get_entity_contexts(ep_cands, h_ids, r_ids, t_ids, ents, rels)
                    ep_cands_with_ctx = [(n, ep_ctx.get(n, "")) for n in ep_cands]
                    best = await llm_resolve_entity(session, question, step["endpoint_query"], ep_cands_with_ctx)
                    idx = ents.index(best) if best and best in ents else None
                    if idx is not None:
                        breakpoints[step["step"]] = idx
                    entity_retrieval_details.append({
                        "role": f"endpoint_step{step['step']}",
                        "query": step["endpoint_query"],
                        "top_k": ep_topk,
                        "selected": best,
                        "selected_idx": idx,
                        "llm_resolved": True,
                    })

        # ── Shared: multi-query GTE (definition + subquestion + question + keyword) + prune + expand ──
        step_candidates = {}
        gte_per_step = {}
        relation_retrieval_details = []
        for step in steps:
            # Build deduplicated queries: definition, subquestion, question, keyword, relation_query
            _seen_q = set()
            queries = []
            for field in ["definition", "subquestion", "question", "keyword", "relation_query"]:
                val = step.get(field, "")
                if val and val.strip():
                    ql = val.strip().lower()
                    if ql not in _seen_q:
                        _seen_q.add(ql)
                        queries.append(val.strip())
            gte_all = {}
            queries_detail = []
            for query in queries:
                rows = await gte_retrieve(session, query, rels, candidate_texts=rel_texts, top_k=10)
                topk = []
                for i, r in enumerate(rows):
                    cand = r.get("candidate", "")
                    score = round(r.get("score", 0), 4)
                    idx_in_rels = rels.index(cand) if cand in rels else None
                    topk.append({"rank": i+1, "candidate": cand, "score": score, "rel_idx": idx_in_rels,
                                 "rel_text": rel_to_text(cand) if idx_in_rels is not None else ""})
                    if idx_in_rels is not None:
                        if idx_in_rels not in gte_all or score > gte_all[idx_in_rels][1]:
                            gte_all[idx_in_rels] = (rels[idx_in_rels], score)
                queries_detail.append({"query": query, "top_k": topk})

            gte_candidates = sorted(gte_all.items(), key=lambda x: -x[1][1])
            candidate_list = [(idx, name, score) for idx, (name, score) in gte_candidates]
            step_candidates[step["step"]] = candidate_list
            gte_per_step[step["step"]] = gte_all
            relation_retrieval_details.append({
                "step": step["step"],
                "queries": queries_detail,
                "gte_candidates_count": len(gte_all),
                "gte_indices": sorted(gte_all.keys()),
            })

        prune_result, prune_debug = await llm_prune_all_relations(session, question, steps, step_candidates)

        step_relations = []
        for step in steps:
            sn = step["step"]
            pruned = prune_result.get(sn, set())
            step_relations.append(pruned)
            gte_all = gte_per_step.get(sn, {})
            resolved_names = [{"idx": ri, "name": gte_all[ri][0]} for ri in sorted(pruned)] if pruned else []
            for rd in relation_retrieval_details:
                if rd["step"] == sn:
                    rd["resolved_indices"] = sorted(pruned)
                    rd["resolved_names"] = resolved_names
                    break

        if step_relations_override:
            for layer_idx, override_set in step_relations_override.items():
                if 0 <= layer_idx < len(step_relations):
                    step_relations[layer_idx] = set(override_set)
                    for rd in relation_retrieval_details:
                        if rd["step"] == steps[layer_idx]["step"]:
                            rd["override_indices"] = sorted(step_relations[layer_idx])
                            rd["override_names"] = [{"idx": ri, "name": rels[ri]} for ri in sorted(step_relations[layer_idx]) if 0 <= ri < len(rels)]
                            break

        prune_debug_field = {
            "prompt": prune_debug["prompt"],
            "response": prune_debug["response"],
            "parsed_yaml": prune_debug.get("parsed_yaml"),
        }

        paths, max_depth, max_cov = [], 0, 0
        answer_candidates = []
        all_subgraph_nodes = set()
        if anchor_idx is not None:
            bp_set = set(breakpoints.values()) - {anchor_idx, None}
            paths, max_depth, max_cov = relation_prior_expand(
                anchor_idx, step_relations, h_ids, r_ids, t_ids, ents,
                explicit_targets=bp_set if bp_set else None)

            # Prefer paths hitting breakpoint endpoints
            paths = prefer_breakpoint_hit_paths(paths, breakpoints, h_ids, r_ids, t_ids, ents)
            if paths:
                max_depth = max(p.get("depth", 0) for p in paths)
                max_cov = max(len(p.get("covered_steps", frozenset())) for p in paths)

            # Collect path nodes
            all_subgraph_nodes = {anchor_idx}
            for path in paths:
                all_subgraph_nodes.update(path["nodes"])

            # HR frontier: path-level (h+r) forward + (r+t) reverse triples
            hr_triples, hr_nodes = _collect_hr_frontier(
                anchor_idx, step_relations, h_ids, r_ids, t_ids,
                paths=paths)
            all_subgraph_nodes |= hr_nodes

            # Extract answer candidates from last 2 hops of each path (+ CVT expansion)
            last2_nodes = set()
            for path in paths:
                nodes = path.get("nodes", [])
                for n in nodes[-2:]:
                    if n != anchor_idx:
                        last2_nodes.add(n)
            for node_idx in sorted(last2_nodes):
                name = ents[node_idx] if 0 <= node_idx < len(ents) else ""
                if is_cvt_like(name):
                    for cvt_idx, _ in expand_through_cvt(node_idx, h_ids, r_ids, t_ids, ents):
                        if cvt_idx != anchor_idx and 0 <= cvt_idx < len(ents) and not is_cvt_like(ents[cvt_idx]):
                            answer_candidates.append(ents[cvt_idx])
                else:
                    answer_candidates.append(name)

            seen = set()
            unique = []
            for c in answer_candidates:
                nc = normalize(c)
                if nc not in seen:
                    seen.add(nc)
                    unique.append(c)
            answer_candidates = unique

        # GT recall from path entities only (not HR frontier)
        path_candidates = _extract_path_candidates(paths, anchor_idx, ents, h_ids, r_ids, t_ids)
        gt_hit = candidate_hit(path_candidates, gt_answers) if path_candidates else False
        return {
            "error": None,
            "raw": raw,
            "decomp_question": decomp_question,
            "steps": steps,
            "anchor_idx": anchor_idx,
            "anchor_name": ents[anchor_idx] if anchor_idx is not None else anchor_name,
            "breakpoints": breakpoints,
            "step_relations_sets": step_relations,
            "entity_retrieval_details": entity_retrieval_details,
            "relation_retrieval_details": relation_retrieval_details,
            "prune_debug": prune_debug_field,
            "step_candidates": step_candidates,
            "paths": paths,
            "max_depth": max_depth,
            "max_cov": max_cov,
            "answer_candidates": answer_candidates,
            "gt_hit": gt_hit,
        }

    def _attempt_score(state):
        if state.get("error"):
            return (-1, -1, -1, -1)
        n_cand = len(state.get("answer_candidates", []))
        under_threshold = 1 if n_cand <= CANDIDATE_THRESHOLD else 0
        return (
            state.get("max_cov", 0),
            under_threshold,
            len(state.get("paths", [])),
            -n_cand,
        )

    planning_attempts = []
    _t0 = time.perf_counter()
    active = await execute_planning()
    _stage_times["planning_primary"] = time.perf_counter() - _t0
    planning_attempts.append({"label": "primary", "score": _attempt_score(active), "anchor": active.get("anchor_name")})

    if active.get("error"):
        return {
            "case_id": pilot_row["case_id"],
            "question": question,
            "error": active["error"],
            "gt_answers": gt_answers,
            "answer_candidates": [],
            "gt_hit": False,
            "raw": active.get("raw"),
        }

    nonempty_layers = [i for i, rs in enumerate(active["step_relations_sets"]) if rs]
    _t0 = time.perf_counter()
    layer_diagnostics = diagnose_layers(
        active["anchor_idx"], active["step_relations_sets"], h_ids, r_ids, t_ids, ents,
        max_hops=3,
    ) if active.get("anchor_idx") is not None else []
    _stage_times["diagnose_layers"] = time.perf_counter() - _t0

    # ── Anchor shortcut: skip noisy layers using already-recorded anchor_hit ──
    # BFS recorded paths reaching each layer from anchor (anchor_hit) and from
    # sequential frontier (frontier_hit). If a layer misses frontier but hits
    # anchor, the previous layer is noise → clear it (no LLM call needed).
    # Re-diagnose after shortcuts to update subsequent layers' frontier info.
    _t0_retry = time.perf_counter()
    shortcut_applied = False
    for li in list(nonempty_layers):
        if li < len(layer_diagnostics) and li > 0:
            d = layer_diagnostics[li]
            if not d["frontier_hit"] and d["anchor_hit"]:
                # Previous layer is noise — clear it
                if active["step_relations_sets"][li - 1]:
                    active["step_relations_sets"][li - 1] = set()
                    shortcut_applied = True

    if shortcut_applied:
        layer_diagnostics = diagnose_layers(
            active["anchor_idx"], active["step_relations_sets"], h_ids, r_ids, t_ids, ents,
            max_hops=3,
        ) if active.get("anchor_idx") is not None else []
        planning_attempts.append({"label": "anchor_shortcut", "score": _attempt_score(active), "anchor": active.get("anchor_name")})

    # Find first truly unreachable layer (both frontier AND anchor miss)
    first_miss = None
    for li in nonempty_layers:
        if li < len(layer_diagnostics):
            d = layer_diagnostics[li]
            if not d["frontier_hit"] and not d["anchor_hit"]:
                first_miss = li
                break

    # Retry B: reselect relations for truly unreachable layer (LLM call)
    if first_miss is not None:
        reselection = await llm_reselect_single_step_relation(
            session, question, active["steps"][first_miss],
            active["step_candidates"], active["step_relations_sets"][first_miss],
        )
        overrides = {first_miss: reselection}
        alt = await execute_planning(
            step_relations_override=overrides,
            retry_note=f"Retry relation selection for step {first_miss + 1}. The previous relation choice was too weak or too noisy.",
        )
        planning_attempts.append({"label": "reselect_failed_layer", "score": _attempt_score(alt), "anchor": alt.get("anchor_name")})
        if _attempt_score(alt) > _attempt_score(active):
            active = alt

    # Retry C: change anchor if still weak
    active_nonempty = [i for i, rs in enumerate(active["step_relations_sets"]) if rs]
    if active_nonempty and active.get("max_cov", 0) < len(active_nonempty):
        explicit_entities = {active.get("anchor_name")} | set(ents[idx] for idx in active.get("breakpoints", {}).values() if idx is not None and 0 <= idx < len(ents))
        explicit_entities = {e for e in explicit_entities if e}
        if len(explicit_entities) >= 2 and active.get("anchor_name"):
            alt = await execute_planning(
                anchor_forbidden=active["anchor_name"],
                retry_note="Choose a different explicit anchor entity from the question and redecompose the plan from that anchor.",
            )
            planning_attempts.append({"label": "anchor_swap_redecompose", "score": _attempt_score(alt), "anchor": alt.get("anchor_name")})
            if _attempt_score(alt) > _attempt_score(active):
                active = alt

    final_layer_diagnostics = diagnose_layers(
        active["anchor_idx"], active["step_relations_sets"], h_ids, r_ids, t_ids, ents,
        max_hops=3,
    ) if active.get("anchor_idx") is not None else []
    _stage_times["retries"] = time.perf_counter() - _t0_retry

    raw = active["raw"]
    decomp_question = active["decomp_question"]
    steps = active["steps"]
    anchor_idx = active["anchor_idx"]
    breakpoints = active["breakpoints"]
    step_relations = active["step_relations_sets"]
    entity_retrieval_details = active["entity_retrieval_details"]
    relation_retrieval_details = active["relation_retrieval_details"]
    prune_debug_field = active["prune_debug"]
    paths = active["paths"]
    max_depth = active["max_depth"]
    answer_candidates = active["answer_candidates"]
    gt_hit = active["gt_hit"]

    # Step 5: Compress paths into logical patterns
    breakpoint_indices = set(breakpoints.values())
    logical_paths = compress_paths(paths, ents, rels, anchor_idx, breakpoint_indices) if paths else []

    # Step 6: Multi-attempt LLM reasoning with rollback
    # Model selects paths, can trigger rollback to get more paths (up to 3 attempts)
    # After all attempts, model reasons over all selected paths' triples
    llm_answer = None
    llm_hit = False
    selected_paths = []
    num_triples = 0
    attempt_log = []
    llm_reasoning_prompt = None
    llm_reasoning_full = None
    _t0_pathsel = time.perf_counter() if logical_paths else None
    if logical_paths:
        try:
            remaining = list(range(len(logical_paths)))

            for attempt in range(3):
                if not remaining:
                    break
                # Present remaining patterns with sequential numbering
                path_lines = []
                idx_map = {}  # display_number -> actual index in logical_paths
                for display_num, actual_idx in enumerate(remaining[:15], 1):
                    path_lines.append(f"{display_num}. {logical_paths[actual_idx]['readable']}")
                    idx_map[display_num] = actual_idx
                paths_text = "\n".join(path_lines)

                select_prompt = f"""Analyze and select reasoning paths for this question.

Question: {question}

Paths:
{paths_text}

Instructions:
1. Ignore hidden entity identities. Judge each path only by its relation sequence and node structure.
2. Select paths by semantic relevance, not by shortest length.
3. Keep a path if it preserves the intended multi-step meaning of the question, even if it is longer, includes bridge nodes, or is not the most direct-looking path.
4. Do not eliminate a path only because it is longer, slightly noisy, or contains extra intermediate structure.
5. When uncertain, prefer recall over precision: keep all paths that are semantically plausible.
6. Remove only paths that clearly contradict the question semantics.

Output format:
<analysis>
Two short sentences max. Do NOT deliberate. Just state which paths fit and why.
</analysis>
<selected>comma-separated path indices only</selected>
<need_more>yes or no</need_more>

Rules:
- Do not copy example numbers.
- Do not leave out a semantically plausible path just because it is longer.
- If several paths are plausible, select all of them."""

                # Try up to 3 times for valid XML output
                sel_raw = ""
                for sel_attempt in range(3):
                    sel_raw = await call_llm(session, [
                        {"role": "system", "content": "You analyze and select reasoning paths for multi-step QA. Your goal is high-recall semantic path selection. Judge semantic fit of relation chains to the question. Keep all semantically plausible paths. Output exactly three XML tags: <analysis>, <selected>, and <need_more>."},
                        {"role": "user", "content": select_prompt},
                    ], max_tokens=600)
                    if extract_xml_tag(sel_raw, "selected"):
                        break

                # Parse selected indices from <selected> tag
                sel_indices = []
                sel_text = extract_xml_tag(sel_raw, "selected") or ""
                for m in re.finditer(r'\d+', sel_text):
                    display_num = int(m.group())
                    if display_num in idx_map:
                        sel_indices.append(idx_map[display_num])

                # Parse rollback flag from <need_more> tag
                need_more_text = extract_xml_tag(sel_raw, "need_more") or "no"
                rollback = "yes" in need_more_text.lower()

                selected_paths.extend(sel_indices)
                attempt_log.append({
                    "attempt": attempt + 1,
                    "selected": sel_indices,
                    "rollback": rollback,
                    "raw": sel_raw.strip(),
                    "prompt": select_prompt,
                    "full_response": sel_raw,
                })

                # Remove selected from remaining pool
                selected_set = set(sel_indices)
                remaining = [i for i in remaining if i not in selected_set]

                # Stop if no rollback or max attempts reached
                if not rollback or attempt >= 2:
                    break

            # Deduplicate selected paths
            selected_paths = list(dict.fromkeys(selected_paths))
            # Fallback: if nothing selected, use top 3 by candidate count
            if not selected_paths:
                selected_paths = list(range(min(3, len(logical_paths))))

            _stage_times["path_select"] = time.perf_counter() - _t0_pathsel

            # Build final reasoning subgraph from selected patterns, grouped by pattern.
            selected_pattern_objs = []
            for idx in selected_paths:
                lp = logical_paths[idx]
                selected_pattern_objs.append(lp)
            selected_pattern_objs.extend(build_endpoint_rescue_patterns(
                paths, selected_pattern_objs, ents, rels, anchor_idx, breakpoint_indices,
            ))
            pat_evidence = build_pattern_evidence_triples(
                selected_pattern_objs, ents, rels, h_ids, r_ids, t_ids, anchor_idx,
                max_grouped_lines=120,
            )
            num_triples = sum(len(pe.triples) for pe in pat_evidence.values())

            if pat_evidence:
                pattern_text = format_pattern_evidence(pat_evidence)

                reason_prompt = f"""QUESTION: {question}

GRAPH EVIDENCE (from Freebase, snapshot circa 2015):
{pattern_text}

━━━ REASONING TASK ━━━

Use the graph evidence to answer the question.
Reason step by step, but keep each step concise: 1 to 2 sentences.

STEP 1 — QUESTION UNDERSTANDING
Explain what the question is asking for, what the answer type is, and what constraint(s) must be satisfied.
Also state whether the last hop is the answer itself or only a verification condition.

STEP 2 — PATTERN COMPARISON
You must evaluate at least two candidate patterns if more than one is available.
For each pattern, say MATCH or MISMATCH and give one short reason why its relation chain semantically matches or mismatches the question.
Then choose the best matching pattern and briefly justify why it is better than the others.

STEP 3 — ANSWER POSITION AND CANDIDATES
Explain where the answer is located in the selected pattern (which hop/node).
List the candidate entities at that position, and mention the graph evidence connecting them.

STEP 4 — CONSTRAINT VERIFICATION AND ANSWER SELECTION

4a — IDENTIFY CONSTRAINTS (explicit AND implicit)

Determine the expected ANSWER TYPE from the question (person, country, language, event, year, etc.).
Identify ALL constraints:

EXPLICIT constraints: directly stated in the question (dates, locations, superlatives, quantities, conditions).

IMPLICIT constraints — apply these heuristics based on question semantics:
- UNIQUE ROLE/POSITION ("the governor", "the president", "the leader", "the capital") WITHOUT a time qualifier → implicit: prefer the MOST RECENT or CURRENT holder
- EVENTS/ACHIEVEMENTS/AWARDS ("wins", "championships", "movies", "albums", "titles") → NO implicit "current" constraint; return ALL matching instances
- ATTRIBUTES/PROPERTIES ("languages spoken", "religions practiced", "government type", "currency") → return ALL that apply
- GROUP MEMBERSHIP ("countries in X", "states bisected by Y", "members of") → return ALL matching members

CRITICAL: The singular/plural form of the question ALONE does not determine answer count. Use the heuristics above.

4b — CONSTRAINT CHECK
For each candidate, verify:
1. TYPE MATCH: Does this candidate match the expected ANSWER TYPE? Discard non-matching types (e.g., discard countries, dates, roles when question asks for languages).
2. EXPLICIT constraints: Does it satisfy all stated conditions?
3. IMPLICIT constraints: Does it satisfy the applicable heuristic?
  - Constraint: [description] → candidate A: PASS/FAIL, candidate B: PASS/FAIL, ...
Collect ALL candidates that pass ALL constraints → PASSING SET.
If graph evidence does NOT show a candidate fails, KEEP it.

4c — OUTPUT DECISION
- PASSING SET has 1 member → output it.
- PASSING SET has multiple + implicit "most recent" applies → output the MOST RECENT member by graph dates. If no dates in evidence, output ALL.
- PASSING SET has multiple + no implicit limit → output ALL members.
- NEVER discard a candidate solely because the question uses singular phrasing.

RULES:
- NEVER decide answer count before constraint checking. Evaluate ALL candidates first.
- Prefer graph evidence over intuition. NEVER use external knowledge to filter.
- The answer may appear at an intermediate hop, not necessarily the terminal node.
- For geographic constraints, only remove if graph evidence explicitly contradicts.
- For temporal constraints, look for date values in evidence. No dates → do NOT filter.
- "When" questions → output event NAME (e.g. "2014 World Series"), NOT raw timestamp.
- Answer MUST be an exact entity string from GRAPH EVIDENCE or CANDIDATE ENTITIES. Never output a bare number, year, or timestamp — use the full entity name.
- When in doubt, output ALL passing candidates. Over-output is better than discarding valid answers.

━━━ EXAMPLES (illustrate METHOD only — do NOT copy answers) ━━━

Example A — TYPE FILTER + IMPLICIT "CURRENT" for unique role:
Q: "Who is the president of France?"
Candidates: [Emmanuel Macron, François Hollande, France, President, 2017-05-14]
→ Answer type: person (president). Type filter: FAIL=[France(country), President(role), 2017-05-14(date)]
→ Remaining: [Emmanuel Macron, François Hollande]. Implicit: unique role no time qualifier → most recent
→ Output: Emmanuel Macron

Example B — EVENTS/ACHIEVEMENTS → return ALL:
Q: "What movies did the actor who played Forrest Gump star in?"
Candidates: [Forrest Gump, Tom Hanks, Saving Private Ryan, Cast Away, Actor, 1994]
→ Answer type: movie. Type filter: FAIL=[Tom Hanks(person), Actor(role), 1994(year)]
→ Remaining: [Forrest Gump, Saving Private Ryan, Cast Away]. Implicit: events/works → ALL
→ Output: Forrest Gump | Saving Private Ryan | Cast Away

Example C — PATTERN SELECTION:
Q: "What countries border France?"
Pattern A: location.location.adjoining_countries (direct, 1-hop)
Pattern B: location.location.containedby → location.location.adjoining_countries (via region, 2-hop)
→ Choose Pattern A: direct semantic match, shorter path, fewer noise entities in candidates

━━━ OUTPUT FORMAT ━━━
<reasoning>
Step 1: 1-2 sentences.
Step 2: 1-2 sentences per pattern evaluated, then 1 sentence for choice.
Step 3: 1-2 sentences.
Step 4a: Answer type + explicit/implicit constraints. Step 4b: TYPE MATCH and PASS/FAIL per candidate. Step 4c: PASSING SET and output decision.
</reasoning>
<answer>\\boxed{{exact entity}}</answer>

Multiple answers: <answer>\\boxed{{cand1}} \\boxed{{cand2}}</answer>
NO text after </answer> tag."""

                _t0_reason = time.perf_counter()
                llm_raw = await call_llm(session, [
                    {"role": "system", "content": "You are a precise graph QA system. Follow the 4-step reasoning process. ALWAYS answer. Compare patterns before choosing. First identify answer type and ALL constraints (explicit + implicit), then filter candidates by type match and constraint check. Apply implicit constraint heuristics: unique role → most recent; events/achievements → all; attributes → all; group membership → all. Singular/plural alone does NOT determine answer count. NEVER skip constraint verification. Exact graph strings only."},
                    {"role": "user", "content": reason_prompt},
                ], max_tokens=1800)
                # Extract answer from <answer>...\boxed{...}...</answer> format
                ans_match = re.search(r'<answer>(.*?)</answer>', llm_raw, re.DOTALL)
                if ans_match:
                    boxed = re.findall(r'\\boxed\{([^}]+)\}', ans_match.group(1))
                    if boxed:
                        llm_answer = " | ".join(b.strip() for b in boxed)
                        llm_hit = candidate_hit([b.strip() for b in boxed], gt_answers)
                    else:
                        llm_answer = ans_match.group(1).strip()
                        llm_hit = candidate_hit([llm_answer], gt_answers)
                else:
                    # Legacy fallback
                    ans_match = re.search(r'ANSWER:\s*(.+)', llm_raw, re.IGNORECASE)
                    if ans_match:
                        llm_answer = ans_match.group(1).strip()
                    else:
                        lines = [l.strip() for l in llm_raw.strip().split('\n') if l.strip()]
                        llm_answer = lines[-1] if lines else llm_raw.strip()
                    llm_hit = candidate_hit([llm_answer], gt_answers)
                llm_reasoning_prompt = reason_prompt
                llm_reasoning_full = llm_raw
                _stage_times["llm_reasoning"] = time.perf_counter() - _t0_reason
        except Exception as e:
            print(f"  LLM reasoning error: {type(e).__name__}: {e}")
            llm_answer = None
            llm_hit = False

    return {
        "case_id": pilot_row["case_id"],
        "question": question,
        "gt_answers": gt_answers,
        "decomposition_prompt": DECOMP_PROMPT,
        "decomposition_question": decomp_question,
        "decomposition": raw,
        "steps_parsed": steps,
        "anchor_idx": anchor_idx,
        "anchor_name": ents[anchor_idx] if anchor_idx is not None else None,
        "breakpoints": {k: ents[v] for k, v in breakpoints.items()},
        "step_relations": [list(r) for r in step_relations],
        "entity_retrieval_details": entity_retrieval_details,
        "relation_retrieval_details": relation_retrieval_details,
        "prune_debug": prune_debug_field,
        "layer_diagnostics": final_layer_diagnostics,
        "planning_attempts": planning_attempts,
        "max_depth": max_depth,
        "num_paths": len(paths),
        "answer_candidates": answer_candidates,
        "gt_hit": gt_hit,
        "num_patterns": len(logical_paths),
        "logical_paths": [lp["readable"] for lp in logical_paths[:10]],
        "pattern_details": [{
            "candidates": lp.get("candidates", []),
            "best_tier": lp.get("best_tier", (0, -1, 0)),
            "endpoint": lp.get("endpoint"),
            "witness_nodes": lp.get("best_raw_path", {}).get("nodes", []),
            "witness_relations": lp.get("best_raw_path", {}).get("relations", []),
        } for lp in logical_paths],
        "llm_answer": llm_answer,
        "llm_hit": llm_hit,
        "selected_paths": selected_paths,
        "num_triples": num_triples,
        "attempt_log": attempt_log,
        "llm_reasoning_prompt": llm_reasoning_prompt,
        "llm_reasoning_full": llm_reasoning_full,
        "stage_times": _stage_times,
    }


async def _run_case_wrapper(session, sample, pilot_row, sem, case_num, total_cases):
    """Wrapper to run a single case with semaphore control and error handling."""
    import time
    t0 = time.perf_counter()
    case_id = pilot_row.get('case_id', '?')

    # Use semaphore if provided (parallel mode), otherwise run directly
    if sem is not None:
        async with sem:
            return await _execute_case(session, sample, pilot_row, case_num, total_cases, t0)
    else:
        return await _execute_case(session, sample, pilot_row, case_num, total_cases, t0)


async def _execute_case(session, sample, pilot_row, case_num, total_cases, t0):
    """Execute a single case and return result with timing."""
    case_id = pilot_row.get('case_id', '?')

    try:
        result = await run_case(session, sample, pilot_row)
        dt = time.perf_counter() - t0

        if result is None:
            return None, dt, case_id, case_num

        status = "HIT" if result["gt_hit"] else "MISS"
        llm_status = "LLM_HIT" if result.get("llm_hit") else "LLM_MISS"
        n_patterns = result.get("num_patterns", 0)
        n_triples = result.get("num_triples", 0)
        cid = result.get('case_id', '?') or '?'

        # Print result as it completes
        print(f"[{case_num}/{total_cases}] {status} | {cid[:30]} | paths={result.get('num_paths',0):6d} | patterns={n_patterns:3d} | triples={n_triples:3d} | {llm_status} | time={dt:.2f}s | llm={str(result.get('llm_answer',''))[:25]} | gt={result.get('gt_answers')}")
        # Print stage timing
        st = result.get("stage_times", {})
        if st:
            parts = " | ".join(f"{k}={v*1000:.0f}ms" for k, v in st.items())
            print(f"  stages: {parts}")

        return result, dt, case_id, case_num

    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"[{case_num}/{total_cases}] ERROR: {case_id}: {type(e).__name__}: {e} (time={dt:.2f}s)")
        return None, dt, case_id, case_num


async def amain():
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-results", default=str(DEFAULT_PILOT))
    parser.add_argument("--cwq-pkl", default=str(DEFAULT_CWQ))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel cases to run (default: 1, max: 32)")
    parser.add_argument("--mode", choices=["case", "stage"], default="case",
                        help="Execution mode: 'case' for per-case parallel, 'stage' for stage-based batch")
    parser.add_argument("--gt-anchor", action="store_true",
                        help="Use ground-truth core_entities from CWQ as anchor (skip NER)")
    parser.add_argument("--prune", choices=["llm", "rerank"], default="llm",
                        help="Relation pruning method: 'llm' (default) or 'rerank' (Qwen3-Reranker)")
    parser.add_argument("--prune-top-k", type=int, default=5,
                        help="Number of relations to keep per step after pruning (default: 5)")
    parser.add_argument("--rerank-model", default="/zhaoshu/llm/Qwen/Qwen3-Reranker-0.6B",
                        help="Reranker model path (Qwen3-Reranker-0.6B)")
    parser.add_argument("--reason-style", choices=["default", "check", "ecot", "entity", "entity-lite"], default="default",
                        help="Stage 8 prompt style: 'default', 'check' (checklist), 'ecot' (evidence-COT), or 'entity' (entity-centric 3-step)")
    parser.add_argument("--inject-decomp", default=None,
                        help="Path to golden results JSON. Injects Stage 0/1/1.5 outputs from golden, skipping LLM calls for those stages.")
    args = parser.parse_args()
    global REASON_STYLE
    REASON_STYLE = args.reason_style

    pilot_rows = json.loads(Path(args.pilot_results).read_text())
    samples = pickle.loads(Path(args.cwq_pkl).read_bytes())
    sample_map = {}
    for s in samples:
        if "id" in s:
            sample_map[s["id"]] = s
        elif "question_id" in s:
            sample_map[s["question_id"]] = s

    # Load mask: exclude truly wrong-type (答非所问) cases
    masked_ids = set()
    if MASK_WRONG_TYPE.exists():
        masked_ids = set(json.loads(MASK_WRONG_TYPE.read_text()))
        print(f"  Masked {len(masked_ids)} wrong-type cases")

    # Prepare cases
    cases_to_run = []
    for idx, pilot_row in enumerate(pilot_rows[:args.limit], 1):
        sample = sample_map.get(pilot_row["case_id"])
        if not sample:
            for s in samples:
                if s.get("question", "") == pilot_row["question"]:
                    sample = s; break
        if not sample:
            print(f"SKIP: {pilot_row['case_id']}")
            continue
        if pilot_row["case_id"] in masked_ids:
            continue
        cases_to_run.append((sample, pilot_row, idx))

    total_cases = len(cases_to_run)

    if args.mode == "stage":
        await _run_stage_mode(cases_to_run, args)
        return

    # ── Original case mode ──
    parallel = max(1, min(args.parallel, 32))
    print(f"\n=== Starting {total_cases} cases with parallelism={parallel} ===")

    rows = []
    case_times = []
    wall_start = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        if parallel == 1:
            # Sequential execution (original behavior)
            for sample, pilot_row, idx in cases_to_run:
                result, dt, case_id, case_num = await _run_case_wrapper(
                    session, sample, pilot_row, None, idx, total_cases
                )
                if result is not None:
                    rows.append(result)
                if dt is not None:
                    case_times.append(dt)
        else:
            # Parallel execution with semaphore
            sem = asyncio.Semaphore(parallel)
            tasks = [
                _run_case_wrapper(session, sample, pilot_row, sem, idx, total_cases)
                for sample, pilot_row, idx in cases_to_run
            ]
            completed = await asyncio.gather(*tasks)

            for result, dt, case_id, case_num in completed:
                if result is not None:
                    rows.append(result)
                if dt is not None:
                    case_times.append(dt)

    wall_time = time.perf_counter() - wall_start

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    cases = len(rows)
    if cases == 0:
        print("\n=== No valid cases processed ===")
        return

    hits = sum(1 for r in rows if r["gt_hit"])
    llm_hits = sum(1 for r in rows if r.get("llm_hit"))
    avg_paths = sum(r.get("num_paths", 0) for r in rows) / cases
    avg_patterns = sum(r.get("num_patterns", 0) for r in rows) / cases

    # Timing statistics
    avg_case_time = sum(case_times) / len(case_times) if case_times else 0
    total_case_time = sum(case_times)

    print(f"\n=== Summary: GT_recall={hits}/{cases} ({100*hits/cases:.1f}%) | LLM_reason={llm_hits}/{cases} ({100*llm_hits/cases:.1f}%) ===")
    print(f"    Avg paths: {avg_paths:.0f} | Avg compressed patterns: {avg_patterns:.1f}")
    print(f"    Timing: Wall={wall_time:.2f}s | Total case time={total_case_time:.2f}s | Avg case={avg_case_time:.2f}s")
    if parallel > 1:
        speedup = total_case_time / wall_time if wall_time > 0 else 1.0
        print(f"    Parallelism: {parallel} | Speedup: {speedup:.2f}x")


# ══════════════════════════════════════════════════════════════════════════════
# Stage-based batch execution functions
# ══════════════════════════════════════════════════════════════════════════════

CANDIDATE_THRESHOLD = 50


def _attempt_score(state):
    """Score a planning attempt for comparison. Penalizes candidate explosion."""
    if state.get("error"):
        return (-1, -1, -1, -1)
    n_cand = len(state.get("answer_candidates", []))
    under_threshold = 1 if n_cand <= CANDIDATE_THRESHOLD else 0
    return (
        state.get("max_cov", 0),
        under_threshold,
        len(state.get("paths", [])),
        -n_cand,
    )


def evaluate_step_relations(anchor_idx, paths, step_relations, ents):
    """Count unique non-anchor candidates per relation per step from traversal paths.

    Returns: dict {step_idx: {rel_idx: count_of_unique_candidates}}
    """
    rel_to_steps = {}
    for si, rs in enumerate(step_relations):
        for r in (rs if isinstance(rs, (set, list)) else set()):
            rel_to_steps.setdefault(r, set()).add(si)

    step_rel_targets = {}
    for path in paths:
        nodes = path.get("nodes", [])
        relations = path.get("relations", [])
        for hop_i, rel_idx in enumerate(relations):
            if hop_i + 1 < len(nodes):
                target_idx = nodes[hop_i + 1]
                if target_idx == anchor_idx:
                    continue
                steps_for_rel = rel_to_steps.get(rel_idx, set())
                for step_idx in steps_for_rel:
                    step_rel_targets.setdefault(step_idx, {}).setdefault(rel_idx, set()).add(target_idx)

    return {s: {r: len(t) for r, t in rels.items()} for s, rels in step_rel_targets.items()}


async def stage_0_ner_resolve(session, cases: List[CaseState]):
    """GTE-only NER entity resolution. No LLM calls."""
    _t0 = time.perf_counter()

    async def _resolve_one(cs: CaseState):
        cs.ents = cs.sample.get("text_entity_list", []) + cs.sample.get("non_text_entity_list", [])
        cs.rels = list(cs.sample.get("relation_list", []))
        cs.h_ids = cs.sample.get("h_id_list", [])
        cs.r_ids = cs.sample.get("r_id_list", [])
        cs.t_ids = cs.sample.get("t_id_list", [])

        cs.ner_scored, _ = await resolve_anchor_ner(
            session, cs.question, cs.ents, cs.rels, cs.h_ids, cs.r_ids, cs.t_ids)
        cs.ner_top_ents = []
        _seen = set()
        for s in cs.ner_scored[:6]:
            if s["entity"] not in _seen:
                cs.ner_top_ents.append((s["entity"], s["gte"]))
                _seen.add(s["entity"])

        # CVT expansion
        cs.ents, cs.rels, cs.h_ids, cs.r_ids, cs.t_ids = expand_cvt_leaves(
            cs.ents, cs.rels, cs.h_ids, cs.r_ids, cs.t_ids)
        cs.rel_texts = list(cs.rels)
        cs.ent_candidates = [e for e in cs.ents if e and len(e) > 1 and not is_cvt_like(e)]

        cs.ner_name_to_ids_expanded = {}
        for i, name in enumerate(cs.ents):
            cs.ner_name_to_ids_expanded.setdefault(name, []).append(i)

    # Limit concurrent GTE calls to avoid overloading server
    _gte_sem = asyncio.Semaphore(3)
    async def _resolve_one_limited(cs):
        async with _gte_sem:
            return await _resolve_one(cs)
    await asyncio.gather(*[_resolve_one_limited(cs) for cs in cases])
    dt = time.perf_counter() - _t0
    for cs in cases:
        cs.stage_times["ner_resolve"] = dt / len(cases)
    print(f"  Stage 0 (NER resolve): {dt:.2f}s for {len(cases)} cases")


async def stage_1_decomposition(session, cases: List[CaseState]):
    """Two-stage decomposition: 1a (entity analysis + rewrite) → 1b (chain decomposition)."""
    _t0 = time.perf_counter()
    active = [cs for cs in cases if cs.active]
    if not active:
        return

    # ══════ Stage 1a: Entity analysis + question rewrite ══════
    prompts_1a = []
    for cs in active:
        if cs.use_ner and cs.ner_top_ents:
            _ents = cs.sample.get("q_entity", [])
            ent_str = "\n".join(f"- {e}" for e in _ents)
            prompt_text = ENTITY_ANALYSIS_PROMPT.format(entities=ent_str, question=cs.question)
            cs._1a_prompt = prompt_text
            prompts_1a.append([{"role": "user", "content": prompt_text}])
        else:
            # Non-NER mode: skip 1a, use DECOMP_PROMPT directly
            decomp_q = f"Question: {cs.question}"
            if cs.anchor_forbidden:
                decomp_q += f"\nDo not use this previous anchor again: {cs.anchor_forbidden}"
            cs.decomp_question = decomp_q
            prompts_1a.append(None)

    # Run 1a for NER cases only
    ner_cases = [cs for cs, p in zip(active, prompts_1a) if p is not None]
    ner_prompts = [p for p in prompts_1a if p is not None]
    ner_responses = await batch_call_llm(session, ner_prompts, max_tokens=800) if ner_prompts else []

    # Parse 1a results
    ner_idx = 0
    for cs in active:
        if not (cs.use_ner and cs.ner_top_ents):
            continue
        raw_1a = ner_responses[ner_idx] if ner_idx < len(ner_responses) else ""
        ner_idx += 1

        # Extract anchor, endpoints, interpretation, answer_type, rewritten from 1a output
        anchor_name = None
        endpoints_str = "none"
        answer_type = None
        rewritten = cs.question  # fallback
        interpretation = ""

        if raw_1a:
            def _xml(tag, text):
                m = re.search(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
                return m.group(1).strip() if m else None
            def _line(prefix, text):
                m = re.search(rf'{prefix}:\s*(.+)', text)
                return m.group(1).strip() if m else None

            anchor_name = _xml('anchor', raw_1a) or _line('Anchor', raw_1a)
            endpoints_str = _xml('endpoints', raw_1a) or _line('Endpoints', raw_1a) or "none"
            interpretation = _xml('interpretation', raw_1a) or _line('Interpretation', raw_1a) or ""
            answer_type = _xml('answer_type', raw_1a) or _line('Answer_type', raw_1a)
            rewritten = _xml('rewritten', raw_1a) or _line('Rewritten', raw_1a)

        cs._1a_anchor = anchor_name
        cs._1a_endpoints = endpoints_str
        cs._1a_answer_type = answer_type
        cs._1a_rewritten = rewritten
        cs._1a_interpretation = interpretation
        cs._1a_raw = raw_1a or ""

    # ══════ Stage 1b: Chain decomposition ══════
    prompts_1b = []
    for cs in active:
        if cs.use_ner and cs.ner_top_ents:
            _ents = cs.sample.get("q_entity", [])
            ent_str = "\n".join(f"- {e}" for e in _ents)
            anchor = cs._1a_anchor or _ents[0].strip()
            endpoints = cs._1a_endpoints or "none"
            answer_type = cs._1a_answer_type or "other"
            rewritten = cs._1a_rewritten or cs.question
            prompt_text = CHAIN_PROMPT.format(
                entities=ent_str, anchor=anchor, endpoints=endpoints,
                answer_type=answer_type, interpretation=cs._1a_interpretation or "",
                rewritten=rewritten, question=cs.question)
            cs.decomp_prompt_formatted = prompt_text
            cs.decomp_question = prompt_text
            prompts_1b.append([{"role": "user", "content": prompt_text}])
        else:
            decomp_q = f"Question: {cs.question}"
            if cs.anchor_forbidden:
                decomp_q += f"\nDo not use this previous anchor again: {cs.anchor_forbidden}"
            cs.decomp_question = decomp_q
            prompts_1b.append([
                {"role": "system", "content": DECOMP_PROMPT},
                {"role": "user", "content": decomp_q},
            ])

    responses = await batch_call_llm(session, prompts_1b, max_tokens=1500)

    for cs, raw in zip(active, responses):
        cs.decomp_raw = raw or ""

        if cs.use_ner and cs.ner_top_ents:
            # ── NER mode: use parse_chain ──
            parsed = parse_chain(raw or "")
            if not parsed or not parsed.get('hops'):
                cs.error = "decomposition failed"
                cs.active = False
                continue

            anchor_name = parsed['anchor']
            # Front-end validation: anchor must match a q_entity
            q_ents = cs.sample.get("q_entity", [])
            if not any(anchor_name.lower().strip() == qe.lower().strip() for qe in q_ents):
                # Fuzzy match: anchor contained in q_entity or vice versa
                an = anchor_name.lower().strip()
                matched = next((qe for qe in q_ents if an in qe.lower() or qe.lower() in an), None)
                if matched:
                    anchor_name = matched
                elif cs._1a_anchor:
                    an1 = cs._1a_anchor.lower().strip()
                    matched1 = next((qe for qe in q_ents if an1 in qe.lower() or qe.lower() in an1), None)
                    if matched1:
                        anchor_name = matched1
                    else:
                        cs.error = "decomposition failed"
                        cs.active = False
                        continue
                else:
                    cs.error = "decomposition failed"
                    cs.active = False
                    continue
            # Convert hops to cs.steps format
            # Prefer 1a answer_type, fallback to chain-parsed
            cs.answer_type = cs._1a_answer_type or parsed.get('answer_type')
            cs.rewritten_question = cs._1a_rewritten or cs.question
            cs.steps = []
            for i, hop in enumerate(parsed['hops']):
                # Find endpoint for this hop from endpoint_entities
                ep = None
                for ep_info in parsed.get('endpoint_entities', []):
                    if ep_info.get('hop') == i + 1:
                        ep = ep_info['entity']
                        break
                # Also check 1a endpoints if chain didn't produce one
                if not ep and cs._1a_endpoints and cs._1a_endpoints.lower() != 'none':
                    ep = cs._1a_endpoints.strip('[] ')
                cs.steps.append({
                    "step": i + 1,
                    "question": hop['relation'],
                    "type": "find",
                    "relation_query": hop.get('keyword', hop['relation']),
                    "definition": hop.get('definition', ''),
                    "keyword": hop.get('keyword', ''),
                    "subquestion": hop.get('subquestion', ''),
                    "endpoint": ep,
                    "endpoint_query": ep,
                })

            # Resolve endpoint indices: e2e-style direct string matching, then GTE+LLM fallback
            cs._pending_endpoints = []  # will collect unresolved endpoints for Stage 2
            cs.breakpoints = {}
            q_entities = cs.sample.get("q_entity", [])

            # Collect endpoints from both 1a and chain parse
            all_endpoints = list(parsed.get('endpoint_entities', []))
            if cs._1a_endpoints and cs._1a_endpoints.lower() != 'none':
                # 1a endpoint — assign to last hop if no hop specified
                ep_name = cs._1a_endpoints.strip('[] ')
                if not any(e['entity'] == ep_name for e in all_endpoints):
                    all_endpoints.append({'entity': ep_name, 'hop': len(cs.steps)})

            for ep_info in all_endpoints:
                ep_name = ep_info.get('entity')
                hop_num = ep_info.get('hop')
                if not ep_name or not hop_num:
                    continue
                en = normalize(ep_name)
                ep_idx = None
                # Pass 1: exact match
                for i, e in enumerate(cs.ents):
                    if normalize(e) == en and not is_cvt_like(e):
                        ep_idx = i
                        break
                # Pass 2: match via q_entities
                if ep_idx is None:
                    for qe in q_entities:
                        qn = normalize(qe)
                        if qn == en or (min(len(qn), len(en)) >= 4 and (qn in en or en in qn)):
                            for i, e in enumerate(cs.ents):
                                if normalize(e) == qn and not is_cvt_like(e):
                                    ep_idx = i
                                    break
                            if ep_idx is not None:
                                break
                # Pass 3: substring match (same threshold as e2e)
                if ep_idx is None:
                    for i, e in enumerate(cs.ents):
                        en2 = normalize(e)
                        if not is_cvt_like(e) and min(len(en), len(en2)) >= 4 and (en in en2 or en2 in en):
                            ep_idx = i
                            break
                if ep_idx is not None:
                    cs.breakpoints[hop_num] = ep_idx
                else:
                    # String match failed — defer to GTE+LLM resolution in Stage 2
                    cs._pending_endpoints.append({"step_idx": hop_num, "query": ep_name})

            # Resolve anchor: e2e-style direct search against full ents list
            q_entities = cs.sample.get("q_entity", [])
            if anchor_name:
                sn = normalize(anchor_name)
                # Pass 1: exact match against full entity list
                for i, e in enumerate(cs.ents):
                    if normalize(e) == sn and not is_cvt_like(e):
                        cs.anchor_idx = i
                        cs.anchor_name = e
                        break
                # Pass 2: substring match via q_entities
                if cs.anchor_idx is None:
                    for qe in q_entities:
                        qn = normalize(qe)
                        if qn in sn or sn in qn:
                            for i, e in enumerate(cs.ents):
                                if normalize(e) == qn and not is_cvt_like(e):
                                    cs.anchor_idx = i
                                    cs.anchor_name = e
                                    break
                            if cs.anchor_idx is not None:
                                break
                # Pass 3: substring match against all entities
                if cs.anchor_idx is None:
                    best = None
                    for i, e in enumerate(cs.ents):
                        en = normalize(e)
                        if (sn in en or en in sn) and not is_cvt_like(e) and len(sn) >= 3:
                            if best is None or len(e) > len(cs.ents[best]):
                                best = i
                    if best is not None:
                        cs.anchor_idx = best
                        cs.anchor_name = cs.ents[best]
            if cs.anchor_idx is None:
                pass  # No fallback — better to skip than use wrong anchor

            cs.entity_retrieval_details.append({
                "role": "anchor_ner",
                "ner_top_ents": cs.ner_top_ents[:6],
                "selected": cs.anchor_name,
                "selected_idx": cs.anchor_idx,
            })
            for step in cs.steps:
                step["entity_query"] = None
        else:
            # ── Non-NER mode: use old parse_decomposition ──
            anchor_eq_name, anchor_eq, cs.steps, cs.answer_type = parse_decomposition(raw or "")
            if not cs.steps:
                cs.error = "decomposition failed"
                cs.active = False
                continue

            cs._pending_endpoints = []
            for step in cs.steps:
                if step["endpoint"] and step.get("endpoint_query"):
                    cs._pending_endpoints.append({"step_idx": step["step"], "query": step["endpoint_query"]})

            cs._pending_anchor_eq = anchor_eq
            cs._pending_anchor_eq_name = anchor_eq_name

    dt = time.perf_counter() - _t0
    for cs in active:
        cs.stage_times["decomposition"] = dt / len(active)
    ok = sum(1 for cs in active if cs.active)
    print(f"  Stage 1 (Decomposition): {dt:.2f}s | {ok}/{len(active)} ok")


async def stage_1_5_decomposition_reflect(session, cases: List[CaseState]):
    """Rule-based retry: re-invoke Stage 1b for 1-step decompositions."""
    _t0 = time.perf_counter()
    active = [cs for cs in cases if cs.active]
    if not active:
        return

    # Rule: only retry 1-step decompositions (CWQ requires multi-hop reasoning)
    retry_cases = []
    retry_prompts = []
    for cs in active:
        if len(cs.steps) <= 1:
            retry_cases.append(cs)
            # Reuse Stage 1b CHAIN_PROMPT with added hint
            _ents = cs.sample.get("q_entity", [])
            ent_str = "\n".join(f"- {e}" for e in _ents)
            anchor = cs._1a_anchor or cs.anchor_name or (_ents[0].strip() if _ents else "N/A")
            endpoints = cs._1a_endpoints or "none"
            answer_type = cs._1a_answer_type or "other"
            rewritten = cs._1a_rewritten or cs.question

            prompt_text = CHAIN_PROMPT.format(
                entities=ent_str, anchor=anchor, endpoints=endpoints,
                answer_type=answer_type, interpretation=cs._1a_interpretation or "",
                rewritten=rewritten, question=cs.question)

            # Add hint about needing 2+ hops
            prev_step = cs.steps[0].get('question', '') if cs.steps else ''
            prev_rel = cs.steps[0].get('relation_query', '') if cs.steps else ''
            hint = f"""

[NOTE: The previous decomposition produced only 1 hop ("{prev_step}" / rel: {prev_rel}), which is INSUFFICIENT for this complex question. You MUST produce at least 2 hops. Think about the intermediate entity between the anchor and the final answer.]"""
            prompt_text += hint
            retry_prompts.append([{"role": "user", "content": prompt_text}])

    n_retry = len(retry_cases)
    if n_retry == 0:
        dt = time.perf_counter() - _t0
        for cs in active:
            cs.stage_times["decomposition_reflect"] = dt / len(active)
        print(f"  Stage 1.5 (Reflect): {dt:.2f}s | 0 retry")
        return

    retry_responses = await batch_call_llm(session, retry_prompts, max_tokens=1500)

    n_applied = 0
    for cs, raw in zip(retry_cases, retry_responses):
        # Always record retry metadata
        cs.decomp_retry = True
        cs.decomp_reflect_raw = raw or ""
        cs.decomp_retry_reason = f"Only {len(cs.steps)} step for complex question"
        cs.decomp_raw = (cs.decomp_raw or "") + "\n\n[RETRY]\n" + (raw or "")

        parsed = parse_chain(raw or "")
        if not parsed or not parsed.get('hops') or len(parsed['hops']) < 2:
            print(f"    Case {cs.case_id}: retry still has {len(parsed.get('hops',[])) if parsed else 0} hops")
            continue

        anchor_name = parsed['anchor']
        q_ents = cs.sample.get("q_entity", [])
        if not any(anchor_name.lower().strip() == qe.lower().strip() for qe in q_ents):
            an = anchor_name.lower().strip()
            matched = next((qe for qe in q_ents if an in qe.lower() or qe.lower() in an), None)
            if matched:
                anchor_name = matched
            elif cs._1a_anchor:
                an1 = cs._1a_anchor.lower().strip()
                matched1 = next((qe for qe in q_ents if an1 in qe.lower() or qe.lower() in an1), None)
                if matched1:
                    anchor_name = matched1
                else:
                    print(f"    Case {cs.case_id}: retry anchor '{anchor_name}' not matched")
                    continue
            else:
                print(f"    Case {cs.case_id}: retry anchor '{anchor_name}' not matched")
                continue

        cs.steps = []
        for i, hop in enumerate(parsed['hops']):
            ep = None
            for ep_info in parsed.get('endpoint_entities', []):
                if ep_info.get('hop') == i + 1:
                    ep = ep_info['entity']
                    break
            cs.steps.append({
                "step": i + 1,
                "question": hop['relation'],
                "type": "find",
                "relation_query": hop.get('keyword', hop['relation']),
                "definition": hop.get('definition', ''),
                "keyword": hop.get('keyword', ''),
                "endpoint": ep,
                "endpoint_query": ep,
            })

        if anchor_name:
            sn = normalize(anchor_name)
            found = False
            for i, e in enumerate(cs.ents):
                if normalize(e) == sn and not is_cvt_like(e):
                    cs.anchor_idx = i
                    cs.anchor_name = e
                    found = True
                    break
            if not found:
                for i, e in enumerate(cs.ents):
                    en = normalize(e)
                    if (sn in en or en in sn) and not is_cvt_like(e) and len(sn) >= 3:
                        cs.anchor_idx = i
                        cs.anchor_name = e
                        found = True
                        break

        print(f"    Case {cs.case_id}: retry applied ({len(cs.steps)} steps, anchor={cs.anchor_name})")

    dt = time.perf_counter() - _t0
    for cs in active:
        cs.stage_times["decomposition_reflect"] = dt / len(active)
    print(f"  Stage 1.5 (Reflect): {dt:.2f}s | {n_retry}/{len(active)} retry")


async def stage_2_entity_resolution(session, cases: List[CaseState]):
    """Batch entity resolution. Flatten all requests across cases into one batch."""
    _t0 = time.perf_counter()
    active = [cs for cs in cases if cs.active]
    if not active:
        return

    # Phase A: GTE retrieval for all entity queries concurrently
    gte_tasks = []  # (kind, cs, query_str)
    for cs in active:
        if not (cs.use_ner and cs.ner_top_ents) and cs._pending_anchor_eq:
            gte_tasks.append(("anchor", cs, cs._pending_anchor_eq))
        for ep in cs._pending_endpoints:
            gte_tasks.append(("endpoint", cs, ep["query"]))

    _gte_sem = asyncio.Semaphore(3)
    async def _gte_limited(coro):
        async with _gte_sem:
            return await coro
    gte_results = await asyncio.gather(*[
        _gte_limited(gte_retrieve(session, query, cs.ent_candidates, top_k=5 if kind == "anchor" else 3))
        for kind, cs, query in gte_tasks
    ], return_exceptions=True)

    # Phase B: Build LLM prompts for entity resolution
    llm_items = []  # (cs, kind, step_idx, cands_with_ctx)
    for i, (kind, cs, query) in enumerate(gte_tasks):
        result = gte_results[i]
        if isinstance(result, Exception):
            continue
        candidates = [r.get("candidate", "") for r in result if r.get("candidate")]
        if len(candidates) <= 1:
            if kind == "anchor" and candidates:
                cs.anchor_name = candidates[0]
                cs.anchor_idx = cs.ents.index(candidates[0]) if candidates[0] in cs.ents else None
            continue

        ctx = get_entity_contexts(candidates, cs.h_ids, cs.r_ids, cs.t_ids, cs.ents, cs.rels)
        cands_with_ctx = [(n, ctx.get(n, "")) for n in candidates]
        llm_items.append((cs, kind, query, cands_with_ctx))

    # Phase C: Batch LLM call
    if llm_items:
        prompts = []
        for cs, kind, query, cands_with_ctx in llm_items:
            cand_lines = []
            for j, (name, c) in enumerate(cands_with_ctx, 1):
                cand_lines.append(f"  {j}. {name} [{c}]" if c else f"  {j}. {name}")
            prompt = f"""Search query: {query}

Candidate entities (with relation context from knowledge graph):
{chr(10).join(cand_lines)}

Which candidate best matches the search query? Use the relation context to identify what each entity actually IS.

<analysis>Brief reasoning about which candidate matches the query</analysis>
<selected>entity name</selected>"""
            prompts.append([
                {"role": "system", "content": "Select the correct entity from candidates. Output <analysis> and <selected> XML tags."},
                {"role": "user", "content": prompt},
            ])

        responses = await batch_call_llm(session, prompts, max_tokens=300)

        for (cs, kind, query, cands_with_ctx), raw in zip(llm_items, responses):
            sel = extract_xml_tag(raw or "", "selected")
            selected = None
            if sel:
                sel = sel.strip().strip('"').strip("'")
                for name, _ in cands_with_ctx:
                    if normalize(sel) == normalize(name):
                        selected = name; break
                if not selected:
                    for name, _ in cands_with_ctx:
                        if normalize(sel) in normalize(name) or normalize(name) in normalize(sel):
                            selected = name; break
            if not selected and cands_with_ctx:
                selected = cands_with_ctx[0][0]

            if kind == "anchor":
                cs.anchor_name = selected
                cs.anchor_idx = cs.ents.index(selected) if selected and selected in cs.ents else None
                cs.entity_retrieval_details.append({
                    "role": "anchor", "query": query, "selected": selected,
                    "selected_idx": cs.anchor_idx, "llm_resolved": True,
                })
            else:
                # Find which endpoint this resolves
                for ep in cs._pending_endpoints:
                    if ep["query"] == query:
                        step_idx = ep["step_idx"]
                        idx = cs.ents.index(selected) if selected and selected in cs.ents else None
                        if idx is not None:
                            cs.breakpoints[step_idx] = idx
                        cs.entity_retrieval_details.append({
                            "role": f"endpoint_step{step_idx}", "query": query,
                            "selected": selected, "selected_idx": idx, "llm_resolved": True,
                        })
                        break

    dt = time.perf_counter() - _t0
    for cs in active:
        cs.stage_times["entity_resolve"] = dt / len(active)
    print(f"  Stage 2 (Entity resolve): {dt:.2f}s | {sum(1 for cs in active if cs.anchor_idx is not None)} anchored")


async def stage_3_gte_relation_retrieval(session, cases: List[CaseState]):
    """Concurrent GTE relation retrieval for all steps of all cases."""
    _t0 = time.perf_counter()
    active = [cs for cs in cases if cs.active]
    if not active:
        return

    # Build flat list of all GTE calls: (cs, step, query)
    # Queries: definition, subquestion, question, keyword, relation_query, + original question
    gte_calls = []
    step_query_counts = []  # track per-(cs, step) query count for result distribution
    for cs in active:
        cs_qcounts = []
        for step in cs.steps:
            queries = []
            for field in ["definition", "subquestion", "question", "keyword", "relation_query"]:
                val = step.get(field, "")
                if val and val.strip():
                    queries.append(val.strip())
            # Add original question as stable semantic anchor (same for all models)
            if cs.question and cs.question.strip():
                queries.append(cs.question.strip())
            # Add rewritten question if available (richer semantics)
            rw = getattr(cs, 'rewritten_question', '') or getattr(cs, '_1a_rewritten', '')
            if rw and rw.strip() and rw.strip().lower() != cs.question.strip().lower():
                queries.append(rw.strip())
            # Deduplicate (case-insensitive)
            seen = set()
            n = 0
            for q in queries:
                ql = q.lower()
                if ql not in seen:
                    seen.add(ql)
                    gte_calls.append((cs, step, q))
                    n += 1
            cs_qcounts.append(n)
        step_query_counts.append(cs_qcounts)

    _gte_sem3 = asyncio.Semaphore(3)
    async def _gte_limited3(coro):
        async with _gte_sem3:
            return await coro
    gte_results = await asyncio.gather(*[
        _gte_limited3(gte_retrieve(session, query, cs.rels, candidate_texts=cs.rel_texts, top_k=10))
        for cs, step, query in gte_calls
    ], return_exceptions=True)

    # Distribute results back
    call_idx = 0
    for cs_idx, cs in enumerate(active):
        cs.step_candidates = {}
        cs.gte_per_step = {}
        cs.relation_retrieval_details = []
        for step_idx, step in enumerate(cs.steps):
            n_queries = step_query_counts[cs_idx][step_idx]
            gte_all = {}
            queries_detail = []
            for _ in range(n_queries):
                result = gte_results[call_idx]
                call_idx += 1
                if isinstance(result, Exception):
                    queries_detail.append({"query": "", "top_k": [], "error": str(result)})
                    continue
                topk = []
                for i, r in enumerate(result):
                    cand = r.get("candidate", "")
                    score = round(r.get("score", 0), 4)
                    idx_in_rels = cs.rels.index(cand) if cand in cs.rels else None
                    topk.append({"rank": i + 1, "candidate": cand, "score": score,
                                 "rel_idx": idx_in_rels,
                                 "rel_text": rel_to_text(cand) if idx_in_rels is not None else ""})
                    if idx_in_rels is not None:
                        if idx_in_rels not in gte_all or score > gte_all[idx_in_rels][1]:
                            gte_all[idx_in_rels] = (cs.rels[idx_in_rels], score)
                queries_detail.append({"query": "", "top_k": topk})

            gte_candidates = sorted(gte_all.items(), key=lambda x: -x[1][1])
            candidate_list = [(idx, name, score) for idx, (name, score) in gte_candidates]
            cs.step_candidates[step["step"]] = candidate_list
            cs.gte_per_step[step["step"]] = gte_all
            cs.relation_retrieval_details.append({
                "step": step["step"], "queries": queries_detail,
                "gte_candidates_count": len(gte_all),
                "gte_indices": sorted(gte_all.keys()),
            })

    dt = time.perf_counter() - _t0
    for cs in active:
        cs.stage_times["gte_retrieve"] = dt / len(active)
    print(f"  Stage 3 (GTE relations): {dt:.2f}s | {call_idx} calls")


async def stage_4_relation_pruning(session, cases: List[CaseState]):
    """Batch LLM relation pruning for all active cases."""
    _t0 = time.perf_counter()
    active = [cs for cs in cases if cs.active]
    if not active:
        return

    # Build prune prompts (reuse existing logic from llm_prune_all_relations)
    prompts = []
    for cs in active:
        chain_lines = []
        step_blocks = []
        for s in cs.steps:
            sn = s["step"]
            ep_str = f" -> endpoint: {s['endpoint']}" if s.get("endpoint") else ""
            chain_lines.append(f"  Step {sn}: {s['question']}{ep_str}")
            cands = cs.step_candidates.get(sn, [])
            if not cands:
                step_blocks.append(f"Step {sn}: {s['question']}\n  Purpose: {s.get('definition', '')}\n  Candidates: (none)")
                continue
            cand_lines = [f"    {i}. {name}" for i, (idx, name, score) in enumerate(cands, 1)]
            step_blocks.append(
                f"Step {sn}: {s['question']}\n  Purpose: {s.get('definition', '')}\n  Candidates:\n" + "\n".join(cand_lines))
        chain_text = "\n".join(chain_lines)
        blocks_text = "\n\n".join(step_blocks)
        prompt = f"""Analyze and select knowledge graph relations for each step of this reasoning chain.

Question: {cs.question}

Reasoning chain:
{chain_text}

Step-by-step candidates:
{blocks_text}

Rules:
1. Each step connects FROM previous output TO next — select bridge relations
2. Select 2-4 relevant relations per step. Pick the best candidates that match the step's purpose.
3. Ignore unrelated attributes
4. If no relations fit a step, output empty list
5. ORDER matters: rank by relevance to the step (most relevant first)

Output format:
<analysis>
One sentence per step: what it needs and which relations fit.
</analysis>
<selected>
step_1: [3, 1]
step_2: [5, 2]
</selected>

Numbers are RANKED: first = most relevant."""
        prompts.append([
            {"role": "system", "content": "You are a knowledge graph relation selector for multi-step QA. Analyze the full chain, then select relevant relations per step. Output <analysis> and <selected> XML tags."},
            {"role": "user", "content": prompt},
        ])

    responses = await batch_call_llm(session, prompts, max_tokens=2000)

    for ci, (cs, raw) in enumerate(zip(active, responses)):
        selected_yaml = extract_xml_tag(raw or "", "selected")
        result = {}
        if selected_yaml:
            for line in selected_yaml.split('\n'):
                line = line.strip()
                m = re.match(r'step_(\d+)\s*:\s*\[(.*?)\]', line)
                if m:
                    sn = int(m.group(1))
                    nums = [int(x.strip()) for x in m.group(2).split(',') if x.strip().isdigit()]
                    cands = cs.step_candidates.get(sn, [])
                    # Preserve LLM ranking order
                    ranked_indices = []
                    seen_idx = set()
                    for n in nums:
                        if 1 <= n <= len(cands):
                            idx = cands[n - 1][0]
                            if idx not in seen_idx:
                                ranked_indices.append(idx)
                                seen_idx.add(idx)
                    result[sn] = ranked_indices

        # Fallback: top-3 GTE for missing steps
        for s in cs.steps:
            sn = s["step"]
            if sn not in result or not result[sn]:
                cands = cs.step_candidates.get(sn, [])
                result[sn] = [idx for idx, _, _ in cands[:3]]

        # Build step_relations: LLM-selected only (no padding)
        cs.step_relations = []
        for step in cs.steps:
            sn = step["step"]
            ranked = result.get(sn, [])
            cs.step_relations.append(ranked[:5])

        cs.prune_debug = {
            "prompt": prompts[ci][1]["content"],
            "response": raw,
            "parsed_yaml": selected_yaml,
            "parsed_result": {sn: list(indices) for sn, indices in result.items()},
            "stage4_step_relations": [list(rs) for rs in cs.step_relations],
        }

        # ── Stage 4.5: Relation Preflight Probe ──
        filtered_rels, preflight_dbg = relation_preflight_probe(
            cs.anchor_idx, cs.step_relations, cs.steps,
            cs.h_ids, cs.r_ids, cs.t_ids, cs.ents, cs.rels,
        )
        cs.step_relations = filtered_rels
        cs.prune_debug["preflight_step_relations"] = [list(rs) for rs in cs.step_relations]
        cs.prune_debug["preflight"] = preflight_dbg

    dt = time.perf_counter() - _t0
    for cs in active:
        cs.stage_times["pruning"] = dt / len(active)
    print(f"  Stage 4 (Relation pruning): {dt:.2f}s")


# ── Reranker-based pruning (alternative to LLM) ──
# Uses Qwen3-Reranker-0.6B: CausalLM with yes/no logit scoring

_RERANK_MODEL = None
_RERANK_TOKENIZER = None
_RERANK_PREFIX_TOKENS = None
_RERANK_SUFFIX_TOKENS = None
_RERANK_TOKEN_FALSE_ID = None
_RERANK_TOKEN_TRUE_ID = None

_RERANK_PREFIX = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n\n\n\n\n"
_RERANK_MAX_LENGTH = 8192
_RERANK_INSTRUCT = "Given a knowledge graph question, retrieve relevant graph relations that connect entities to answer the question"


def _load_rerank_model(model_path):
    global _RERANK_MODEL, _RERANK_TOKENIZER, _RERANK_PREFIX_TOKENS, _RERANK_SUFFIX_TOKENS
    global _RERANK_TOKEN_FALSE_ID, _RERANK_TOKEN_TRUE_ID
    if _RERANK_MODEL is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Loading reranker: {model_path} ...")
    _RERANK_TOKENIZER = AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True)
    _RERANK_MODEL = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).cuda().eval()
    _RERANK_PREFIX_TOKENS = _RERANK_TOKENIZER.encode(_RERANK_PREFIX, add_special_tokens=False)
    _RERANK_SUFFIX_TOKENS = _RERANK_TOKENIZER.encode(_RERANK_SUFFIX, add_special_tokens=False)
    _RERANK_TOKEN_FALSE_ID = _RERANK_TOKENIZER.convert_tokens_to_ids("no")
    _RERANK_TOKEN_TRUE_ID = _RERANK_TOKENIZER.convert_tokens_to_ids("yes")
    print(f"  Reranker loaded. (yes_id={_RERANK_TOKEN_TRUE_ID}, no_id={_RERANK_TOKEN_FALSE_ID})")


def _rerank_score_pairs(pairs):
    """Score query-document pairs using Qwen3-Reranker yes/no logit method.

    pairs: list of formatted instruction strings (one per candidate)
    Returns: list of float scores (probability of "yes")
    """
    import torch
    max_content = _RERANK_MAX_LENGTH - len(_RERANK_PREFIX_TOKENS) - len(_RERANK_SUFFIX_TOKENS)
    inputs = _RERANK_TOKENIZER(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_content
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = _RERANK_PREFIX_TOKENS + ele + _RERANK_SUFFIX_TOKENS
    inputs = _RERANK_TOKENIZER.pad(inputs, padding=True, return_tensors="pt", max_length=_RERANK_MAX_LENGTH)
    inputs = {k: v.to(_RERANK_MODEL.device) for k, v in inputs.items()}

    with torch.no_grad():
        batch_scores = _RERANK_MODEL(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, _RERANK_TOKEN_TRUE_ID]
        false_vector = batch_scores[:, _RERANK_TOKEN_FALSE_ID]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
    return scores


async def stage_4_rerank_pruning(cases: List[CaseState], model_path: str, top_k: int = 5):
    """Reranker-based relation pruning using Qwen3-Reranker-0.6B (CausalLM yes/no scoring)."""
    _t0 = time.perf_counter()
    active = [cs for cs in cases if cs.active]
    if not active:
        return

    _load_rerank_model(model_path)

    for cs in active:
        cs.step_relations = []
        for step in cs.steps:
            sn = step["step"]
            cands = cs.step_candidates.get(sn, [])
            if not cands:
                cs.step_relations.append([])
                continue

            step_query = step.get('definition', '') or step.get('question', '') or step.get('relation_query', '') or cs.question
            cand_names = [name for _, name, _ in cands]

            # Format each candidate as an instruction pair for the reranker
            pairs = []
            for cname in cand_names:
                text = f"<Instruct>: {_RERANK_INSTRUCT}\n<Query>: {step_query}\n<Document>: {cname}"
                pairs.append(text)

            scores = _rerank_score_pairs(pairs)

            # Sort candidates by reranker score, keep top_k
            ranked = sorted(zip(cands, scores), key=lambda x: -x[1])
            selected_indices = [idx for (idx, _, _), _ in ranked[:top_k]]
            cs.step_relations.append(selected_indices)

        cs.prune_debug = {"method": "rerank", "model": model_path}

    dt = time.perf_counter() - _t0
    for cs in active:
        cs.stage_times["pruning"] = dt / len(active)
    print(f"  Stage 4 (Rerank pruning): {dt:.2f}s")


def _canonicalize_by_step_hits(path, layer_rels):
    """Canonicalize a raw path by truncating each step at its first target-relation hit.

    For each step, scan the relation sequence from the current position and find the
    first edge matching that step's target relations. Keep edges up to that hit, then
    continue scanning from the hit point for the next step. This prevents a step from
    accumulating extra relations after its first valid hit.

    CVT nodes at hit points are preserved — the next step continues from them.
    """
    nodes = path["nodes"]
    rels = path["relations"]

    if not rels or not layer_rels:
        return path

    keep_nodes = [nodes[0]]
    keep_rels = []
    committed_nodes = [nodes[0]]
    covered = set()
    scan_start = 0

    for step_idx, target_rels in enumerate(layer_rels):
        if not target_rels:
            continue
        target_set = set(target_rels)
        hit = None
        for e in range(scan_start, len(rels)):
            if rels[e] in target_set:
                hit = e
                break
        if hit is None:
            break
        # Keep all edges from scan_start through hit
        for e in range(scan_start, hit + 1):
            keep_rels.append(rels[e])
            keep_nodes.append(nodes[e + 1])
        committed_nodes.append(nodes[hit + 1])
        covered.add(step_idx)
        scan_start = hit + 1

    if not keep_rels:
        return path

    return {
        "nodes": keep_nodes,
        "relations": keep_rels,
        "committed_nodes": committed_nodes,
        "covered_steps": frozenset(covered),
        "depth": len(keep_nodes) - 1,
    }


def frontier_expand_layers(anchor_idx, step_relations, steps,
                           h_ids, r_ids, t_ids, entity_list,
                           beam_width=80, max_hops_per_step=2):
    """Multi-step traversal: single BFS per step, hit-and-stop per branch.

    Per step:
    1. From each active path's endpoint, BFS up to K hops
    2. When an edge matches a target relation → record that path, STOP the branch
    3. Other branches continue independently (backtrack to explore other edges)
    4. First encountered target relation per branch — no stacking
    5. Matched endpoints form the next step's frontier

    Step-skip fallback: if entire step produces zero matches, borrow the
    next step's target relations and re-explore from the current frontier.

    step_relations: List[List[int]] — target relation indices per step.
    Returns (paths, max_depth, max_cov) compatible with relation_prior_expand.
    """
    n_layers = len(step_relations)
    if n_layers == 0:
        return [], 0, 0

    layer_rels: List[List[int]] = step_relations
    if not any(layer_rels):
        return [], 0, 0

    # Build adjacency: node_idx → list of (neighbor_idx, rel_idx)
    adj: Dict[int, list] = {}
    for i in range(len(h_ids)):
        h, r, t = h_ids[i], r_ids[i], t_ids[i]
        adj.setdefault(h, []).append((t, r))
        adj.setdefault(t, []).append((h, r))

    n_ents = len(entity_list)

    def _expand_through_cvt(nb_idx, current_idx, seen):
        return [(nb_idx, [nb_idx], [])]

    def _bfs_step(paths, rels, cover_step):
        """Single BFS pass: find K-hop paths ending with a target relation."""
        results = []
        pb = max(beam_width // max(len(paths), 1), 5)
        for path in paths:
            start = path["nodes"][-1]
            committed_nodes = list(path.get("committed_nodes", [path["nodes"][0]]))
            p_seen = set(committed_nodes)
            frontier = [(start, [], [])]
            hits = []
            for hop in range(max_hops_per_step):
                nf = []
                for cur, in_nodes, in_rels in frontier:
                    seen = p_seen | set(in_nodes)
                    for nb, rel in adj.get(cur, []):
                        if nb in seen:
                            continue
                        # Inverse-pair loop detection
                        if in_rels and in_rels[-1] == rel:
                            continue
                        if rel in rels:
                            for fn, en, er in _expand_through_cvt(nb, cur, seen):
                                hits.append((fn, in_nodes + en, in_rels + [rel] + er))
                        else:
                            if hop < max_hops_per_step - 1:
                                nb_name = entity_list[nb] if 0 <= nb < n_ents else ""
                                if is_cvt_like(nb_name):
                                    for cvt_nb, cvt_rel in adj.get(nb, []):
                                        if cvt_nb not in seen and cvt_nb != cur:
                                            if cvt_rel in rels:
                                                hits.append((cvt_nb, in_nodes + [nb, cvt_nb],
                                                             in_rels + [rel, cvt_rel]))
                                            else:
                                                nf.append((cvt_nb, in_nodes + [nb, cvt_nb],
                                                            in_rels + [rel, cvt_rel]))
                                else:
                                    nf.append((nb, in_nodes + [nb], in_rels + [rel]))
                if len(nf) > pb:
                    nf = nf[:pb]
                frontier = nf
            for end_node, extra_nodes, extra_rels in hits:
                results.append({
                    "nodes": path["nodes"] + extra_nodes,
                    "relations": path["relations"] + extra_rels,
                    "committed_nodes": committed_nodes + [end_node],
                    "covered_steps": path["covered_steps"] | frozenset({cover_step}),
                    "depth": path["depth"] + len(extra_nodes),
                })
            if not hits:
                results.append(path)
        return results

    # Active paths: nodes, relations, covered_steps
    active = [{"nodes": [anchor_idx], "relations": [],
               "committed_nodes": [anchor_idx],
               "covered_steps": frozenset(), "depth": 0}]

    step_idx = 0
    while step_idx < n_layers:
        target_rels = set(layer_rels[step_idx])
        if not target_rels:
            step_idx += 1
            continue

        new_active = _bfs_step(active, target_rels, step_idx)
        advance = 1

        # Step-skip fallback: entire step unmatched → borrow next step's relations
        any_matched = any(step_idx in p["covered_steps"] for p in new_active)
        if not any_matched and step_idx + 1 < n_layers:
            fallback_rels = set(layer_rels[step_idx + 1])
            if fallback_rels:
                fallback = _bfs_step(active, fallback_rels, step_idx + 1)
                if any(step_idx + 1 in p["covered_steps"] for p in fallback):
                    new_active = fallback
                    advance = 2

        if not new_active:
            break

        # Beam prune: group by (endpoint, covered_steps) for diversity
        if len(new_active) > beam_width:
            grouped: Dict[tuple, list] = {}
            for p in new_active:
                sig = (p["nodes"][-1], p["covered_steps"])
                grouped.setdefault(sig, []).append(p)
            for sig, group in grouped.items():
                group.sort(key=lambda x: (-len(x["covered_steps"]), x["depth"]))
            result = [group[0] for group in grouped.values()]
            overflow = []
            for group in grouped.values():
                overflow.extend(group[1:])
            overflow.sort(key=lambda x: (-len(x["covered_steps"]), x["depth"]))
            remaining = beam_width - len(result)
            if remaining > 0:
                result.extend(overflow[:remaining])
            new_active = result[:beam_width]

        active = new_active
        step_idx += advance

    # Convert to output format
    paths = [p for p in active if p["depth"] > 0]
    if not paths:
        return [], 0, 0

    # Post-processing: canonicalize each path by step-level first-hit truncation.
    # For each step, keep only edges up to the first hit of that step's target relations,
    # then continue from the hit point for the next step. This prevents a single step
    # from accumulating extra relations after its first valid hit.
    canonicalized = []
    for p in paths:
        canon = _canonicalize_by_step_hits(p, layer_rels)
        if canon and canon["depth"] > 0:
            canonicalized.append(canon)
    if canonicalized:
        paths = canonicalized

    # Sort by layers covered (desc), then depth (asc: prefer shorter)
    paths.sort(key=lambda p: (-len(p["covered_steps"]), p["depth"]))

    # Add matched_relations for compatibility
    for p in paths:
        p["matched_relations"] = frozenset(p["relations"])

    max_depth = max(p["depth"] for p in paths)
    max_cov = max(len(p["covered_steps"]) for p in paths)

    return paths, max_depth, max_cov


async def stage_5_graph_traversal(cases: List[CaseState]):
    """Hybrid graph traversal: frontier-first, relation_prior_expand fallback for weak cases."""
    _t0 = time.perf_counter()
    active = [cs for cs in cases if cs.active]

    def _traverse_one(cs: CaseState):
        if cs.anchor_idx is None:
            cs.needs_direct_answer = True
            return
        bp_set = set(cs.breakpoints.values()) - {cs.anchor_idx, None}
        n_steps = len(cs.steps)
        # explicit_targets from resolved endpoints (same as e2e)
        explicit_targets = list(bp_set) if bp_set else None

        # ── Always frontier-first (same as e2e) ──
        paths, max_depth, max_cov = frontier_expand_layers(
            cs.anchor_idx, cs.step_relations, cs.steps,
            cs.h_ids, cs.r_ids, cs.t_ids, cs.ents)

        # RPE fallback for broader coverage
        if max_cov < n_steps or n_steps <= 1:
            rpe_paths, rpe_depth, rpe_cov = relation_prior_expand(
                cs.anchor_idx, [set(rs) for rs in cs.step_relations],
                cs.h_ids, cs.r_ids, cs.t_ids, cs.ents,
                explicit_targets=explicit_targets)
            if rpe_cov > max_cov:
                paths, max_depth, max_cov = rpe_paths, rpe_depth, rpe_cov
            elif rpe_paths:
                existing_sigs = {(tuple(p["relations"][:3]), p["nodes"][-1]) for p in paths}
                for rp in rpe_paths:
                    sig = (tuple(rp["relations"][:3]), rp["nodes"][-1])
                    if sig not in existing_sigs:
                        paths.append(rp)
                        existing_sigs.add(sig)
        # Prefer paths hitting breakpoint endpoints
        paths = prefer_breakpoint_hit_paths(
            paths, cs.breakpoints, cs.h_ids, cs.r_ids, cs.t_ids, cs.ents
        )
        if paths:
            max_depth = max(p.get("depth", 0) for p in paths)
            max_cov = max(len(p.get("covered_steps", frozenset())) for p in paths)
        cs.paths = paths
        cs.max_depth = max_depth
        cs.max_cov = max_cov

        # Collect subgraph nodes
        cs.all_subgraph_nodes = {cs.anchor_idx}
        for path in cs.paths:
            cs.all_subgraph_nodes.update(path["nodes"])

        # HR frontier: path-level (h+r) forward + (r+t) reverse triples
        expanded_rels = [set(rs) for rs in cs.step_relations]
        hr_triples, hr_nodes = _collect_hr_frontier(
            cs.anchor_idx, expanded_rels, cs.h_ids, cs.r_ids, cs.t_ids,
            paths=cs.paths)
        cs.all_subgraph_nodes |= hr_nodes

        # Answer candidates from last 2 hops of each path (+ CVT expansion)
        answer_candidates = []
        last2_nodes = set()
        for path in cs.paths:
            nodes = path.get("nodes", [])
            for n in nodes[-2:]:
                if n != cs.anchor_idx:
                    last2_nodes.add(n)
        for node_idx in sorted(last2_nodes):
            name = cs.ents[node_idx] if 0 <= node_idx < len(cs.ents) else ""
            if is_cvt_like(name):
                for cvt_idx, _ in expand_through_cvt(node_idx, cs.h_ids, cs.r_ids, cs.t_ids, cs.ents):
                    if cvt_idx != cs.anchor_idx and 0 <= cvt_idx < len(cs.ents) and not is_cvt_like(cs.ents[cvt_idx]):
                        answer_candidates.append(cs.ents[cvt_idx])
            else:
                answer_candidates.append(name)
        seen = set()
        unique = []
        for c in answer_candidates:
            nc = normalize(c)
            if len(nc) < 2:
                continue
            if not c.isascii():
                continue
            if nc not in seen:
                seen.add(nc)
                unique.append(c)
        cs.answer_candidates = unique

        # ── Level 1-2: Progressive relation/layer filtering for candidate explosion ──
        if len(cs.paths) > CANDIDATE_THRESHOLD and cs.paths:
            step_rel_quality = evaluate_step_relations(
                cs.anchor_idx, cs.paths, cs.step_relations, cs.ents)

            filtered_step_relations = []
            any_removed = False
            layers_skipped = 0
            for li, rs in enumerate(cs.step_relations):
                rs_set = set(rs) if not isinstance(rs, set) else rs
                if not rs_set:
                    filtered_step_relations.append(rs_set)
                    continue
                rel_counts = step_rel_quality.get(li, {})
                clean = {r for r in rs_set if rel_counts.get(r, 0) <= CANDIDATE_THRESHOLD}
                noisy = {r for r in rs_set if rel_counts.get(r, 0) > CANDIDATE_THRESHOLD}

                if clean:
                    filtered_step_relations.append(clean)
                    if noisy:
                        any_removed = True
                else:
                    # Level 2: all relations in this layer are noisy → skip
                    filtered_step_relations.append(set())
                    any_removed = True
                    layers_skipped += 1

            if any_removed:
                bp_set_filt = set(cs.breakpoints.values()) - {cs.anchor_idx, None}
                paths_new, max_depth_new, max_cov_new = relation_prior_expand(
                    cs.anchor_idx, filtered_step_relations,
                    cs.h_ids, cs.r_ids, cs.t_ids, cs.ents,
                    explicit_targets=list(bp_set_filt) if bp_set_filt else None,
                    max_hops=len(filtered_step_relations))
                paths_new = prefer_breakpoint_hit_paths(
                    paths_new, cs.breakpoints, cs.h_ids, cs.r_ids, cs.t_ids, cs.ents)

                # Rebuild candidates from filtered paths (last 2 hops only)
                last2_new = set()
                for p in paths_new:
                    nodes = p.get("nodes", [])
                    for n in nodes[-2:]:
                        if n != cs.anchor_idx:
                            last2_new.add(n)

                new_candidates = []
                for node_idx in sorted(last2_new):
                    if node_idx == cs.anchor_idx:
                        continue
                    name = cs.ents[node_idx] if 0 <= node_idx < len(cs.ents) else ""
                    if is_cvt_like(name):
                        for cvt_idx, _ in expand_through_cvt(node_idx, cs.h_ids, cs.r_ids, cs.t_ids, cs.ents):
                            if cvt_idx != cs.anchor_idx and 0 <= cvt_idx < len(cs.ents) and not is_cvt_like(cs.ents[cvt_idx]):
                                new_candidates.append(cs.ents[cvt_idx])
                    else:
                        new_candidates.append(name)
                seen_n = set()
                unique_n = []
                for c in new_candidates:
                    nc = normalize(c)
                    if len(nc) < 2 or not c.isascii():
                        continue
                    if nc not in seen_n:
                        seen_n.add(nc)
                        unique_n.append(c)

                old_max_cov = cs.max_cov
                old_path_count = len(cs.paths)
                new_max_cov = max((len(p.get("covered_steps", frozenset())) for p in paths_new), default=0)
                accepted_filter = (
                    bool(paths_new)
                    and new_max_cov >= old_max_cov
                    and len(unique_n) < len(cs.answer_candidates)
                )
                filter_debug = {
                    "triggered": True,
                    "accepted": accepted_filter,
                    "before_step_relations": [list(rs) for rs in cs.step_relations],
                    "after_step_relations": [list(rs) for rs in filtered_step_relations],
                    "before_candidates": len(cs.answer_candidates),
                    "after_candidates": len(unique_n),
                    "before_max_cov": old_max_cov,
                    "after_max_cov": new_max_cov,
                    "before_paths": old_path_count,
                    "after_paths": len(paths_new),
                    "relation_candidate_counts": {
                        str(li): {str(r): c for r, c in counts.items()}
                        for li, counts in step_rel_quality.items()
                    },
                }
                if not isinstance(cs.prune_debug, dict):
                    cs.prune_debug = {}
                cs.prune_debug["level_filter"] = filter_debug

                if accepted_filter:
                    cs.paths = paths_new
                    cs.step_relations = filtered_step_relations
                    cs.answer_candidates = unique_n
                    cs.all_subgraph_nodes = {cs.anchor_idx}
                    for p in paths_new:
                        cs.all_subgraph_nodes.update(p.get("nodes", []))
                    if paths_new:
                        cs.max_depth = max(p.get("depth", 0) for p in paths_new)
                        cs.max_cov = max(len(p.get("covered_steps", frozenset())) for p in paths_new)
                    cs.layers_skipped_by_filter = layers_skipped

        # GT recall from path entities only (not HR frontier)
        cs.path_candidates = _extract_path_candidates(cs.paths, cs.anchor_idx, cs.ents, cs.h_ids, cs.r_ids, cs.t_ids)
        cs.gt_hit = candidate_hit(cs.path_candidates, cs.gt_answers) if cs.path_candidates else False
        cs.gt_hit_strict = strict_candidate_hit(cs.path_candidates, cs.gt_answers) if cs.path_candidates else False
        gt_stats = compute_match_stats(cs.path_candidates, cs.gt_answers)
        cs.gt_f1 = gt_stats['f1']

        # Flag for direct answer if no paths found
        if not cs.paths:
            cs.needs_direct_answer = True

    # Run graph traversal (CPU-bound) concurrently
    await asyncio.gather(*[asyncio.to_thread(_traverse_one, cs) for cs in active])

    dt = time.perf_counter() - _t0
    for cs in active:
        cs.stage_times["graph_traverse"] = dt / len(active)
    hits = sum(1 for cs in active if cs.gt_hit)
    strict_hits = sum(1 for cs in active if cs.gt_hit_strict)
    direct = sum(1 for cs in active if cs.needs_direct_answer)
    print(f"  Stage 5 (Graph traverse): {dt:.2f}s | GT={hits}/{len(active)} | strict={strict_hits}/{len(active)} | direct={direct}")


async def stage_6_diagnosis_retry(session, cases: List[CaseState]):
    """Safety net + Level 3-4 retry for candidate explosion."""
    _t0 = time.perf_counter()
    active = [cs for cs in cases if cs.active]
    if not active:
        return

    direct_count = sum(1 for cs in active if cs.needs_direct_answer)

    # ── Level 3: Re-prune relations for candidate explosion cases ──
    explosive = [cs for cs in active
                 if len(cs.paths) > CANDIDATE_THRESHOLD and not cs.needs_direct_answer]
    l3_count = 0
    for cs in explosive:
        # Try re-pruning with stricter prompt
        steps_for_prune = cs.steps
        if not steps_for_prune:
            continue
        try:
            prune_result, _ = await llm_prune_all_relations(
                session, cs.question, steps_for_prune, cs.step_candidates)
            # Build new step_relations from re-prune, keeping only top-2 per step
            new_step_relations = []
            for step in steps_for_prune:
                sn = step["step"]
                ranked = prune_result.get(sn, [])
                new_step_relations.append(set(ranked[:2]))  # stricter: max 2 relations
        except Exception:
            continue

        # Re-traverse with stricter relations
        bp_set = set(cs.breakpoints.values()) - {cs.anchor_idx, None}
        try:
            paths_new, max_depth_new, max_cov_new = relation_prior_expand(
                cs.anchor_idx, new_step_relations,
                cs.h_ids, cs.r_ids, cs.t_ids, cs.ents,
                explicit_targets=list(bp_set) if bp_set else None,
                max_hops=len(new_step_relations))
        except Exception:
            continue
        paths_new = prefer_breakpoint_hit_paths(
            paths_new, cs.breakpoints, cs.h_ids, cs.r_ids, cs.t_ids, cs.ents)

        # Rebuild candidates (last 2 hops only)
        last2_new = set()
        for p in paths_new:
            nodes = p.get("nodes", [])
            for n in nodes[-2:]:
                if n != cs.anchor_idx:
                    last2_new.add(n)

        new_candidates = []
        for node_idx in sorted(last2_new):
            if node_idx == cs.anchor_idx:
                continue
            name = cs.ents[node_idx] if 0 <= node_idx < len(cs.ents) else ""
            if is_cvt_like(name):
                for cvt_idx, _ in expand_through_cvt(node_idx, cs.h_ids, cs.r_ids, cs.t_ids, cs.ents):
                    if cvt_idx != cs.anchor_idx and 0 <= cvt_idx < len(cs.ents) and not is_cvt_like(cs.ents[cvt_idx]):
                        new_candidates.append(cs.ents[cvt_idx])
            else:
                new_candidates.append(name)
        seen_n = set()
        unique_n = []
        for c in new_candidates:
            nc = normalize(c)
            if len(nc) < 2 or not c.isascii():
                continue
            if nc not in seen_n:
                seen_n.add(nc)
                unique_n.append(c)

        if len(unique_n) <= CANDIDATE_THRESHOLD:
            cs.paths = paths_new
            cs.step_relations = new_step_relations
            cs.answer_candidates = unique_n
            cs.all_subgraph_nodes = all_nodes_new
            if paths_new:
                cs.max_depth = max(p.get("depth", 0) for p in paths_new)
                cs.max_cov = max(len(p.get("covered_steps", frozenset())) for p in paths_new)
            cs.path_candidates = _extract_path_candidates(cs.paths, cs.anchor_idx, cs.ents, cs.h_ids, cs.r_ids, cs.t_ids)
            cs.gt_hit = candidate_hit(cs.path_candidates, cs.gt_answers) if cs.path_candidates else False
            cs.gt_hit_strict = strict_candidate_hit(cs.path_candidates, cs.gt_answers) if cs.path_candidates else False
            cs.gt_f1 = compute_match_stats(cs.path_candidates, cs.gt_answers)['f1']
            l3_count += 1

    # ── Level 4: Anchor swap for persistent explosion ──
    still_explosive = [cs for cs in active
                       if len(cs.paths) > CANDIDATE_THRESHOLD and not cs.needs_direct_answer]
    l4_count = 0
    for cs in still_explosive:
        # Try a different anchor from NER top ents
        current_anchor = cs.anchor_name
        new_anchor = None
        new_anchor_idx = None
        for ent_name, gte_score in cs.ner_top_ents:
            if ent_name == current_anchor:
                continue
            # Resolve this entity
            from collections import defaultdict
            ner_name_map = defaultdict(list)
            for idx, name in enumerate(cs.ents):
                if name == ent_name:
                    ner_name_map[name].append(idx)
            if ner_name_map.get(ent_name):
                new_anchor = ent_name
                new_anchor_idx = ner_name_map[ent_name][0]
                break

        if new_anchor is None:
            continue

        # Swap anchor and re-traverse
        cs.anchor_name = new_anchor
        cs.anchor_idx = new_anchor_idx
        # Re-run pruning for new anchor
        try:
            prune_result, _ = await llm_prune_all_relations(
                session, cs.question, cs.steps, cs.step_candidates)
            new_step_relations = []
            for step in cs.steps:
                sn = step["step"]
                ranked = prune_result.get(sn, [])
                new_step_relations.append(set(ranked[:2]))
        except Exception:
            continue

        bp_set = set(cs.breakpoints.values()) - {cs.anchor_idx, None}
        try:
            paths_new, max_depth_new, max_cov_new = relation_prior_expand(
                cs.anchor_idx, new_step_relations,
                cs.h_ids, cs.r_ids, cs.t_ids, cs.ents,
                explicit_targets=list(bp_set) if bp_set else None,
                max_hops=len(new_step_relations))
        except Exception:
            continue
        paths_new = prefer_breakpoint_hit_paths(
            paths_new, cs.breakpoints, cs.h_ids, cs.r_ids, cs.t_ids, cs.ents)

        last2_new = set()
        for p in paths_new:
            nodes = p.get("nodes", [])
            for n in nodes[-2:]:
                if n != cs.anchor_idx:
                    last2_new.add(n)

        new_candidates = []
        for node_idx in sorted(last2_new):
            if node_idx == cs.anchor_idx:
                continue
            name = cs.ents[node_idx] if 0 <= node_idx < len(cs.ents) else ""
            if is_cvt_like(name):
                for cvt_idx, _ in expand_through_cvt(node_idx, cs.h_ids, cs.r_ids, cs.t_ids, cs.ents):
                    if cvt_idx != cs.anchor_idx and 0 <= cvt_idx < len(cs.ents) and not is_cvt_like(cs.ents[cvt_idx]):
                        new_candidates.append(cs.ents[cvt_idx])
            else:
                new_candidates.append(name)
        seen_n = set()
        unique_n = []
        for c in new_candidates:
            nc = normalize(c)
            if len(nc) < 2 or not c.isascii():
                continue
            if nc not in seen_n:
                seen_n.add(nc)
                unique_n.append(c)

        if len(unique_n) < len(cs.answer_candidates):
            cs.paths = paths_new
            cs.step_relations = new_step_relations
            cs.answer_candidates = unique_n
            cs.all_subgraph_nodes = {cs.anchor_idx}
            for p in paths_new:
                cs.all_subgraph_nodes.update(p.get("nodes", []))
            if paths_new:
                cs.max_depth = max(p.get("depth", 0) for p in paths_new)
                cs.max_cov = max(len(p.get("covered_steps", frozenset())) for p in paths_new)
            cs.path_candidates = _extract_path_candidates(cs.paths, cs.anchor_idx, cs.ents, cs.h_ids, cs.r_ids, cs.t_ids)
            cs.gt_hit = candidate_hit(cs.path_candidates, cs.gt_answers) if cs.path_candidates else False
            cs.gt_hit_strict = strict_candidate_hit(cs.path_candidates, cs.gt_answers) if cs.path_candidates else False
            cs.gt_f1 = compute_match_stats(cs.path_candidates, cs.gt_answers)['f1']
            l4_count += 1

    dt = time.perf_counter() - _t0
    for cs in active:
        cs.stage_times["retries"] = dt / len(active)
    hits_after = sum(1 for cs in active if cs.gt_hit)
    print(f"  Stage 6 (Safety net): {dt:.2f}s | {direct_count} direct | L3_reprune={l3_count} L4_anchor_swap={l4_count} | GT={hits_after}/{len(active)}")


async def stage_7_path_selection(session, cases: List[CaseState]):
    """Batch LLM path selection. Pre-filtered, single round with fallback."""
    _t0 = time.perf_counter()
    active = [cs for cs in cases if cs.active]

    # Compress paths
    for cs in active:
        breakpoint_indices = set(cs.breakpoints.values())
        cs.logical_paths = compress_paths(
            cs.paths, cs.ents, cs.rels, cs.anchor_idx, breakpoint_indices) if cs.paths else []

    cases_with_paths = [cs for cs in active if cs.logical_paths]
    if not cases_with_paths:
        return

    # Single round: pre-filter and select
    prompts = []
    for cs in cases_with_paths:
        # Pre-filter: deduplicate by relation chain prefix, keep max 6
        seen_chains = set()
        filtered_indices = []
        for i, lp in enumerate(cs.logical_paths):
            # Use first 2 relations as chain signature for diversity
            chain_sig = tuple(lp["rel_chain"][:2])
            if chain_sig in seen_chains and len(seen_chains) >= 3:
                continue  # Skip duplicate chains if we already have 3+ diverse ones
            seen_chains.add(chain_sig)
            filtered_indices.append(i)
            if len(filtered_indices) >= 10:
                break
        # If filtering removed too many, keep original top-10
        if len(filtered_indices) < 2:
            filtered_indices = list(range(min(10, len(cs.logical_paths))))

        cs._idx_map = {}
        path_lines = []
        for display_num, actual_idx in enumerate(filtered_indices, 1):
            lp = cs.logical_paths[actual_idx]
            ncands = len(lp.get("candidates", []))
            has_ep = "→ endpoint" if lp.get("endpoint") else ""
            path_lines.append(f"{display_num}. {lp['readable']}  ({ncands} candidates{has_ep})")
            cs._idx_map[display_num] = actual_idx

        # Build decomposition context for path selection
        decomp_context = ""
        if hasattr(cs, 'rewritten_question') and cs.rewritten_question:
            decomp_context += f"Rewritten question: {cs.rewritten_question}\n"
        if cs.answer_type:
            decomp_context += f"Answer type: {cs.answer_type}\n"
        if cs.steps:
            hop_descs = []
            for s in cs.steps:
                kw = s.get('keyword', '')
                defn = s.get('definition', '')
                if kw or defn:
                    hop_descs.append(f"  - {kw}: {defn}" if defn else f"  - {kw}")
            if hop_descs:
                decomp_context += "Expected relation pattern:\n" + "\n".join(hop_descs) + "\n"
        endpoint_names = [v for v in cs.breakpoints.values() if v is not None and v != cs.anchor_idx]
        if endpoint_names:
            ep_strs = [cs.ents[i] for i in endpoint_names if 0 <= i < len(cs.ents)]
            if ep_strs:
                decomp_context += f"Endpoint constraints: {', '.join(ep_strs)}\n"

        select_prompt = f"""Select reasoning paths for this question.

Question: {cs.question}
{decomp_context}
Paths (sorted by relevance, top is best):
{chr(10).join(path_lines)}

Task:
Evaluate each path against the expected relation pattern and assign it to one of three categories:

- Strong Match: The relation chain closely matches the intended reasoning structure.
- Partial but Useful: The path does not fully match the intended structure, but captures an important part of the reasoning and may still help identify or verify the answer.
- Mismatch: The path is semantically off-track and should not be selected unless no better alternatives exist.

Selection rules:
1. Prefer Strong Match paths first.
2. If fewer than 2 Strong Match paths exist, add Partial but Useful paths until you have 2 to 4 total.
3. Prefer semantic diversity among the selected paths.
4. Prefer paths that reach different candidate sets.
5. Avoid Mismatch paths unless there are no better alternatives.

Important: It is acceptable that only one path is a Strong Match. In that case, you must supplement it with the best Partial but Useful path(s), rather than selecting a Mismatch path.

<analysis>
Path 1: [Strong Match / Partial / Mismatch] — one short reason.
Path 2: [Strong Match / Partial / Mismatch] — one short reason.
...
Selected: explain in 1-2 sentences why these paths were chosen.
</analysis>
<selected>comma-separated path numbers</selected>"""

        prompts.append([
            {"role": "system", "content": "You select diverse reasoning paths for graph QA. Always select 2-4 paths with different relation types. Output <analysis> and <selected>."},
            {"role": "user", "content": select_prompt},
        ])
        cs.path_select_prompt = select_prompt

    responses = await batch_call_llm(session, prompts, max_tokens=400)

    for cs, raw in zip(cases_with_paths, responses):
        sel_text = extract_xml_tag(raw or "", "selected") or ""
        sel_indices = []
        for m in re.finditer(r'\d+', sel_text):
            display_num = int(m.group())
            if display_num in cs._idx_map:
                sel_indices.append(cs._idx_map[display_num])

        cs.selected_paths = list(dict.fromkeys(sel_indices))
        cs.path_select_response = raw or ""

    # Ensure minimum 2 selected paths (fallback to top patterns)
    for cs in cases_with_paths:
        if len(cs.selected_paths) < 2 and cs.logical_paths:
            for i in range(min(3, len(cs.logical_paths))):
                if i not in cs.selected_paths:
                    cs.selected_paths.append(i)
                if len(cs.selected_paths) >= 2:
                    break

    dt = time.perf_counter() - _t0
    for cs in active:
        cs.stage_times["path_select"] = dt / len(active)
    sel_counts = [len(cs.selected_paths) for cs in cases_with_paths]
    print(f"  Stage 7 (Path select): {dt:.2f}s | selected={sel_counts}")


async def stage_8_answer_reasoning(session, cases: List[CaseState]):
    """Batch LLM answer reasoning with direct-answer fallback for failed graph cases."""
    _t0 = time.perf_counter()

    # Split into normal cases and direct-answer (safety net) cases
    active = [cs for cs in cases if cs.active]
    normal_cases = [cs for cs in active if cs.selected_paths and not cs.needs_direct_answer]
    direct_cases = [cs for cs in active if cs.needs_direct_answer]

    # --- Normal cases: graph-based reasoning ---
    prompts = []
    cases_with_triples = []
    for cs in normal_cases:
        selected_pattern_objs = [cs.logical_paths[i] for i in cs.selected_paths if i < len(cs.logical_paths)]
        if not selected_pattern_objs:
            continue
        breakpoint_indices = set(cs.breakpoints.values())
        selected_pattern_objs = list(selected_pattern_objs)
        selected_pattern_objs.extend(build_endpoint_rescue_patterns(
            cs.paths, selected_pattern_objs, cs.ents, cs.rels, cs.anchor_idx, breakpoint_indices,
        ))
        pat_evidence = build_pattern_evidence_triples(
            selected_pattern_objs, cs.ents, cs.rels, cs.h_ids, cs.r_ids, cs.t_ids, cs.anchor_idx,
            max_grouped_lines=120)
        cs.num_triples = sum(len(pe.triples) for pe in pat_evidence.values())
        # Bug fix: Don't skip cases with empty triples if we have candidates from Stage 7
        # The candidates might already contain the answer (e.g., Case [27]: Blue Ivy is in candidates)
        if not pat_evidence and not cs.answer_candidates:
            continue
        cases_with_triples.append(cs)

        # Bug fix: When pat_evidence is empty but we have candidates, use candidates as evidence
        if not pat_evidence:
            # Generate a candidate-based prompt when triples are empty
            candidates_str = ", ".join(cs.answer_candidates[:20]) if cs.answer_candidates else "No candidates found"
            selected_paths_str = ", ".join([f"Path {i+1}: {cs.logical_paths[i].get('readable', '')}"
                                           for i in cs.selected_paths if i < len(cs.logical_paths)])
            pattern_text = f"""SELECTED PATHS:
{selected_paths_str}

ANSWER CANDIDATES FROM GRAPH TRAVERSAL:
{candidates_str}

Note: Graph triples could not be extracted, but answer candidates are available from the path traversal."""
        else:
            pattern_text = format_pattern_evidence(pat_evidence)

        answer_type_hint = f"\nAnswer type: {cs.answer_type}" if cs.answer_type else ""
        rewritten_hint = ""
        if hasattr(cs, 'rewritten_question') and cs.rewritten_question and cs.rewritten_question != cs.question:
            rewritten_hint = f"\nRewritten: {cs.rewritten_question}"

        reason_prompt = f"""QUESTION: {cs.question}
{answer_type_hint}{rewritten_hint}

GRAPH EVIDENCE:
{pattern_text}

━━━ ENTITY-CENTRIC REASONING ━━━

STEP 1 — QUESTION UNDERSTANDING
Answer type: what kind of entity the question seeks (person, country, event, year, etc.).
Explicit constraints: conditions directly stated (dates, locations, superlatives, quantities).
Implicit constraint heuristic (pick one):
  - UNIQUE ROLE ("the governor/president/leader") without time qualifier → MOST RECENT only
  - EVENTS/ACHIEVEMENTS ("wins/championships/movies/albums") → return ALL matching
  - ATTRIBUTES/PROPERTIES ("languages/religions/currency") → return ALL that apply
  - GROUP MEMBERSHIP ("countries in / states in / members of") → return ALL matching members

STEP 2 — PER-CANDIDATE CONSTRAINT VERIFICATION
For EVERY candidate entity found in GRAPH EVIDENCE (including intermediate nodes and CVT attribute values), check:

2a. TYPE MATCH: Does this candidate's type match the answer type?
  Format: entity → KEEP (type matches) / REMOVE (type mismatch, state why)

2b. EXPLICIT CONSTRAINT CHECK (for type-matched candidates):
  For each explicit constraint from the question, check against graph evidence:
  Format: Constraint "[description]": entity1 PASS (evidence: ...), entity2 FAIL (evidence: ...)
  If graph evidence does NOT show failure → KEEP the entity.
  No dates in evidence → do NOT filter by time.

2c. IMPLICIT CONSTRAINT CHECK (for candidates passing 2b):
  Apply the heuristic from Step 1:
  - If unique role → pick MOST RECENT by graph dates; no dates → output ALL
  - If events/attributes/group → ALL candidates PASS

  Format: entity → PASS / FAIL (one reason)

STEP 3 — OUTPUT DECISION
Collect ALL candidates that PASS steps 2a + 2b + 2c.
- 1 entity → output it
- Multiple + unique role heuristic → pick MOST RECENT by graph dates; no dates → output ALL
- Multiple + events/attributes/group → output ALL
- Zero entities passed → <answer>None</answer>

RULES:
- Graph evidence ONLY. No outside knowledge.
- ANY entity in GRAPH EVIDENCE is a valid answer — including intermediate nodes and CVT attribute values.
- NEVER decide answer count before completing Step 2. Evaluate ALL candidates first.
- Singular/plural phrasing alone does NOT determine answer count.
- "When" questions → output event NAME (e.g. "2014 World Series"), NOT raw timestamp.
- Answer MUST be an exact entity string from GRAPH EVIDENCE or CANDIDATE ENTITIES. Never output a bare number, year, or timestamp — use the full entity name.
- If unsure whether an entity satisfies a constraint → KEEP it.
- Over-output is better than discarding valid answers.
- For "group/organization that fought in/participated in" questions, political entities (countries, confederacies, alliances) are valid answer types — not only military units.

━━━ EXAMPLES (illustrate METHOD only) ━━━

Example A — TYPE FILTER + unique role:
Q: "Who is the president of France?"
2a (type person): KEEP [Macron, Hollande]. REMOVE [France(country), President(role), 2017-05-14(date)].
2b: No explicit time constraint.
2c (unique role, most recent): Macron PASS (term start 2017-05-14), Hollande FAIL (term start 2012-05-15).
3: Macron.

Example B — EVENTS → return ALL:
Q: "What movies did the actor who played Forrest Gump star in?"
2a (type movie): KEEP [Forrest Gump, Saving Private Ryan, Cast Away]. REMOVE [Tom Hanks(person), Actor(role), 1994(year)].
2b: No additional constraints beyond basic fact.
2c (events → ALL): All PASS.
3: Forrest Gump | Saving Private Ryan | Cast Away.

━━━ OUTPUT FORMAT ━━━
<reasoning>
Step 1: answer type, explicit constraints, implicit heuristic.
Step 2a: type match per candidate.
Step 2b: explicit constraint check per candidate.
Step 2c: implicit constraint check per candidate.
Step 3: passing set and output decision.
</reasoning>
<answer>\\boxed{{exact entity}}</answer>
Multiple: <answer>\\boxed{{e1}} \\boxed{{e2}}</answer>
No valid entity: <answer>None</answer>
NO text after </answer> tag."""

        cs.llm_reasoning_prompt = reason_prompt
        if REASON_STYLE == "entity-lite":
            # ── Lite version for 9B models: 2-step high-recall, no None ──
            n_cand = len(cs.answer_candidates) if cs.answer_candidates else 0
            cand_limit = min(n_cand, 60)
            cand_names = list(dict.fromkeys(cs.answer_candidates[:cand_limit])) if cs.answer_candidates else []
            cand_list = "\n".join(f"  - {c}" for c in cand_names) if cand_names else "  (see graph evidence above)"
            atype = cs.answer_type or "(infer from question)"
            reason_prompt = f"""QUESTION: {cs.question}
{answer_type_hint}{rewritten_hint}

GRAPH EVIDENCE:
{pattern_text}

CANDIDATE ENTITIES:
{cand_list}

━━━ ANSWER SELECTION ━━━

STEP 1 — Identify what the question needs.
Answer type: {atype}
Key constraints from the question: list them briefly.
Cardinality: Does the question ask for ONE specific role (e.g. "the governor", "the president")? If yes → pick the most recent one with dates in evidence. Otherwise → output ALL matching candidates.

STEP 2 — Evaluate each candidate.
For each candidate in CANDIDATE ENTITIES:
- KEEP if type roughly matches answer type.
- KEEP if no evidence contradicts it.
- REMOVE only if graph evidence explicitly shows it is WRONG type or contradicts a constraint.

RULES:
- Use graph evidence only. No outside knowledge.
- Do NOT remove a candidate for missing evidence. Remove only if evidence explicitly contradicts.
- Do NOT output None. If uncertain, output the best matching candidate(s).
- Prefer over-output to under-output.
- "When" questions → output event NAME (e.g. "2014 World Series"), NOT raw timestamp.
- Answer MUST be an exact entity string from GRAPH EVIDENCE or CANDIDATE ENTITIES.
- Copy entity strings exactly. Never output Freebase IDs (m.0xxx).
- Over-output is better than discarding valid answers.

━━━ OUTPUT ━━━
<reasoning>
Step 1: answer type, constraints, cardinality decision.
Step 2: per-candidate KEEP/REMOVE with one reason each.
</reasoning>
<answer>\\boxed{{exact entity}}</answer>
Multiple: <answer>\\boxed{{e1}} \\boxed{{e2}}</answer>
NO text after </answer> tag."""
            cs.llm_reasoning_prompt = reason_prompt
            prompts.append([
                {"role": "system", "content": "You are a precise graph QA assistant. Always output at least one entity from CANDIDATE ENTITIES. Never output None. Copy entity strings exactly."},
                {"role": "user", "content": reason_prompt},
            ])
        elif REASON_STYLE == "check":
            # ── Checklist v2: mechanical checks + type/granularity verification ──
            n_cand = len(cs.answer_candidates) if cs.answer_candidates else 0
            cand_limit = min(n_cand, 60)
            cand_names = list(dict.fromkeys(cs.answer_candidates[:cand_limit])) if cs.answer_candidates else []
            cand_list = "\n".join(f"  - {c}" for c in cand_names) if cand_names else "  (see graph evidence above)"
            atype = cs.answer_type or "(infer from question)"
            type_check_lines = "\n".join(
                f"  {c} -> TYPE: [matches {atype}?] -> KEEP / REMOVE"
                for c in cand_names[:25]
            )
            reason_prompt = f"""QUESTION: {cs.question}
{answer_type_hint}{rewritten_hint}

GRAPH EVIDENCE:
{pattern_text}

CANDIDATE ENTITIES:
{cand_list}

=== CHECKLIST ===

□ ANSWER TYPE: {atype}

□ TYPE FILTER — check each candidate's type matches answer type:
{type_check_lines}

□ CONSTRAINTS from question:
  - ____
  Per kept candidate: PASS / FAIL

□ GRANULARITY CHECK:
  Any candidate is a parent/child of another (e.g. city vs stadium, country vs sport)?
  -> Pick the one matching question specificity.

□ CARDINALITY (pick one):
  [ ] unique role ("the X", no time) -> MOST RECENT only
  [ ] events / achievements -> ALL
  [ ] attributes / properties -> ALL
  [ ] group membership -> ALL
  [ ] none of the above -> ALL remaining

□ KEPT AFTER FILTERS: [list here]

RULES:
- Graph evidence only. No outside knowledge.
- No dates in evidence -> do NOT remove by time.
- "When" -> event NAME, not raw year.
- Answer MUST be an exact entity string from GRAPH EVIDENCE or CANDIDATE ENTITIES. Never output a bare number, year, or timestamp — use the full entity name.
- Answer may be at intermediate hop.
- In doubt -> KEEP.
- NEVER output "None". If all removed, pick most specific candidate.

ANSWER STRING (copy exactly from CANDIDATE ENTITIES above):
____

<answer>\\boxed{{exact entity}}</answer>
Multiple: <answer>\\boxed{{e1}} \\boxed{{e2}}</answer>
NO text after </answer>."""
            cs.llm_reasoning_prompt = reason_prompt
            prompts.append([
                {"role": "system", "content": "You are a graph fact-checker. Fill the checklist mechanically. One word per judgement (KEEP/REMOVE, PASS/FAIL). No paragraphs. Always output an answer. Copy entity strings exactly from CANDIDATE ENTITIES."},
                {"role": "user", "content": reason_prompt},
            ])
        elif REASON_STYLE == "ecot":
            # ── Evidence-COT hybrid: checklist structure + 1-sentence evidence per item ──
            n_cand = len(cs.answer_candidates) if cs.answer_candidates else 0
            cand_limit = min(n_cand, 60)
            cand_names = list(dict.fromkeys(cs.answer_candidates[:cand_limit])) if cs.answer_candidates else []
            cand_list = "\n".join(f"  - {c}" for c in cand_names) if cand_names else "  (see graph evidence above)"
            atype = cs.answer_type or "(infer from question)"
            type_check_lines = "\n".join(
                f"  {c} -> [{atype}?] KEEP/REMOVE because [one fact from graph]"
                for c in cand_names[:25]
            )
            reason_prompt = f"""QUESTION: {cs.question}
{answer_type_hint}{rewritten_hint}

GRAPH EVIDENCE:
{pattern_text}

CANDIDATE ENTITIES:
{cand_list}

=== STRUCTURED EVIDENCE CHECK ===

1. ANSWER TYPE: {atype}

2. TYPE + EVIDENCE CHECK (one fact per candidate):
{type_check_lines}

3. CONSTRAINT CHECK:
   Question requires: ____
   Kept candidates that satisfy it: ____

4. GRANULARITY: If any kept candidate is a parent of another (city vs venue), pick the specific one.

5. CARDINALITY: unique role -> MOST RECENT | events/attributes -> ALL | in doubt -> ALL

6. KEPT: [list final kept candidates here]

RULES:
- Graph evidence only. No outside knowledge.
- No dates -> do NOT filter by time.
- "When" -> event NAME (e.g. "2014 World Series"), NOT raw timestamp or year.
- NEVER output "None". If all removed, pick most specific kept.
- Answer MUST be an exact entity string from GRAPH EVIDENCE or CANDIDATE ENTITIES. Never output a bare number, year, or timestamp — use the full entity name.

<answer>\\boxed{{exact entity from CANDIDATE ENTITIES}}</answer>
Multiple: <answer>\\boxed{{e1}} \\boxed{{e2}}</answer>
NO text after </answer>."""
            cs.llm_reasoning_prompt = reason_prompt
            prompts.append([
                {"role": "system", "content": "You are a precise graph QA checker. For each candidate, give ONE fact from graph evidence. Keep total under 10 lines. Answer MUST be an exact string from the candidate list. No outside knowledge."},
                {"role": "user", "content": reason_prompt},
            ])
        elif REASON_STYLE == "entity":
            # ── Entity-centric 3-step: basic fact → constraint check → output ──
            n_cand = len(cs.answer_candidates) if cs.answer_candidates else 0
            cand_limit = min(n_cand, 60)
            cand_names = list(dict.fromkeys(cs.answer_candidates[:cand_limit])) if cs.answer_candidates else []
            cand_list = "\n".join(f"  - {c}" for c in cand_names) if cand_names else "  (see graph evidence above)"
            atype = cs.answer_type or "(infer from question)"

            # Extract relation names from selected patterns for basic-fact disambiguation
            rel_names = set()
            for i in cs.selected_paths:
                if i < len(cs.logical_paths):
                    lp = cs.logical_paths[i]
                    for hop in lp.get("hops", []):
                        r = hop.get("relation", "")
                        if r:
                            rel_names.add(r)
            rel_list = ", ".join(sorted(rel_names)[:12]) if rel_names else "(see evidence)"

            reason_prompt = f"""QUESTION: {cs.question}
{answer_type_hint}{rewritten_hint}

GRAPH EVIDENCE:
{pattern_text}

CANDIDATE ENTITIES (traversal endpoints):
{cand_list}

NOTE: Any entity name appearing in GRAPH EVIDENCE (including intermediate nodes, CVT attribute values, and ← also lines) is also a valid answer. Do NOT restrict answers to the list above only.

━━━ ENTITY-CENTRIC REASONING ━━━

STEP 1 — QUESTION UNDERSTANDING
Answer type: what kind of entity the question seeks (person, country, event, year, etc.).
Available relations in evidence: {rel_list}
Explicit constraints from the question (time, location, superlatives, quantities, etc.): ____
Implicit constraint heuristic (pick one):
  - UNIQUE ROLE ("the governor/president/leader") without time qualifier → MOST RECENT only
  - EVENTS/ACHIEVEMENTS ("wins/championships/movies/albums") → return ALL matching
  - ATTRIBUTES/PROPERTIES ("languages/religions/currency") → return ALL that apply
  - GROUP MEMBERSHIP ("countries in / states in / members of") → return ALL matching members

STEP 2 — PER-CANDIDATE CONSTRAINT VERIFICATION
For EVERY candidate entity found in GRAPH EVIDENCE (including intermediate nodes and CVT attribute values), check:

2a. TYPE MATCH: Does this candidate's type match the answer type?
  Format: entity → KEEP (type matches) / REMOVE (type mismatch, state why)

2b. EXPLICIT CONSTRAINT CHECK (for type-matched candidates):
  For each explicit constraint from the question, check against graph evidence:
  Format: Constraint "[description]": entity1 PASS (evidence: ...), entity2 FAIL (evidence: ...)
  If graph evidence does NOT show failure → KEEP the entity.
  No dates in evidence → do NOT filter by time.

2c. IMPLICIT CONSTRAINT CHECK (for candidates passing 2b):
  Apply the heuristic from Step 1:
  - If unique role → pick MOST RECENT by graph dates; no dates → output ALL
  - If events/attributes/group → ALL candidates PASS

  Format: entity → PASS / FAIL (one reason)

STEP 3 — OUTPUT DECISION
Collect ALL candidates that PASS steps 2a + 2b + 2c.
- 1 entity → output it
- Multiple + unique role heuristic → pick MOST RECENT by graph dates; no dates → output ALL
- Multiple + events/attributes/group → output ALL
- Zero entities passed → <answer>None</answer>

RULES:
- Graph evidence ONLY. No outside knowledge.
- ANY entity in GRAPH EVIDENCE is a valid answer — including intermediate nodes, CVT attribute values, and entities in ← also lines.
- NEVER decide answer count before completing Step 2. Evaluate ALL candidates first.
- Singular/plural phrasing alone does NOT determine answer count.
- "When" questions → output event NAME (e.g. "2014 World Series"), NOT raw timestamp.
- Answer MUST be an exact entity string from GRAPH EVIDENCE or CANDIDATE ENTITIES. Never output a bare number, year, or timestamp — use the full entity name.
- If unsure whether an entity satisfies a constraint → KEEP it.
- Over-output is better than discarding valid answers.
- For "group/organization that fought in/participated in" questions, political entities (countries, confederacies, alliances) are valid answer types — not only military units.

━━━ EXAMPLES (illustrate METHOD only) ━━━

Example A — TYPE FILTER + unique role:
Q: "Who is the president of France?"
2a (type person): KEEP [Macron, Hollande]. REMOVE [France(country), President(role), 2017-05-14(date)].
2b: No explicit time constraint.
2c (unique role, most recent): Macron PASS (term start 2017-05-14), Hollande FAIL (term start 2012-05-15).
3: Macron.

Example B — EVENTS → return ALL:
Q: "What movies did the actor who played Forrest Gump star in?"
2a (type movie): KEEP [Forrest Gump, Saving Private Ryan, Cast Away]. REMOVE [Tom Hanks(person), Actor(role), 1994(year)].
2b: No additional constraints beyond basic fact.
2c (events → ALL): All PASS.
3: Forrest Gump | Saving Private Ryan | Cast Away.

━━━ OUTPUT FORMAT ━━━
<reasoning>
Step 1: answer type, explicit constraints, implicit heuristic.
Step 2a: type match per candidate.
Step 2b: explicit constraint check per candidate.
Step 2c: implicit constraint check per candidate.
Step 3: passing set and output decision.
</reasoning>
<answer>\\boxed{{exact entity}}</answer>
Multiple: <answer>\\boxed{{e1}} \\boxed{{e2}}</answer>
No valid entity: <answer>None</answer>
NO text after </answer> tag."""
            cs.llm_reasoning_prompt = reason_prompt
            prompts.append([
                {"role": "system", "content": "You are a precise graph QA system using per-candidate constraint verification. First identify answer type and constraints, then check EACH candidate against type + explicit + implicit constraints, then output ALL passing candidates. Any entity in GRAPH EVIDENCE is valid, not just CANDIDATE ENTITIES. NEVER decide answer count before checking all candidates. Over-output is better than discarding."},
                {"role": "user", "content": reason_prompt},
            ])
        else:
            prompts.append([
                {"role": "system", "content": "You are a precise graph QA system using per-candidate constraint verification. First identify answer type and constraints, then check EACH candidate against type + explicit + implicit constraints, then output ALL passing candidates. Any entity in GRAPH EVIDENCE is valid, not just CANDIDATE ENTITIES. NEVER decide answer count before checking all candidates. Over-output is better than discarding."},
                {"role": "user", "content": reason_prompt},
            ])

    # --- Direct answer cases: parametric-only reasoning ---
    for cs in direct_cases:
        prompt = f"""QUESTION: {cs.question}

No reliable graph paths were found for this question. Use your parametric knowledge to answer directly.

<reasoning>One short sentence.</reasoning>
<answer>\\boxed{{exact entity}}</answer>"""
        cs.llm_reasoning_prompt = prompt
        prompts.append([
            {"role": "system", "content": "You are a precise QA system over Freebase (circa 2015). Answer the question directly using your knowledge. Output <reasoning> and <answer> XML tags."},
            {"role": "user", "content": prompt},
        ])

    # Batch all prompts together (normal + direct)
    if prompts:
        responses = await batch_call_llm(session, prompts, max_tokens=2400)

        all_answered = cases_with_triples + direct_cases
        for cs, raw in zip(all_answered, responses):
            cs.llm_reasoning_full = raw or ""
            llm_preds = []
            ans_match = re.search(r'<answer>(.*?)</answer>', raw or "", re.DOTALL)
            if ans_match:
                boxed = re.findall(r'\\boxed\{([^}]+)\}', ans_match.group(1))
                if boxed:
                    llm_preds = [b.strip() for b in boxed]
                    cs.llm_answer = " | ".join(llm_preds)
                else:
                    cs.llm_answer = ans_match.group(1).strip()
                    llm_preds = [cs.llm_answer]
            else:
                lines = [l.strip() for l in (raw or "").strip().split('\n') if l.strip()]
                cs.llm_answer = lines[-1] if lines else ""
                llm_preds = [cs.llm_answer]
            cs.llm_hit = candidate_hit(llm_preds, cs.gt_answers)
            llm_stats = compute_match_stats(llm_preds, cs.gt_answers)
            cs.llm_f1 = llm_stats['f1']
            cs.llm_precision = llm_stats['precision']
            cs.llm_recall = llm_stats['recall']

    # ── Front-end defense: retry if answer is a raw year/timestamp instead of entity ──
    retry_cases = []
    retry_prompts = []
    for cs in cases_with_triples:
        if not cs.answer_candidates or not cs.llm_answer:
            continue
        preds = [p.strip() for p in cs.llm_answer.split(" | ")]
        invalid_preds = []
        for p in preds:
            # Only retry if answer looks like a raw year/timestamp (not a real entity)
            if re.fullmatch(r'\d{4}(-\d{2}-\d{2}.*|-08:00)?', p):
                invalid_preds.append(p)
                continue
            # Skip retry for anything else — it's likely a valid entity from evidence tree
        if invalid_preds:
            retry_cases.append(cs)
            cand_str = ", ".join(cs.answer_candidates[:20])
            # P4: detect if invalid answer is a raw year for a "when" question
            is_year_retry = any(re.fullmatch(r'\d{4}(-\d{2}-\d{2}.*|-08:00)?', p) for p in invalid_preds)
            year_hint = ""
            if is_year_retry:
                year_hint = """
NOTE: You output a raw year/timestamp. The question asks "when" — the answer must be the
EVENT ENTITY NAME (e.g. "2008 NBA Finals"), not just the year number. Pick the event entity."""
            retry_prompts.append([
                {"role": "system", "content": "Your previous answer was NOT a valid entity from the graph. You MUST pick from the candidate list. Use exact entity strings only — never raw years or timestamps."},
                {"role": "user", "content": f"""QUESTION: {cs.question}

Your previous answer was: {cs.llm_answer}
This is NOT a valid entity from the graph.

VALID CANDIDATES (pick one or more exact strings):
{cand_str}
{year_hint}
Pick the candidate that best answers the question. Output ONLY the exact candidate string.

<answer>\\boxed{{candidate}}</answer>"""},
            ])

    # P2: batch retry outside the loop
    if retry_cases:
        retry_responses = await batch_call_llm(session, retry_prompts, max_tokens=400)
        for cs, raw in zip(retry_cases, retry_responses):
            ans_match = re.search(r'<answer>(.*?)</answer>', raw or "", re.DOTALL)
            if ans_match:
                boxed = re.findall(r'\\boxed\{([^}]+)\}', ans_match.group(1))
                if boxed:
                    new_answer = " | ".join(b.strip() for b in boxed)
                else:
                    new_answer = ans_match.group(1).strip()
            else:
                lines = [l.strip() for l in (raw or "").strip().split('\n') if l.strip()]
                new_answer = lines[-1] if lines else cs.llm_answer
            cs.llm_answer = new_answer
            cs.llm_reasoning_full += f"\n\n[FRONTEND RETRY] Original: invalid. Retry answer: {new_answer}"
            llm_preds = [p.strip() for p in new_answer.split(" | ")]
            cs.llm_hit = candidate_hit(llm_preds, cs.gt_answers)
            llm_stats = compute_match_stats(llm_preds, cs.gt_answers)
            cs.llm_f1 = llm_stats['f1']
            cs.llm_precision = llm_stats['precision']
            cs.llm_recall = llm_stats['recall']

    # ── Path reselect retry: if LLM says "None", try different paths ──
    none_cases = [cs for cs in cases_with_triples
                  if cs.llm_answer and cs.llm_answer.strip().lower() in ("none", "n/a", "null", "")
                  and cs.logical_paths and len(cs.logical_paths) > len(cs.selected_paths)]
    if none_cases:
        retry_prompts = []
        for cs in none_cases:
            prev_sel = set(cs.selected_paths)
            remaining = [i for i in range(len(cs.logical_paths)) if i not in prev_sel]
            if not remaining:
                continue
            # Re-select from remaining paths: pick top diverse ones not previously selected
            seen_chains = set()
            new_indices = []
            for i in remaining:
                lp = cs.logical_paths[i]
                chain_sig = tuple(lp.get("rel_chain", [])[:2])
                if chain_sig in seen_chains and len(seen_chains) >= 3:
                    continue
                seen_chains.add(chain_sig)
                new_indices.append(i)
                if len(new_indices) >= 4:
                    break
            if len(new_indices) < 1:
                new_indices = remaining[:3]
            cs.selected_paths = new_indices
            cs._path_reselect_retry = True

            # Rebuild evidence with new paths
            selected_pattern_objs = [cs.logical_paths[i] for i in new_indices if i < len(cs.logical_paths)]
            breakpoint_indices = set(cs.breakpoints.values())
            selected_pattern_objs.extend(build_endpoint_rescue_patterns(
                cs.paths, selected_pattern_objs, cs.ents, cs.rels, cs.anchor_idx, breakpoint_indices,
            ))
            pat_evidence = build_pattern_evidence_triples(
                selected_pattern_objs, cs.ents, cs.rels, cs.h_ids, cs.r_ids, cs.t_ids, cs.anchor_idx,
                max_grouped_lines=120)
            if pat_evidence:
                pattern_text = format_pattern_evidence(pat_evidence)
            else:
                candidates_str = ", ".join(cs.answer_candidates[:20]) if cs.answer_candidates else "No candidates"
                pattern_text = f"ANSWER CANDIDATES FROM GRAPH TRAVERSAL:\n{candidates_str}"

            answer_type_hint = f"\nAnswer type: {cs.answer_type}" if cs.answer_type else ""
            rewritten_hint = ""
            if hasattr(cs, 'rewritten_question') and cs.rewritten_question and cs.rewritten_question != cs.question:
                rewritten_hint = f"\nRewritten: {cs.rewritten_question}"

            retry_prompt = f"""QUESTION: {cs.question}
{answer_type_hint}{rewritten_hint}

GRAPH EVIDENCE (re-selected paths — previous paths had insufficient evidence):
{pattern_text}

Previous reasoning found NO valid answer. These are ALTERNATIVE paths. Find the answer here.

<evidence_summary>
Brief evaluation of new paths.
</evidence_summary>
<answer>\\boxed{{exact entity}}</answer>
Multiple: <answer>\\boxed{{e1}} \\boxed{{e2}}</answer>
NO text after </answer>."""
            retry_prompts.append([
                {"role": "system", "content": "You are a precise graph QA system. Previous paths had insufficient evidence. These are NEW alternative paths. Find the answer from the graph evidence. ALWAYS answer. Exact graph strings only."},
                {"role": "user", "content": retry_prompt},
            ])

        if retry_prompts:
            retry_responses = await batch_call_llm(session, retry_prompts, max_tokens=2000)
            for cs, raw in zip([c for c in none_cases if hasattr(c, '_path_reselect_retry')], retry_responses):
                ans_match = re.search(r'<answer>(.*?)</answer>', raw or "", re.DOTALL)
                if ans_match:
                    boxed = re.findall(r'\\boxed\{([^}]+)\}', ans_match.group(1))
                    if boxed:
                        new_answer = " | ".join(b.strip() for b in boxed)
                    else:
                        new_answer = ans_match.group(1).strip()
                else:
                    lines = [l.strip() for l in (raw or "").strip().split('\n') if l.strip()]
                    new_answer = lines[-1] if lines else cs.llm_answer
                cs.llm_reasoning_full += f"\n\n[PATH RESELECT RETRY] New paths selected. Retry answer: {new_answer}"
                cs.llm_answer = new_answer
                llm_preds = [p.strip() for p in new_answer.split(" | ")]
                cs.llm_hit = candidate_hit(llm_preds, cs.gt_answers)
                llm_stats = compute_match_stats(llm_preds, cs.gt_answers)
                cs.llm_f1 = llm_stats['f1']
                cs.llm_precision = llm_stats['precision']
                cs.llm_recall = llm_stats['recall']
                if hasattr(cs, '_path_reselect_retry'):
                    del cs._path_reselect_retry

    # Mark all cases as completed
    for cs in cases:
        if cs.active:
            cs.active = False
            cs.stage_times.setdefault("llm_reasoning", time.perf_counter() - _t0)

    dt = time.perf_counter() - _t0
    for cs in active:
        cs.stage_times["llm_reasoning"] = dt / max(len(active), 1)
    llm_hits = sum(1 for cs in cases_with_triples + direct_cases if cs.llm_hit)
    print(f"  Stage 8 (Answer reasoning): {dt:.2f}s | LLM={llm_hits}/{len(cases_with_triples)}+{len(direct_cases)}direct")


def _case_state_to_result_dict(cs: CaseState) -> Dict[str, Any]:
    """Convert CaseState to result dict matching run_case() output format."""
    logical_paths = cs.logical_paths or []
    return {
        "case_id": cs.case_id,
        "question": cs.question,
        "gt_answers": cs.gt_answers,
        "decomposition_prompt": cs.decomp_prompt_formatted or (CHAIN_PROMPT if cs.use_ner else DECOMP_PROMPT),
        "decomposition_question": cs.decomp_question,
        "decomposition": cs.decomp_raw,
        "decomp_retry": cs.decomp_retry,
        "decomp_reflect_raw": cs.decomp_reflect_raw or "",
        "decomp_retry_reason": cs.decomp_retry_reason or "",
        "stage_1a_raw": cs._1a_raw or "",
        "stage_1a_prompt": cs._1a_prompt or "",
        "stage_1a_anchor": cs._1a_anchor or "",
        "stage_1a_endpoints": cs._1a_endpoints or "",
        "stage_1a_interpretation": cs._1a_interpretation or "",
        "stage_1a_answer_type": cs._1a_answer_type or "",
        "stage_1a_rewritten": cs._1a_rewritten or "",
        "steps_parsed": cs.steps,
        "anchor_idx": cs.anchor_idx,
        "anchor_name": cs.ents[cs.anchor_idx] if cs.anchor_idx is not None and 0 <= cs.anchor_idx < len(cs.ents) else cs.anchor_name,
        "breakpoints": {k: cs.ents[v] for k, v in cs.breakpoints.items() if v is not None and 0 <= v < len(cs.ents)},
        "step_relations": [list(r) for r in cs.step_relations],
        "entity_retrieval_details": cs.entity_retrieval_details,
        "relation_retrieval_details": cs.relation_retrieval_details,
        "prune_debug": cs.prune_debug,
        "layer_diagnostics": cs.layer_diagnostics,
        "planning_attempts": cs.planning_attempts,
        "max_depth": cs.max_depth,
        "num_paths": len(cs.paths),
        "answer_candidates": cs.answer_candidates,
        "gt_hit": cs.gt_hit,
        "gt_hit_strict": cs.gt_hit_strict,
        "gt_f1": cs.gt_f1,
        "num_patterns": len(logical_paths),
        "logical_paths": [lp["readable"] for lp in logical_paths[:10]],
        "pattern_details": [{
            "candidates": lp.get("candidates", []),
            "best_tier": lp.get("best_tier", (0, -1, 0)),
            "endpoint": lp.get("endpoint"),
            "witness_nodes": lp.get("best_raw_path", {}).get("nodes", []),
            "witness_relations": lp.get("best_raw_path", {}).get("relations", []),
        } for lp in logical_paths],
        "llm_answer": cs.llm_answer,
        "llm_hit": cs.llm_hit,
        "llm_f1": cs.llm_f1,
        "llm_precision": cs.llm_precision,
        "llm_recall": cs.llm_recall,
        "selected_paths": cs.selected_paths,
        "num_triples": cs.num_triples,
        "attempt_log": cs.attempt_log,
        "llm_reasoning_prompt": cs.llm_reasoning_prompt,
        "llm_reasoning_full": cs.llm_reasoning_full,
        "stage_times": cs.stage_times,
        "error": cs.error,
    }


async def _run_stage_mode(cases_to_run, args):
    """Process all cases stage-by-stage with batch LLM calls."""
    import time as _time

    # Initialize CaseStates
    case_states = []
    for sample, pilot_row, idx in cases_to_run:
        gt = pilot_row.get("gt", pilot_row.get("ground_truth", pilot_row.get("gt_answers", [])))
        cs = CaseState(
            case_id=pilot_row["case_id"],
            case_num=idx,
            sample=sample,
            pilot_row=pilot_row,
            question=pilot_row["question"],
            gt_answers=gt if isinstance(gt, list) else [gt],
        )
        case_states.append(cs)

    total = len(case_states)
    print(f"\n=== Stage-batch mode: {total} cases ===")

    wall_start = _time.perf_counter()

    # GT anchor override: use q_entity from ROG-CWQ pkl as anchor
    gt_anchor_map = {}  # case_id -> q_entity_id_list[0]
    if getattr(args, 'gt_anchor', False):
        pkl_dir = ROOT / "data/cwq_processed"
        for split in ['test', 'train', 'val']:
            pkl_path = pkl_dir / f"{split}.pkl"
            if pkl_path.exists():
                import pickle
                with open(pkl_path, 'rb') as f:
                    pkl_data = pickle.load(f)
                for d in pkl_data:
                    qe_ids = d.get("q_entity_id_list", [])
                    if qe_ids:
                        gt_anchor_map[d["id"]] = {
                            "idx": qe_ids[0],
                            "name": d.get("q_entity", [""])[0],
                        }
        print(f"  GT anchor: loaded {len(gt_anchor_map)} cases with q_entity")

    async with aiohttp.ClientSession() as session:
        # Always run Stage 0 to load KG data (ents, rels, h_ids, etc.)
        await stage_0_ner_resolve(session, case_states)

        # ── Inject golden Stage 1a/1/1.5 outputs if requested ──
        inject_map = {}
        if getattr(args, 'inject_decomp', None):
            inject_map = {r["case_id"]: r for r in json.loads(Path(args.inject_decomp).read_text())}
            injected = 0
            for cs in case_states:
                g = inject_map.get(cs.case_id)
                if not g:
                    continue
                # Stage 1a (entity analysis)
                cs._1a_raw = g.get("stage_1a_raw")
                cs._1a_anchor = g.get("stage_1a_anchor")
                cs._1a_endpoints = g.get("stage_1a_endpoints")
                cs._1a_answer_type = g.get("stage_1a_answer_type")
                cs._1a_rewritten = g.get("stage_1a_rewritten")
                cs._1a_interpretation = g.get("stage_1a_interpretation")
                cs.answer_type = g.get("stage_1a_answer_type") or g.get("answer_type")
                if cs._1a_rewritten:
                    cs.rewritten_question = cs._1a_rewritten
                # Stage 1 (decomposition)
                cs.decomp_raw = g.get("decomposition")
                cs.decomp_question = g.get("decomposition_question", "")
                cs.steps = g.get("steps_parsed", [])
                cs.anchor_name = g.get("anchor_name")
                # Resolve anchor_idx from ents by name
                cs.anchor_idx = None
                if cs.anchor_name and cs.ents:
                    for i, e in enumerate(cs.ents):
                        if e == cs.anchor_name:
                            cs.anchor_idx = i
                            break
                    if cs.anchor_idx is None:
                        # Fuzzy: find entity containing anchor name
                        an_lower = cs.anchor_name.lower()
                        for i, e in enumerate(cs.ents):
                            if an_lower in e.lower() or e.lower() in an_lower:
                                cs.anchor_idx = i
                                break
                # Breakpoints: convert {str_step: str_entity} → {int_step: int_idx}
                bp_raw = g.get("breakpoints", {})
                cs.breakpoints = {}
                if bp_raw:
                    ent_to_idx = {e: i for i, e in enumerate(cs.ents)}
                    for k, v in bp_raw.items():
                        idx = ent_to_idx.get(v)
                        if idx is not None:
                            cs.breakpoints[int(k)] = idx
                # Stage 1.5 (reflection)
                cs.decomp_retry = g.get("decomp_retry", False)
                cs.decomp_reflect_raw = g.get("decomp_reflect_raw")
                cs.decomp_retry_reason = g.get("decomp_retry_reason")
                injected += 1
            print(f"  Injected golden decomp for {injected}/{total} cases (skipping Stage 1/1.5)")

        # Override anchor with GT q_entity after stage 0 (ents loaded)
        if gt_anchor_map:
            overridden = 0
            for cs in case_states:
                gt_info = gt_anchor_map.get(cs.case_id)
                if not gt_info:
                    continue
                idx = gt_info["idx"]
                if 0 <= idx < len(cs.ents):
                    cs.anchor_idx = idx
                    cs.anchor_name = cs.ents[idx]
                    overridden += 1
            print(f"  GT anchor: overridden {overridden}/{total} anchors")

        if not inject_map:
            await stage_1_decomposition(session, case_states)
            await stage_1_5_decomposition_reflect(session, case_states)
        await stage_2_entity_resolution(session, case_states)

        # Re-apply GT anchor after stage 2 (stage 2 may override it)
        if gt_anchor_map:
            for cs in case_states:
                gt_info = gt_anchor_map.get(cs.case_id)
                if gt_info:
                    idx = gt_info["idx"]
                    if 0 <= idx < len(cs.ents):
                        cs.anchor_idx = idx
                        cs.anchor_name = cs.ents[idx]
        await stage_3_gte_relation_retrieval(session, case_states)
        if args.prune == "rerank":
            await stage_4_rerank_pruning(case_states, args.rerank_model, args.prune_top_k)
        else:
            await stage_4_relation_pruning(session, case_states)
        await stage_5_graph_traversal(case_states)
        await stage_6_diagnosis_retry(session, case_states)
        await stage_7_path_selection(session, case_states)
        await stage_8_answer_reasoning(session, case_states)

    wall_time = _time.perf_counter() - wall_start

    # Collect results
    rows = [_case_state_to_result_dict(cs) for cs in case_states]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    cases = len(rows)
    if cases == 0:
        print("\n=== No valid cases processed ===")
        return

    hits = sum(1 for r in rows if r["gt_hit"])
    llm_hits = sum(1 for r in rows if r.get("llm_hit"))
    errors = sum(1 for r in rows if r.get("error"))

    # Timing
    all_times = []
    for cs in case_states:
        total_stage = sum(cs.stage_times.values())
        all_times.append(total_stage)

    avg_case = sum(all_times) / len(all_times) if all_times else 0
    total_case = sum(all_times)

    # Per-case detail
    for i, (r, cs) in enumerate(zip(rows, case_states)):
        gt_mark = "✓" if r["gt_hit"] else "✗"
        gt_strict_mark = "S" if r.get("gt_hit_strict") else " "
        llm_mark = "✓" if r.get("llm_hit") else "✗"
        gt_f1_val = r.get('gt_f1', 0)
        llm_f1_val = r.get('llm_f1', 0)
        cov = getattr(cs, 'max_cov', 0)
        nsteps = len(getattr(cs, 'steps', []))
        print(f"  [{i}] GT{gt_mark}{gt_strict_mark}(F1={gt_f1_val:.2f}) LLM{llm_mark}(F1={llm_f1_val:.2f}) cov={cov}/{nsteps} paths={len(getattr(cs,'paths',[]))} | {r['question'][:60]}")

    strict_hits = sum(1 for r in rows if r.get("gt_hit_strict"))
    # Macro-averaged F1
    gt_f1_avg = sum(r.get('gt_f1', 0) for r in rows) / cases if cases else 0
    llm_f1_avg = sum(r.get('llm_f1', 0) for r in rows) / cases if cases else 0
    llm_p_avg = sum(r.get('llm_precision', 0) for r in rows) / cases if cases else 0
    llm_r_avg = sum(r.get('llm_recall', 0) for r in rows) / cases if cases else 0
    print(f"\n=== Summary: GT_recall={hits}/{cases} ({100*hits/cases:.1f}%) | strict={strict_hits}/{cases} ({100*strict_hits/cases:.1f}%) | GT_F1={gt_f1_avg:.3f} | LLM_reason={llm_hits}/{cases} ({100*llm_hits/cases:.1f}%) | LLM_P={llm_p_avg:.3f} R={llm_r_avg:.3f} F1={llm_f1_avg:.3f} | Errors={errors} ===")
    print(f"    Timing: Wall={wall_time:.2f}s | Sum stages={total_case:.2f}s | Avg case={avg_case:.2f}s")
    speedup = total_case / wall_time if wall_time > 0 else 1.0
    print(f"    Batch speedup: {speedup:.2f}x")

    # Dump failed case traces to file (full pipeline, no truncation)
    fail_traces = []
    for i, (r, cs) in enumerate(zip(rows, case_states)):
        if r.get("gt_hit") and r.get("llm_hit"):
            continue
        lines = [f"{'='*70}"]
        lines.append(f"Case [{i}]: {r.get('case_id','')}")
        lines.append(f"Q: {r.get('question','')}")
        lines.append(f"GT: {r.get('gt_answers',[])}")
        lines.append(f"LLM: {r.get('llm_answer','')}")
        lines.append(f"GT_hit={r.get('gt_hit',False)} GT_F1={r.get('gt_f1',0):.2f} LLM_hit={r.get('llm_hit',False)} LLM_P={r.get('llm_precision',0):.2f} R={r.get('llm_recall',0):.2f} F1={r.get('llm_f1',0):.2f}")
        if r.get("error"):
            lines.append(f"Error: {r['error']}")

        # === Stage 1a: Entity Analysis ===
        stage_1a_prompt = r.get("stage_1a_prompt", "")
        stage_1a_raw = r.get("stage_1a_raw", "")
        if stage_1a_raw or stage_1a_prompt:
            lines.append(f"\n{'='*40} STAGE 1a: Entity Analysis {'='*40}")
            if stage_1a_prompt:
                lines.append(f"--- 1a Input Prompt ---")
                lines.append(stage_1a_prompt)
            if stage_1a_raw:
                lines.append(f"--- 1a Output ---")
                lines.append(stage_1a_raw[:2000])
            lines.append(f"--- 1a Parsed ---")
            lines.append(f"  Anchor: {r.get('stage_1a_anchor','')}")
            lines.append(f"  Endpoints: {r.get('stage_1a_endpoints','')}")
            lines.append(f"  Interpretation: {r.get('stage_1a_interpretation','')}")
            lines.append(f"  Answer_type: {r.get('stage_1a_answer_type','')}")
            lines.append(f"  Rewritten: {r.get('stage_1a_rewritten','')}")

        # === Stage 1b: Chain Decomposition ===
        lines.append(f"\n{'='*40} STAGE 1b: Chain Decomposition {'='*40}")
        decomp_prompt = r.get("decomposition_prompt", "")
        if decomp_prompt:
            lines.append(f"--- 1b Prompt (formatted) ---")
            lines.append(decomp_prompt[:2000])
        decomp_raw = r.get("decomposition", "")
        if decomp_raw:
            lines.append(f"--- 1b Result ---")
            lines.append(decomp_raw)
        lines.append(f"Anchor: {r.get('anchor_name','')} (idx={r.get('anchor_idx','')})")
        bp = r.get("breakpoints", {})
        if bp:
            lines.append(f"Breakpoints: {bp}")
        steps = r.get("steps_parsed", [])
        for si, s in enumerate(steps):
            lines.append(f"  Step {si}: {s}")

        # === Stage 3-4: GTE + Pruning ===
        lines.append(f"\n{'='*40} STAGE 3-4: GTE + Pruning {'='*40}")
        step_rels = getattr(cs, 'step_relations', [])
        rels_list = getattr(cs, 'rels', [])
        if step_rels:
            for si, rels in enumerate(step_rels):
                named = [f"{ri}:{rels_list[ri]}" if ri < len(rels_list) else str(ri) for ri in (rels if isinstance(rels, list) else list(rels))]
                lines.append(f"  Step {si} relations: {named}")
        gte_details = r.get("relation_retrieval_details", [])
        if gte_details:
            lines.append(f"  GTE details: {json.dumps(gte_details, ensure_ascii=False)[:1000]}")

        # === Stage 5: Traversal ===
        lines.append(f"\n{'='*40} STAGE 5: Graph Traversal {'='*40}")
        paths = getattr(cs, 'paths', [])
        ents = getattr(cs, 'ents', [])
        if paths:
            lines.append(f"--- Raw Paths ({len(paths)}) ---")
            for pi, p in enumerate(paths):
                nodes = p.get("nodes", [])
                rels_p = p.get("relations", [])
                chain_parts = []
                for ni in range(len(nodes)):
                    chain_parts.append(ents[nodes[ni]] if 0 <= nodes[ni] < len(ents) else f"#{nodes[ni]}")
                    if ni < len(rels_p):
                        chain_parts.append(f"--[{rels_list[rels_p[ni]]}]-->" if rels_p[ni] < len(rels_list) else f"--[{rels_p[ni]}]-->")
                lines.append(f"  Path {pi}: {' '.join(chain_parts)}")

        # === Stage 7: Path Selection ===
        lines.append(f"\n{'='*40} STAGE 7: Path Selection {'='*40}")
        logical_paths = getattr(cs, 'logical_paths', [])
        if logical_paths:
            lines.append(f"--- Logical Paths ({len(logical_paths)}) ---")
            for li, lp in enumerate(logical_paths):
                readable = lp.get("readable", "")
                cands = lp.get("candidates", [])
                endpoint = lp.get("endpoint")
                lines.append(f"  LP {li}: {readable}")
                lines.append(f"         candidates({len(cands)}): {cands[:10]}{'...' if len(cands)>10 else ''}")
                if endpoint is not None:
                    ep_name = ents[endpoint] if isinstance(endpoint, int) and 0 <= endpoint < len(ents) else str(endpoint)
                    lines.append(f"         endpoint: {ep_name}")
        sel_prompt = getattr(cs, 'path_select_prompt', None)
        if sel_prompt:
            lines.append(f"--- Stage 7 LLM Prompt ---")
            lines.append(sel_prompt)
        sel_response = getattr(cs, 'path_select_response', None)
        if sel_response:
            lines.append(f"--- Stage 7 LLM Response ---")
            lines.append(sel_response)
        sel_paths = getattr(cs, 'selected_paths', [])
        if sel_paths is not None:
            lines.append(f"--- Selected path indices: {sel_paths} ---")

        # === Stage 8: Answer Reasoning ===
        lines.append(f"\n{'='*40} STAGE 8: Answer Reasoning {'='*40}")
        reasoning_prompt = getattr(cs, 'llm_reasoning_prompt', None)
        if reasoning_prompt:
            lines.append(f"--- Stage 8 LLM Prompt ---")
            lines.append(reasoning_prompt)
        reasoning = r.get("llm_reasoning_full", "")
        if reasoning:
            lines.append(f"--- Stage 8 LLM Response ---")
            lines.append(reasoning)

        # === Candidates ===
        ans_cands = r.get("answer_candidates", [])
        if ans_cands:
            lines.append(f"\n--- Answer Candidates ({len(ans_cands)}) ---")
            for ci in range(0, len(ans_cands), 10):
                lines.append(f"  {ans_cands[ci:ci+10]}")

        fail_traces.append('\n'.join(lines))
    if fail_traces:
        trace_path = out / "fail_traces.txt"
        trace_path.write_text('\n\n'.join(fail_traces), encoding='utf-8')
        print(f"  Fail traces written to: {trace_path} ({len(fail_traces)} cases)")


if __name__ == "__main__":
    asyncio.run(amain())
