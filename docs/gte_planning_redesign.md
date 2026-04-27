# GTE-Based Planning Redesign — Change Report

> Date: 2026-04-11/12
> Status: In Progress (pending prompt review and testing)

## 0. 2026-04-12 Clarified Contract

This redesign was further clarified after implementation review:

- GTE retrieval is **not** a separate agent stage.
- The system first decomposes the question into:
  - one `[MAIN]` question that directly asks for the final answer slot
  - zero or more `[ATTR]` questions that describe properties used to judge candidates
- After decomposition, the system automatically runs GTE retrieval for both `[MAIN]` and `[ATTR]` questions.
- The planning model then produces exactly one `plan()`:
  - `[MAIN]` determines `anchor`, `related`, `maybe_related`
  - `[ATTR]` determines `constraint_entities`, `constraint_relations`
- Constraint handling is **entity-first**:
  - if an attribute question yields a concrete entity, prefer `constraint_entities`
  - use `constraint_relations` when no stable entity is available
- After action selection, the backend performs batch constraint checking over the selected action space candidates:
  - connectivity to `constraint_entities`
  - attribute matches for `constraint_relations`
  - CVT hits are shown with restricted local-neighbor summaries instead of unbounded expansion

---

## 1. Background

The original KGQA pipeline had a "Discovery" phase that used LLM tool calls (`check_entities`, `explore_schema`) to find relevant entities and relations in a knowledge graph subgraph. This was slow (multiple LLM rounds) and unreliable (model often failed to find the right entities/relations).

**Redesign goal**: Replace Discovery with GTE-large semantic retrieval, which directly matches the question against ALL entities and ALL relations in the subgraph, achieving ~100% quality in top-10 recall.

---

## 2. Architecture Overview

### Old Flow
```
Question → Agent 0 Decomposition → check_entities (LLM tool) → explore_schema (LLM tool) → Agent 2 Planning → BFS → Filter → Answer
```

### New Flow
```
Question → Agent 0 Decomposition → GTE Semantic Retrieve (backend) → LLM Planning (3 judgments) → BFS → Constraint Filter → Answer
```

**Key change**: Discovery phase is eliminated. The backend `/v2/semantic_retrieve` endpoint uses GTE embeddings to rank all entities and relations against the question/sub-questions. The LLM only needs to make classification judgments from GTE-recalled candidates.

---

## 3. Detailed Changes

### 3.1 Backend: `/v2/semantic_retrieve` Endpoint

**File**: `/zhaoshu/SubgraphRAG-main/retrieve/graph_server.py`

**New models added** (after line 168):
```python
class SemanticRetrieveRequest(BaseModel):
    sample_id: str
    queries: List[str] = Field(..., description="List of text queries (question + sub-questions)")
    top_k: int = 10
    gte_url: str = "http://localhost:8003"

class SemanticRetrieveResponse(BaseModel):
    success: bool
    entities_per_query: List[List[Dict]]  # [{candidate, score}, ...]
    relations_per_query: List[List[Dict]]
    total_entities: int = 0
    total_relations: int = 0
```

**New endpoint** `/v2/semantic_retrieve`:
- Gets all entities from `m.id2entity.values()`, filters CVT nodes
- Gets all relations from `m.relation2id.keys()`
- Encodes relations using "type property" format (last 2 segments, e.g., "sports team championships")
- Calls GTE-large (port 8003) to encode question + entities + relations
- Returns ranked results per query (cosine similarity)

**Verified performance** (12 ground truth cases):
- Entity Top-1: 58% (for anchor), ~100% quality in top-10
- Relation Top-10: ~100% quality (manually verified)
- Sub-questions give better relation recall than original question for multi-hop cases

### 3.2 Backend: Constraint Fallback Fix

**File**: `/zhaoshu/SubgraphRAG-main/retrieve/graph_server.py`

**Change** (line ~2472): When constraints are provided but NO candidates match, return ALL candidates instead of empty set.

```python
# Before:
final_filter_set = set()  # empty — LLM gets nothing

# After:
final_filter_set = None  # return full set — LLM can still review
```

**Rationale**: LLM can extract useful information from the full candidate set, but cannot do anything with an empty set.

### 3.3 Frontend: New Planning Prompt

**File**: `/zhaoshu/subgraph/scripts/run_stage_pipeline_test.py`

**`_GTE_PLAN_SYSTEM`** (line ~1241): Rewritten system prompt.

**Design**: The LLM makes 3 decisions for each GTE-recalled candidate:

| Decision | Options | Purpose |
|----------|---------|---------|
| 1. Relevance | relevant / ignore | Filter noise from GTE results |
| 2. Entity Role | anchor / constraint_entities | anchor = traversal start; constraint = filter candidates |
| 3. Relation Role | related / constraint_relations | related = BFS exploration; constraint = filter candidates |

**Key rules**:
- Model can select ALL relevant relations (no limit). Backend executes top-5.
- `[MAIN]` relations → prefer `related`
- `[HOP]` relations → `related`
- `[FILTER]` relations → prefer `constraint_relations`
- Non-anchor relevant entities → `constraint_entities`
- Irrelevant items → simply ignore

**Includes 1 concrete example** showing the full input → reasoning → output flow.

**User content template** (line ~1320): Updated to prompt the 3-decision framework.

### 3.4 Frontend: Planning Function `_run_planning_gte`

**File**: `/zhaoshu/subgraph/scripts/run_stage_pipeline_test.py` (line ~1298)

This function was already implemented. Key features:
- Formats GTE relations grouped by sub-question
- Formats GTE entities as numbered list
- Calls LLM with new prompt
- Parses plan() call from model output
- Anchor correction via fuzzy matching + GTE top-1 fallback
- Auto-adds constraint entities if model didn't set any
- Retries on parse failure (max 2 retries)

### 3.5 Diagnostic Scripts (Reference)

**File**: `/zhaoshu/subgraph/scripts/diag_gte_full_retrieve.py`
- Tests GTE retrieval against full subgraph for 12 GT cases
- Results: Anchor Top-1 58%, Relation Top-5 38% (oracle), ~100% actual quality

**File**: `/zhaoshu/subgraph/scripts/diag_gte_subq_retrieve.py`
- Tests sub-question vs original question retrieval
- Key finding: HOP sub-questions give 62% relation Top-3 (vs 33% for original)

**File**: `/zhaoshu/subgraph/scripts/diag_gte_topk_detail.py`
- Prints top-10 entities and relations per case for manual inspection

---

## 4. Backend Constraint Mechanism (Existing, Unchanged)

For reference, the existing constraint filtering in the backend:

**Plan endpoint** (`/v2/plan`, line 1883-1897):
- `constraint_relations` and `constraint_entities` from plan are injected into every action hint

**Action execution** (`_execute_select`, line 2039-2147):
- For each candidate entity:
  - Checks if it has values for `constraint_relations` (attribute check)
  - Checks if it has a path to `constraint_entities` (connectivity check)
- Classifies constraints: GLOBAL (matched all), PARTIAL (matched some), NONE
- Uses PARTIAL constraints preferentially (discriminative power)
- **Hard filtering**: Only returns candidates that match at least one active constraint

**Flow**:
```
anchor + related relations → BFS → N candidates
constraint_entities + constraint_relations → filter → only matching candidates shown
```

---

## 5. Pending Review Issues

An independent architect review identified the following issues with the new prompt:

### P0 (Must Fix)
1. **Anchor selection principle missing**: 9B model may select answer-type entities (e.g., year, country) as anchor. Need explicit priority rules.
2. **Example "skip" format inconsistency**: Example uses "→ skip" but output format doesn't define this. Model may produce unparseable output.
3. **[HOP] relation explanation insufficient**: Multi-hop relations may be misclassified as constraint_relations.
4. **Constraint timing unclear**: Model may confuse traversal vs filtering roles.

### P1 (Recommended)
5. Need 2-3 additional examples (negative cases, complex multi-hop)
6. Low similarity threshold handling rules
7. Behavior when no [FILTER] sub-questions exist

---

## 6. Files Modified

| File | Change Type | Description |
|------|------------|-------------|
| `/zhaoshu/SubgraphRAG-main/retrieve/graph_server.py` | Added | `/v2/semantic_retrieve` endpoint + models |
| `/zhaoshu/SubgraphRAG-main/retrieve/graph_server.py` | Modified | Constraint fallback: empty set → full set |
| `/zhaoshu/subgraph/scripts/run_stage_pipeline_test.py` | Modified | `_GTE_PLAN_SYSTEM` prompt rewritten |
| `/zhaoshu/subgraph/scripts/run_stage_pipeline_test.py` | Modified | User content template updated |

## 7. Files Created (Diagnostic, Reference Only)

| File | Purpose |
|------|---------|
| `/zhaoshu/subgraph/scripts/diag_gte_full_retrieve.py` | Full subgraph GTE retrieval diagnostic |
| `/zhaoshu/subgraph/scripts/diag_gte_subq_retrieve.py` | Sub-question vs original question retrieval test |
| `/zhaoshu/subgraph/scripts/diag_gte_topk_detail.py` | Top-10 entities/relations detail printer |

---

## 8. Testing Plan (Pending)

1. Verify `/v2/semantic_retrieve` endpoint works ✓ (tested via curl)
2. Run pipeline with new prompt on 12 GT cases
3. Compare planning quality vs old discovery-based approach
4. Fix P0 prompt issues based on review feedback
5. Full CWQ test set evaluation
