# 3-Layer KGQA Architecture: Relation-Prior-Constrained Subgraph Construction

## Overview

This document describes the 3-layer prompt architecture for a KGQA system based on "relation-prior-constrained subgraph construction". The system decomposes questions, resolves relations via semantic retrieval, builds constrained subgraphs, and reasons over them.

## Architecture Diagram

```
Question
    |
    v
+-------------------------+
|  Layer 1: Decomposition  |  LLM
|  - Extract anchor         |
|  - Build intent chain     |
|  - Relation candidates    |
+-------------------------+
    | (anchor, intent_chain, target_entity, relation_candidates)
    v
+-------------------------+
|  Layer 2: GTE Resolution |  Backend (deterministic)
|  - Semantic retrieval     |
|  - Phrase → KG relation   |
+-------------------------+
    | (resolved_relations per intent step)
    v
+-------------------------+
|  Backend: Subgraph Build |  Backend
|  - Depth-based expansion  |
|  - Path retention        |
|  - Target filtering      |
+-------------------------+
    | (constrained subgraph triples)
    v
+-------------------------+
|  Layer 3: Reasoning      |  LLM
|  - Path evaluation        |
|  - Answer extraction      |
+-------------------------+
    |
    v
Answer
```

## Key Principle: Depth-Based Path Retention

The backend subgraph construction uses **depth-based path retention**:
- Expand from anchor using resolved relations per intent step
- Keep only the DEEPEST paths (most relation layers matched)
- If target_entity is present, filter further to paths reaching the target
- Shallow paths are pruned unless no deep paths exist

This ensures the LLM in Layer 3 receives the most complete reasoning chains.

## File Structure

```
prompts/
├── layer1_system_v2.txt      # Layer 1 system prompt
├── layer1_user_v2.txt        # Layer 1 user template with few-shot
├── layer2_design.md          # Layer 2 specification (not a prompt)
├── layer3_system_v2.txt      # Layer 3 system prompt
├── layer3_user_v2.txt        # Layer 3 user template with few-shot
├── edge_cases_v2.md          # Edge cases and failure modes
└── ARCHITECTURE.md           # This file
```

## Layer 1: Question Decomposition

**Purpose**: Decompose CWQ question into structured intent with relation candidate phrases

**Input**: Raw question text

**Output**: JSON with anchor, intent_chain, target_entity

**System Prompt**: `layer1_system_v2.txt`
- Defines JSON schema
- Establishes anchor selection priority (named entities > answer-types)
- Specifies intent chain ordering (execution order)
- Describes relation candidate format (3 phrases, 2-4 words each)
- Handles target_entity constraints

**User Template**: `layer1_user_v2.txt`
- 5 few-shot examples covering:
  - Multi-hop with target entity (Alta Verapaz)
  - Multi-hop without target (Michelle Bachelet)
  - Temporal question (Lou Seal)
  - Inverse reasoning (Libya anthem)
  - Value constraint filtering (Portuguese + child labor)

**Key Design Decisions**:
- 3 relation candidates per intent (balance between recall and noise)
- Intent chain MUST be in execution order
- Target entity only for explicit named constraints
- Temporal/value modifiers encoded in relation phrases

## Layer 2: GTE Relation Resolution

**Purpose**: Resolve natural language phrases to KG relation names

**Input**: Relation candidate phrases from Layer 1, sample's relation_list

**Output**: Top-3 KG relations per phrase, ranked by cosine similarity

**Specification**: `layer2_design.md`
- Uses GTE-large for semantic encoding
- Encodes relations as "type property" format (last 2 segments)
- Threshold: 0.3 cosine similarity (below threshold still returns results)
- Deduplicates across phrases within same intent step

**Key Design Decisions**:
- Deterministic (no LLM call)
- Can be cached for performance
- Graceful degradation on low similarity
- Falls back to lexical matching if GTE unavailable

## Layer 3: Subgraph Reasoning

**Purpose**: Extract answer from constrained subgraph

**Input**: Question, anchor, target_entity, subgraph triples

**Output**: Final answer string

**System Prompt**: `layer3_system_v2.txt`
- Emphasizes exclusive subgraph usage
- Defines path following strategy
- Handles multiple paths (prioritize deeper)
- Specifies output format (concise answer only)

**User Template**: `layer3_user_v2.txt`
- 4 few-shot examples showing:
  - Target entity filtering
  - Multi-hop reasoning
  - Temporal value extraction
  - Inverse relation traversal

**Key Design Decisions**:
- No reasoning trace in output
- Temporal values extracted from terminal entities
- Explicit failure messages for unanswerable cases
- Depth-based path prioritization

## Integration Flow

```python
# Pseudocode
def answer_question(question: str, sample: SubgraphSample) -> str:
    # Layer 1: Decomposition
    decomposition = llm_call(
        system=layer1_system_v2,
        user=layer1_user_v2.format(question=question)
    )
    # Returns: {"anchor": "...", "intent_chain": [...], "target_entity": "..."}

    # Layer 2: GTE Resolution (backend)
    resolved_relations = {}
    for step in decomposition["intent_chain"]:
        for phrase in step["relation_candidates"]:
            resolved_relations[phrase] = gte_retrieve(
                phrase,
                sample.relation_list,
                top_k=3
            )

    # Backend: Subgraph Construction
    subgraph = build_constrained_subgraph(
        anchor=decomposition["anchor"],
        resolved_relations=resolved_relations,
        target_entity=decomposition["target_entity"],
        sample=sample
    )
    # Uses depth-based path retention

    # Layer 3: Reasoning
    answer = llm_call(
        system=layer3_system_v2,
        user=layer3_user_v2.format(
            question=question,
            anchor=decomposition["anchor"],
            target_entity=decomposition["target_entity"],
            subgraph=format_triples(subgraph.triples)
        )
    )

    return answer
```

## Backend Subgraph Construction Algorithm

```
function build_constrained_subgraph(anchor, relations, target, sample):
    paths = []
    max_depth = len(relations)  # One step per intent

    for intent_step in relations:
        # Expand using resolved relations for this step
        for relation in intent_step:
            new_paths = expand_along_relation(current_paths, relation)
            paths.extend(new_paths)

    # Depth-based retention
    max_depth_found = max(path.depth for path in paths)
    deep_paths = [p for p in paths if p.depth == max_depth_found]

    # Target filtering if specified
    if target:
        filtered = [p for p in deep_paths if p.terminus == target]
        if filtered:
            deep_paths = filtered
        # Else: return deep paths anyway; Layer 3 will report mismatch

    return merge_paths_to_subgraph(deep_paths)
```

## Testing Strategy

1. **Unit Testing**:
   - Layer 1: Verify JSON parsing, intent ordering, anchor selection
   - Layer 2: Verify GTE retrieval, deduplication, fallback
   - Layer 3: Verify answer extraction from synthetic subgraphs

2. **Integration Testing**:
   - End-to-end on the 5 few-shot examples
   - Compare against known CWQ ground truth

3. **Evaluation**:
   - Full CWQ test set
   - Metrics: Exact Match accuracy, F1 for multi-answer cases

## Migration Notes

This architecture replaces:
- Old: Agent 0 Decomposition → Discovery (LLM tools) → Agent 2 Planning → BFS
- New: Layer 1 Decomposition → Layer 2 GTE → Backend Constrained Expansion → Layer 3 Reasoning

Key differences:
- Discovery phase eliminated (replaced by deterministic GTE)
- Planning phase eliminated (intent chain from Layer 1 guides expansion)
- BFS replaced by depth-based path retention
- Reasoning separated into dedicated Layer 3

## Future Enhancements

1. **Adaptive top-k**: Adjust GTE retrieval count based on confidence
2. **Relation composition**: Allow multi-relation patterns within a single intent step
3. **Confidence calibration**: Expose confidence scores from all layers
4. **Caching**: Cache Layer 1 decompositions and Layer 2 retrievals
5. **Ensemble**: Run multiple Layer 3 candidates and vote on answer
