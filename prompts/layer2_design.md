# Layer 2: GTE Relation Resolution

## Purpose
Resolve natural language relation candidate phrases from Layer 1 to actual KG relation names using semantic similarity.

## Input
- For each intent step in the intent_chain: 3 relation_candidate phrases
- The sample's relation_list (all relations in the subgraph)

## Process
For each relation_candidate phrase:
1. Encode the phrase using GTE-large embedding model
2. Encode all relations in the sample's relation_list
3. Compute cosine similarity between phrase and each relation
4. Return top-3 relations per phrase (ranked by similarity score)

## Output Format
```python
{
  "intent_step_1": {
    "phrase_1": [
      {"relation": "location.administrative_division.country", "score": 0.85},
      {"relation": "location.country.administrative_divisions", "score": 0.72},
      {"relation": "location.location.contains", "score": 0.68}
    ],
    "phrase_2": [...],
    "phrase_3": [...]
  },
  "intent_step_2": {...}
}
```

## Backend Implementation Notes
- Use the existing GTE-large service at port 8003
- Encode relations in "type property" format (last 2 segments of full relation)
- Example: "location.administrative_division.country" → "administrative division country"
- Threshold: cosine_sim >= 0.3 (below this, still return top 3 but flag as low confidence)
- Deduplicate relations across phrases within the same intent step

## Integration Point
This is NOT a separate LLM call. It is a deterministic backend operation that happens between Layer 1 and Layer 3.

## Edge Cases
| Case | Handling |
|------|----------|
| No matches above 0.3 | Return top 3 anyway; let Layer 3 filter |
| Empty relation_list | Fail fast; return empty dict |
| All relations identical | Deduplicate to single entry |
| GTE service unavailable | Fall back to lexical keyword matching |

## Relation Resolution Pool
The relation_list comes from the sample's subgraph:
- File: typically `m.relation2id.keys()` from the sample data
- Scope: ALL relations present in this specific question's subgraph
- Size: varies by sample, typically 50-200 relations

## Deterministic Nature
This layer is deterministic:
- Same input → same output (modulo GTE floating point variance)
- No LLM inference required
- Can be cached for performance
