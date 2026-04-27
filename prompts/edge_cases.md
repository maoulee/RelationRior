# Edge Cases and Failure Modes

## Layer 1: Question Decomposition

| Case | Handling |
|------|----------|
| Multiple anchors | Choose the one that starts the reasoning chain |
| Ambiguous intent order | Use the chain that requires fewer hops |
| No clear target | Set `target_entity: null` |
| Value constraints (1.8, etc.) | Include in intent, NOT as target_entity |
| Temporal questions ("last won", "once") | Include temporal aspect in relation_candidates |
| Invalid JSON output | Retry with stronger format instruction |

## Layer 2: GTE Relation Resolution

| Case | Handling |
|------|----------|
| No semantic match (cosine sim < 0.3) | Expand top_k to 10 or fall back to lexical matching |
| Duplicate relations across candidates | Deduplicate while preserving order |
| Relation candidates too generic | Increase top_k to 10 for recall |
| Empty relation_list | Return empty; fail fast to Layer 3 |
| All similarities below threshold | Return top 5 anyway; let Layer 3 filter |

## Layer 3: Subgraph Reasoning

| Case | Handling |
|------|----------|
| No path from anchor to answer | Return "Cannot be determined from provided subgraph" |
| Multiple valid answers | Return all, comma-separated |
| Target entity constraint not satisfied | Return "No valid path to target entity" |
| Cyclic paths | Follow the path matching most intent steps |
| Ambiguous entity mentions | Use disambiguation from subgraph context |
| Empty subgraph | Return "Insufficient information to answer" |
| Disconnected subgraph components | Follow the component containing the anchor |

## Backend Subgraph Construction

| Case | Handling |
|------|----------|
| No neighbors found at step | Prune that branch; continue with others |
| Max depth exceeded | Return deepest partial path found |
| Target entity not in graph | Return paths without target; LLM will report mismatch |
| Memory limit on subgraph | Prioritize by path depth, then by confidence score |
| Multiple anchor matches | Use entity with most outgoing relations |

## System-Level Failures

| Case | Handling |
|------|----------|
| Layer 1 produces invalid JSON | Catch exception; return structured error |
| Layer 2 GTE model unavailable | Fall back to keyword/lexical matching |
| Layer 3 LLM timeout | Return subgraph as fallback answer |
| KG query failure | Return "Knowledge base unavailable" |
| End-to-end latency > 10s | Implement caching for frequent questions |
