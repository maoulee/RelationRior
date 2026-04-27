# Edge Cases and Failure Modes

## Layer 1: Question Decomposition

| Case | Handling |
|------|----------|
| Multiple entity mentions | Choose the one that STARTS the reasoning chain (given not asked) |
| Answer-type entities as anchor | Avoid: prefer named entities (Lou Seal not 2014, Alta Verapaz not Guatemala) |
| Ambiguous intent order | Use the chain requiring fewer hops; multi-hop questions have clear progression |
| No clear target entity | Set `target_entity: null` |
| Value constraints (1.8, "once") | Include in relation_candidates, NOT as target_entity |
| Temporal modifiers ("last", "most recent") | Include temporal aspect in relation_candidates phrases |
| Invalid JSON output | Retry with stronger format instruction; use JSON schema validation |
| Single-hop questions | intent_chain with 1 step |
| Complex multi-hop (4+ steps) | Create sequential steps; break complex intents into atomic hops |
| Implicit entities (team with mascot) | Include as relation_candidate phrase, not as explicit entity |

## Layer 2: GTE Relation Resolution

| Case | Handling |
|------|----------|
| No semantic match (cosine_sim < 0.3) | Return top 3 anyway; let Layer 3 filter |
| Duplicate relations across candidates | Deduplicate while preserving order (keep highest score) |
| Relation candidates too generic ("is related to") | Increase top_k to 10 for recall; GTE handles noise |
| Empty relation_list | Return empty dict; fail fast to Layer 3 with insufficient info |
| All similarities below 0.25 | Return top 3 with low-confidence flag; Layer 3 will reject if unusable |
| Relation phrases match same KG relation | Deduplicate to single entry |
| GTE service unavailable | Fall back to lexical keyword matching (split and match tokens) |
| Relation name casing mismatch | Case-insensitive comparison; normalize to lowercase |

## Layer 3: Subgraph Reasoning

| Case | Handling |
|------|----------|
| No path from anchor to answer | "Cannot be determined from provided subgraph" |
| Multiple valid answers | Return all, comma-separated (e.g., "French, German, Spanish") |
| Target entity constraint not satisfied | "No valid path to target entity [X]" |
| Cyclic paths in subgraph | Follow the path matching most intent steps; depth ranking handles this |
| Ambiguous entity mentions in subgraph | Use disambiguation from subgraph context (relation types guide selection) |
| Empty subgraph | "Insufficient information to answer" |
| Disconnected subgraph components | Follow the component containing the anchor |
| Multiple paths to different answers | Prefer deeper path; if same depth, return all answers |
| Temporal value extraction | Parse from terminal entity; return value not entity name (2014 not "2014 World Series") |
| Numeric constraints in path | Verify the value matches question constraint; otherwise reject path |

## Backend: Subgraph Construction

| Case | Handling |
|------|----------|
| No neighbors found at step | Prune that branch; continue with other intent branches |
| Max depth exceeded | Return deepest partial path found; depth-based retention handles this |
| Target entity not in final graph | Return paths without target; Layer 3 will report constraint failure |
| Memory limit on subgraph | Prioritize by path depth first, then by confidence/GTE score |
| Multiple anchor matches in entity pool | Use entity with most outgoing relations (highest centrality) |
| CVT node explosion | Expand CVT locally only; restrict to top-5 neighbors by confidence |
| Relation candidate list empty | Skip expansion for that intent step; continue with other steps |

## System-Level Failures

| Case | Handling |
|------|----------|
| Layer 1 produces invalid JSON | Catch exception; return structured error with retry hint |
| Layer 2 GTE model unavailable | Fall back to keyword/lexical matching; degrade gracefully |
| Layer 3 LLM timeout | Return subgraph as structured fallback; expose raw triples |
| KG query failure (backend) | Return "Knowledge base unavailable" with error code |
| End-to-end latency > 10s | Implement caching for frequent questions |
| Prompt injection detected | Sanitize input; reject malformed questions |
| Rate limiting on GTE service | Queue requests; return partial results if timeout |

## Integration Edge Cases

| Case | Handling |
|------|----------|
| Layer 1 returns empty intent_chain | Reject; require at least 1 intent step |
| Layer 2 returns empty relations for all phrases | Use full subgraph (no constraint); let Layer 3 find answer |
| Layer 3 receives subgraph with only anchor | Answer is anchor itself only if question asks for anchor property |
| Target entity in Layer 1 but not in subgraph | Layer 3 will report "No valid path to target entity" |
| Anchor entity not in subgraph | Backend error; return "Anchor entity not found in knowledge base" |
| Relation candidates match >50 relations | Cap at top 50 per phrase; Layer 3 filters further |

## Confidence Scoring

| Layer | Confidence Metric | Action |
|-------|-------------------|--------|
| Layer 1 | Valid JSON + all required fields | Proceed; else retry |
| Layer 2 | Avg cosine_sim across all matches | <0.4: warn but proceed; <0.25: flag low confidence |
| Layer 3 | Path depth >= intent_chain length | High confidence; else medium |
| End-to-end | All layers complete + path exists | Return answer with confidence flag |
