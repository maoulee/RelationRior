# V2 Architecture: Skill Aggregation + Audit Agent

## Overview

Three-stage enhancement:
1. **Pre-reasoning: Skill Aggregation** - LLM synthesizes retrieved skills into unified guidance
2. **Reasoning: Context Cleanup** - Strip reasoning drafts after each turn
3. **Post-reasoning: Audit Agent** - Replace critique with flexible audit + websearch

Plus: Enhanced skill mining with stability detection.

## Task Breakdown

### Complex (Codex)
- T1: Skill Aggregation module (`skill_aggregator.py`)
- T2: Audit Agent module (`audit_agent.py`) - replace critique
- T3: Integration into `run_skill_enhanced_test.py`

### Medium (Local Team)
- T4: Context cleanup logic (strip `<reasoning>` blocks)
- T5: Skill mining stability detection
- T6: Testing + analysis

## Key Files

### New Files
- `src/subgraph_kgqa/skill_mining/skill_aggregator.py` - Skill aggregation
- `src/subgraph_kgqa/inference/audit_agent.py` - Audit agent (replaces critique in selective mode)

### Modified Files
- `scripts/run_skill_enhanced_test.py` - Integration
- `src/subgraph_kgqa/inference/decision_consistency.py` - Remove critique, delegate to audit_agent
- `src/subgraph_kgqa/skill_mining/` - Enhanced mining with stability

## Environment Variables
- `KGQA_ENABLE_SKILL_AGGREGATION=1` - Enable skill aggregation stage
- `KGQA_ENABLE_AUDIT_AGENT=1` - Enable audit agent (replaces critique)
- `KGQA_ENABLE_CONTEXT_CLEANUP=1` - Enable reasoning draft cleanup

## Rollback
All features gated by env vars. Set to 0 to revert to current baseline:
```bash
KGQA_ENABLE_SKILL_AGGREGATION=0
KGQA_ENABLE_AUDIT_AGENT=0
KGQA_ENABLE_CONTEXT_CLEANUP=0
KGQA_ENABLE_DECISION_CONSISTENCY=1  # critique still works as fallback
```
