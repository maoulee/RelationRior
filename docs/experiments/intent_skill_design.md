# Intent-Aware Skill System Design

## Baseline Configuration (Best Performance on diag12)

```bash
# Core inference
KGQA_ENABLE_DECISION_CONSISTENCY=1
KGQA_CONSISTENCY_MODE=selective          # verify-then-search (zero drift)
KGQA_ENABLE_WEB_SEARCH=1                 # websearch rescue on disagreement
KGQA_ENABLE_INTENT_ANALYSIS=1            # NEW: intent-aware skill injection

# LLM settings
KGQA_LLM_API_URL=http://127.0.0.1:8000/v1
KGQA_MODEL_NAME=qwen35-9b-local
KGQA_ENABLE_THINKING=0
KGQA_LLM_TEMPERATURE=0.1

# Skill settings
KGQA_SKILL_REASONING_INJECTION_MODE=per_skill
KGQA_SKILL_AUDIT_MODE=conflict_only
```

## Results on diag12 (12 cases)

| Config | Avg F1 | Hit@1 | EM | FE |
|---|---|---|---|---|
| Baseline (no consistency) | 0.6156 | — | — | — |
| Old probe-first selective | 0.6627 | 0.8333 | 0.3333 | 2 |
| Verify-then-search + websearch | 0.6639 | 0.7500 | 0.5000 | 6 |
| **+ Intent analysis** | **0.7238** | **0.8333** | **0.5833** | **1** |

## Architecture

### Files Modified/Created
- `src/subgraph_kgqa/skill_mining/schemas.py` — 3 new dataclasses + CaseSkillCard fields
- `src/subgraph_kgqa/skill_mining/intent_analyzer.py` — NEW: intent analysis module
- `src/subgraph_kgqa/inference/decision_consistency.py` — selective mode rewrite (verify-then-search)
- `scripts/run_skill_enhanced_test.py` — integration + verification stage

### Feature Gating
All new features are OFF by default:
- `KGQA_ENABLE_INTENT_ANALYSIS=0` (default) — intent analysis disabled
- `KGQA_ENABLE_WEB_SEARCH=0` (default) — websearch rescue disabled
- `KGQA_ENABLE_DECISION_CONSISTENCY=0` (default) — consistency disabled

### Rollback
Set `KGQA_ENABLE_INTENT_ANALYSIS=0` to revert to pre-intent behavior. No changes to runtime.py.
