# Prompt Redundancy Map — Baseline Analysis
# Generated: 2026-04-05
# Purpose: A/B comparison framework for prompt consolidation

## Injection Flow (per turn)

Each turn the model receives:
1. Tool results (backend output)
2. Stage transition prompt (from plug_v12_feedback.py)
3. Skill hint (from retriever.py, injected by runtime.py)
4. Protocol-guard reminder (small, per-turn)

## Redundancy Summary

| # | Guidance Rule | Stage 1→2 | Stage 2→3 | Stage 3→4 | Skill Reasoning Hint | Stage 5 | Total | Home Stage |
|---|---|---|---|---|---|---|---|---|
| 1 | "Stay in ONE primary action space" | - | prefer ONE | Select ONE | "stay inside one primary" | "Stay inside ONE primary" | 4x | 3→4 |
| 2 | "Judge WHOLE path semantics" | - | ✓ CORE RULE | "WHOLE path semantics" | - | "WHOLE path semantics best matches" | 3x | 2→3 |
| 3 | "Do not merge action spaces" | - | - | "Do not merge by default" | "Do not merge different" | "Do NOT merge multiple" | 3x | 3→4 |
| 4 | "Filter/collapse inside action space" | constraint analysis | - | OPTION A/B filter | "keep all or filter subset" | "FILTER INSIDE PRIMARY" (6 lines) | 3x | 3→4 |
| 5 | "Use EXACT graph strings" | Spelling Check | - | - | "Use exact graph strings only" | "EXACT graph string" (4 lines) | 3x | 5 |
| 6 | "Knowledge vs Graph evidence" | - | - | - | "Skill NOT override" | "KNOWLEDGE VS GRAPH" (3 lines) | 2x | 5 |
| 7 | "Do not force collapse" | - | - | - | "Do not force collapse" | "Do NOT force a collapse" | 2x | 5 |
| 8 | `\boxed{Exact Graph String}` template | - | - | - | - | output_format + template | echo bug | 5 |

## Token Budget Estimate (per turn with skills)

| Source | Stage 1→2 | Stage 2→3 | Stage 3→4 | Stage 5 |
|---|---|---|---|---|
| Stage prompt | ~500 | ~400 | ~450 | ~700 |
| Skill hint | ~300 | ~200* | - | ~300 |
| Total | ~800 | ~600* | ~450 | ~1000 |
| *with action hints enabled | | | | |

## Consolidation Plan

### Principle
Each rule appears at exactly ONE stage (its "home"). Skill hints provide ONLY concrete data, not reasoning rules.

### Home Stage Assignments

**Stage 1→2 (Plan)** — Answer-type reasoning
- Keep all: ANSWER-TYPE-FIRST, CONSTRAINT ANALYSIS, REFLECTION CHECKPOINT
- No changes needed (already clean)

**Stage 2→3 (Action)** — Action selection reasoning
- Home for: "Judge WHOLE path semantics", "Prefer ONE primary action_id"
- Keep all CORE RULE
- No changes needed

**Stage 3→4 (Refine)** — Winner selection + filter
- Home for: "Stay in ONE action space", "Do not merge", "Filter inside action space"
- Keep all STEP 1 + STEP 2
- No changes needed

**Stage 5 (Final Reasoning)** — Format output + evidence evaluation
- Home for: "EXACT graph strings", "Knowledge vs Graph", "Do not force collapse"
- DELETE: Core Protocol #1 (action space selection → already decided at 3→4)
- DELETE: Core Protocol #2 (filter analysis → already done at 3→4)
- DELETE: Reflection items about action space / filter
- FIX: `\boxed{Exact Graph String}` → replace with `\boxed{Entity Name Here}`
- SIMPLIFY reasoning template: remove STEP 1/2/3 (already done), keep SPELLING only

### Skill Hint Changes

**build_relation_stage_hint()** — NO CHANGE
- Already provides only concrete data

**build_action_stage_hint()** — MINOR TRIM
- DELETE last 2 lines: "Skill experience does NOT override" / "prefer graph evidence"

**build_reasoning_stage_hint()** — MAJOR TRIM
- DELETE: "Primary action-space policy" block (3 lines) → redundant with 3→4
- DELETE: "Fixed output protocol" block (5 lines) → redundant with Stage 5
- KEEP ONLY: concrete final_selection_experience lines + one-line header

### Expected Impact
- Stage 5 token reduction: ~200-300
- Skill reasoning hint token reduction: ~150
- Total redundant guidance eliminated: ~400 tokens per turn 4-5
- Each rule appears 1x instead of 2-4x
- `\boxed{}` echo bug fixed
