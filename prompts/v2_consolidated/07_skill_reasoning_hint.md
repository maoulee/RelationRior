# Skill Reasoning Hint (V2 Consolidated)
# Source: retriever.py build_reasoning_stage_hint()
# Changes from V1:
#   - DELETED "Primary action-space policy" block (3 lines) → redundant with Stage 3→4
#   - DELETED "Fixed output protocol" block (5 lines) → redundant with Stage 5
#   - KEEP ONLY: concrete final_selection_experience lines + header
#   - Kept anti-collapse line as it's the home stage for that rule

## V2 Output (per_skill mode):
```
[RETRIEVED SKILL EXPERIENCE: FINAL SELECTION]
Concrete selection experiences from similar solved questions:

- Selected skill experiences:
  - From `{question}`:
    - {experience line 1}
    - {experience line 2}
  - From `{question}`:
    - {experience line 1}
```

## V2 Output (aggregate mode):
```
[RETRIEVED SKILL EXPERIENCE: FINAL SELECTION]
Concrete selection experiences from similar solved questions:

- Final-selection experience from similar questions:
  - {experience line 1}
  - {experience line 2}
  - ...

- If graph-visible evidence does not distinguish candidates, do not force a single-answer collapse.
```
