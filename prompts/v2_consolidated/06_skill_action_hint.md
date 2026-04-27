# Skill Action Hint (V2 Consolidated)
# Source: retriever.py build_action_stage_hint()
# Changes from V1:
#   - DELETED last 2 generic lines: "Skill experience does NOT override..." / "If graph suggests..."
#   - These are redundant with Stage 5 Knowledge vs Graph section

## V2 Output:
```
[RETRIEVED SKILL EXPERIENCE: ACTION SELECTION]
Below are action-selection experiences from similar solved questions.
Use them as soft priors to avoid common action errors.

- Action-space experience:
  - {concrete experience line}

- Common pitfalls to avoid:
  - {pitfall line}

- Constraint guidance:
  - {constraint line}

- Temporal awareness:
  - {temporal hint}
```
