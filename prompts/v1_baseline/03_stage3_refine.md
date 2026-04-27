# Stage 3→4: Select Action Space & Refine Candidates
# Source: plug_v12_feedback.py _stage3_execution() (full version)

```
[STAGE 3→4: SELECT ACTION SPACE & REFINE CANDIDATES]

Tools executed. Your task: Select the MOST RELEVANT action space (subgraph), and REFINE candidates if needed.

━━━ STEP 1: SELECT THE WINNING ACTION SPACE ━━━

1. SCAN ALL TOOL RESULTS
   - Identify the tool output that best represents the PRIMARY answer subgraph.
   - Discard tool outputs that are irrelevant or noisy.

2. RELEVANCE EVALUATION
   - Does this action space's WHOLE path semantics match the question?
   - Does it return the right answer type and main relation meaning?
   - If one action already captures the intended answer space, prefer it over looser alternatives.

3. SELECT THE WINNING SPACE
   - Select ONE winning action space.
   - Ignore candidates from discarded spaces.
   - Do not merge multiple action spaces by default.

━━━ STEP 2: CONSTRAINT CHECK (POST-HOC ANALYSIS) ━━━
**Precondition**: This step is ONLY needed if you did NOT specify constraints in the plan() stage.
If you already used constraint_relations/constraint_entities in plan(), the backend has auto-filtered.

MANDATORY RULE:
- If the selected action block explicitly shows `[Suggested Filter Relations]` and the selected action still has multiple candidates, you MUST call `filter()` before moving to final reasoning.
- Do not skip this just because entity names "look obvious".

[OPTION A: POST-HOC FILTERING NEEDED?]
- **When**: Plan had NO constraints, and the winning action space contains multiple candidates but the question implies a narrower subset.
- **Action**: Use `filter()` to inspect candidate attributes inside the selected action space.

[OPTION B: PROCEED DIRECTLY]
- **When**: Plan already had constraints (filtering done), OR all non-conflicting candidates from the winning action should remain.
- **Action**: Extract candidates using <candidates> tag.

━━━ OUTPUT FORMAT ━━━
Choose ONE based on your Analysis:

[OPTION A: POST-HOC REFINE]
<reasoning>
  [SELECTION]
  - Winner: Action Space N (relation X)
  [ANALYSIS]
  - Plan had no constraints, need post-hoc filtering.
  - Constraint to verify: "..."
</reasoning>
<act>
  <candidates>
    - Entity 1
    - Entity 2
  </candidates>
  <query>filter(...)</query>
</act>

[OPTION B: PROCEED]
<reasoning>
  [SELECTION]
  - Winner: Action Space N (relation X)
  [ANALYSIS]
  - Constraints: Already applied in plan / Not needed.
  - Action: Extracting candidates for final reasoning.
</reasoning>
<act>
  <candidates>
    - Entity 1
    - Entity 2
  </candidates>
</act>
```

## NO Skill Hint at this stage (action hint only injected at stage 2/3)
