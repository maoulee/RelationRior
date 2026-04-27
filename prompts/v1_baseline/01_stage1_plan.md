# Stage 1→2: Build Retrieval Plan
# Source: plug_v12_feedback.py _stage1_planning() (full version, not checklist/compact)

```
[STAGE 1→2: BUILD RETRIEVAL PLAN]

Discovery complete. Now construct your retrieval plan.

━━━ CORE PHILOSOPHY ━━━
The knowledge graph is human-constructed. There is NO perfect relation for every question.
Select the BEST AVAILABLE option. NEVER refuse because options are imperfect.

━━━ THINKING STEPS (ANSWER-TYPE-FIRST) ━━━

1. ANSWER TYPE ANALYSIS (CRITICAL)
   - What TYPE of entity is the answer? (Person? Country? Date? Role?)
   - Find relations whose OUTPUT TYPE matches the answer type.
   - These are your PRIMARY candidates for `related`.

2. STARTING POINT ANALYSIS
   - Which verified entity is the anchor point?
   - Find relations that connect the anchor to the answer type.

3. RELATION PRIORITIZATION
   - `related`: Relations whose OUTPUT TYPE = Answer Type (highest priority)
   - `maybe_related`: Intermediate hops or alternative paths

4. CONSTRAINT ANALYSIS (CHAIN OF THOUGHT)
   **Hypothesis**: Assume there are multiple candidates for the answer. How do we distinguish the correct one?
   
   A. **Entity Aspect**:
      - Does the question require the answer to have a connection to another specific entity?
      - If yes, verify if that entity exists in tool outputs.
      - Input into `constraint_entities` ONLY if it is a verified non-anchor entity that directly constrains the answer.
      - Do NOT treat every non-anchor question mention as a constraint entity.
   
   B. **Attribute Aspect**:
      - Does the question require specific characteristics (e.g., gender, time, location)?
      - How does the correct answer differ from other potential candidates?
      - Analyze this key attribute and checks relations from `explore_schema`.
      - If a suitable relation exists, input into `constraint_relations`.

━━━ REFLECTION CHECKPOINT (CRITICAL) ━━━
Before calling plan(), verify EACH field with these checks:

1. **Spelling Check**:
   - MUST be EXACTLY what the tool output provided.
   - Cannot be partial or invented. Double check every character.

2. **Exclusion Check**:
   - `constraint_entities` MUST NOT be the `anchor`.
   - `constraint_relations` MUST NOT be in `related` or `maybe_related`.
   - A non-anchor mention is a `constraint_entity` only if it directly constrains the answer set.

3. **Field Verification**:
   □ anchor: From check_entities EXACT match or Oracle Entities?
   □ related: Does its OUTPUT TYPE match the Answer Type?
   □ constraint_entities: Distinct from anchor? Valid tool output? Directly answer-constraining?
   □ constraint_relations: Distinct from related/maybe_related?

━━━ OUTPUT FORMAT ━━━
✅ ALLOWED: plan()
❌ FORBIDDEN: action(), <answer>

<reasoning>
  [ANSWER TYPE]
  - Answer type: ...
  
  [STARTING POINT]
  - Anchor: ...
  
  [RELATION PRIORITIZATION]
  - related: ...
  - maybe_related: ...
  
  [CONSTRAINT ANALYSIS]
  - Multi-answer Hypothesis: If multiple answers exist...
  - Entity Aspect: Only verified, non-anchor, answer-constraining entities -> constraint_entities
  - Attribute Aspect: Key distinguishing characteristic? -> constraint_relations
  
  [REFLECTION CHECKLIST]
  □ Spelling Check: All inputs match tool output exactly
  □ Exclusion Check: Constraints are distinct from anchor/related
  □ All fields verified OK
</reasoning>
<act>
  <query>plan(question="...", ...)</query>
</act>
```

## Injected Skill Hint: build_relation_stage_hint()
# Source: retriever.py:726-813

```
[RETRIEVED SKILL EXPERIENCE: RELATION CANDIDATES]
Below are relation candidates aggregated from similar solved questions.
Treat them as action-space priors only.
Use a skill relation only if it also appears in the CURRENT explored schema or CURRENT suggested relations.

- Frequent domains in similar questions: `{domains}`
- Candidate relations from similar questions:
  - `{relation}` (seen in {count} similar question(s))
    Example questions: `{q1}`; `{q2}`

- Possible second-entity / title constraints from similar questions:
  - `{entity}` (seen in {count} similar question(s))
    Example questions: `{q1}`; `{q2}`
  - Only use these when the CURRENT question clearly contains or verifies a matching second entity / title / franchise.

- If a skill relation is absent from the CURRENT schema, ignore it.
- Use the remaining valid relations as `related` / `maybe_related` candidates only.
```
