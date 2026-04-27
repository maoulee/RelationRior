# Stage 2→3: Select Action Space
# Source: plug_v12_feedback.py _stage2_planning() (action_id variant — most common in production)

```
[STAGE 2→3: SELECT ACTION SPACE]

Plan executed. {action_count} action candidates generated.
Your task: choose the best action_id(s) and execute them.

━━━ CORE RULE ━━━
- Do NOT rewrite the path by hand.
- Use the returned action_id only.
- Read the Logic Pattern and Analogical Example before choosing.
- Choose by WHOLE path semantics and returned answer type, not by a single tempting relation token.
- Prefer ONE primary action_id.
- You may execute up to THREE action_ids only if multiple options are genuinely promising and still semantically plausible.
- If none of the options fit, re-plan. Do NOT construct a replacement path.

Available Action IDs:
- {preview}

━━━ OUTPUT FORMAT ━━━
✅ ALLOWED: one to three `select_action(action_id="...")` calls
❌ FORBIDDEN: inventing action() paths, <answer>

<reasoning>
  [ACTION SPACE ANALYSIS]
  - A1: ...
  - A2: ...

  [SELECTION]
  - Selected: A1 / A2 because ...
</reasoning>
<act>
  <query>select_action(action_id="A1")</query>
</act>
```

## Injected Skill Hint: build_action_stage_hint() [GATED by KGQA_ENABLE_ACTION_STAGE_HINTS]
# Source: retriever.py:953-1033

```
[RETRIEVED SKILL EXPERIENCE: ACTION SELECTION]
Below are action-selection experiences from similar solved questions.
Use them as soft priors to avoid common action errors.

- Action-space experience:
  - {concrete experience line from cards}

- Common pitfalls to avoid:
  - {pitfall line from cards}

- Constraint guidance:
  - {constraint line from cards}

- Temporal awareness:
  - {temporal hint from cards}

Skill experience does NOT override current graph evidence.
If the graph suggests a different action, prefer graph evidence.
```
