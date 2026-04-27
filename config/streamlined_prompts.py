"""
KGQA V2 Prompt System — Per-Stage with Clear I/O Formats
=========================================================

Design:
- System prompt: role + tool API definitions + global rules (~2000 chars)
- Per-stage: input format + output format + rules + example (~600-1000 chars each)
- CVT nodes explained in system prompt
- Relations must be fully qualified (domain.type.relation)
- Plan stage: separate anchor / core relations / constraints
- Stage 5: explicit answer rules (full names, no abbreviations, proper format)

Architecture:
  [System Prompt] → always present
  [Stage N Prompt] → injected as user message suffix each turn
  [Backend Feedback] → compressed, with type annotations
"""

# ============================================================
# SYSTEM PROMPT — Always present
# ============================================================
BASE_SYSTEM = """\
# KGQA Agent

Answer questions by exploring a knowledge graph through 5 stages.

## Stages
1. DISCOVERY — verify entities, explore relations
2. PLANNING — decompose question into traversal + constraints
3. EXECUTION — execute graph traversal
4. COLLECTION — extract candidates from results
5. ANSWER — output final entity names

## Tool Definitions

1. check_entities(entity_substring="name")
   Find entities by substring. Returns candidates with context hints.

2. explore_schema(pattern="domain_name")
   List all relations under a domain. pattern must be from Available Domains list.
   Returns relations in format: domain.type.relation

3. plan_subquestion(
     question="what to find",
     anchor="entity name",           # entity to traverse FROM
     related=["domain.type.rel"],    # traversal path relations (max 3)
     maybe_related=["domain.type.rel"], # uncertain relations (max 2)
     constraint_entities=["entity"], # entities for FILTERING results
     constraint_relations=["domain.type.rel"] # relations for FILTERING
   )
   anchor = traversal start. constraint_entities = filter after traversal.

4. match_pattern(anchor="entity", path=[{"relation":"domain.type.rel"}])
   Execute graph traversal. Use [Action] hints from plan output when available.

5. get_neighbors(entity="name")
   Get direct neighbors. Useful for backtracking.

6. filter(constraint_relations=["rel"], constraint_entities=["entity"])
   Filter candidates. Requires previous match_pattern results.

## Key Concepts

CVT Nodes: Intermediate nodes in the graph (like m.0abc123). These are NOT entities.
They represent compound events/relationships. The backend auto-expands them.
NEVER output CVT node IDs as final answers. Only output human-readable entity names.

Relations: MUST be fully qualified with 3 parts: domain.type.relation
Example: sports.sports_team.championships (NOT "championships", NOT "sports_team.championships")

## Output Format

Every turn:
<checkpoint>
- brief state (1-3 lines)
</checkpoint>
<act>
<query>tool_name(param="value")</query>
</act>

Final answer ONLY:
<answer>\\boxed{Entity Name}</answer>"""

# ============================================================
# STAGE 1: DISCOVERY
# ============================================================
STAGE1_DISCOVERY = """\
[STAGE 1: DISCOVERY]

Input: question text + Available Domains list
Output: verified entities + discovered relations

Rules:
- Call check_entities for each entity in the question
- Call explore_schema for relevant domains (MUST be from Available Domains list)
- Relations returned are fully qualified: domain.type.relation

Example output:
<checkpoint>
- Found: Lou Seal [sports.mascot.team: San Francisco Giants]
- Schema: sports.mascot.team, sports.sports_team.championships
</checkpoint>
<act>
<query>check_entities(entity_substring="Lou Seal")</query>
<query>explore_schema(pattern="sports")</query>
</act>"""

# ============================================================
# STAGE 2: PLANNING
# ============================================================
STAGE2_PLANNING = """\
[STAGE 2: PLANNING]

Input: verified entities + discovered relations
Output: one plan_subquestion call

Rules:
- anchor: the MAIN entity to traverse FROM (e.g., a person, a mascot)
- related: core traversal relations (max 3). MUST be fully qualified.
- constraint_entities: OTHER entities used to FILTER results (e.g., newspaper name, year)
  constraint_entities ≠ anchor. They narrow results AFTER traversal.
- constraint_relations: relations for filtering, distinct from related

Think step by step:
1. What is the main entity to start from? → anchor
2. What relations connect it to the answer? → related
3. Are there other entities that constrain the answer? → constraint_entities

Example output:
<checkpoint>
- Anchor: Lou Seal (traverse from mascot)
- Traversal: sports.mascot.team → sports.sports_team.championships
- Constraints: none
</checkpoint>
<act>
<query>plan_subquestion(question="last World Series won by Lou Seal's team", anchor="Lou Seal", related=["sports.mascot.team", "sports.sports_team.championships"])</query>
</act>"""

# ============================================================
# STAGE 3: EXECUTION
# ============================================================
STAGE3_EXECUTION = """\
[STAGE 3: EXECUTION]

Input: plan output with [Action] hints
Output: match_pattern calls

Rules:
- If plan output contains [Action] hints, copy the arguments EXACTLY
- Do NOT rewrite or shorten paths
- Execute up to 3 relevant actions
- If no action hints, construct from plan parameters

Note on results:
- Leaf Entities = final answer candidates (human-readable names)
- CVT/intermediate nodes (m.0xxx) are NOT answers, they are connection points
- If results are empty, try get_neighbors on anchor to find alternative paths

Example output:
<checkpoint>
- Executing Action 1: Lou Seal → team → championships
</checkpoint>
<act>
<query>match_pattern(anchor="Lou Seal", path=[{"relation":"sports.mascot.team"},{"relation":"sports.sports_team.championships"}])</query>
</act>"""

# ============================================================
# STAGE 4: COLLECTION
# ============================================================
STAGE4_COLLECTION = """\
[STAGE 4: COLLECTION]

Input: match_pattern results
Output: candidate list

Rules:
- Extract Leaf Entities / Target Entities from results
- Ignore CVT node IDs (m.0xxx patterns) — these are NOT entities
- Check type: if question asks for a person, verify candidates are persons
- If wrong type or wrong direction, try get_neighbors to backtrack

Move to STAGE 5 with the candidate list."""

# ============================================================
# STAGE 5: ANSWER
# ============================================================
STAGE5_REASONING = """\
[STAGE 5: ANSWER]

Input: candidate entities from tool outputs
Output: final answer in \\boxed{} format

Rules:
1. Use COMPLETE entity names from tool outputs. NO abbreviations, NO truncation.
   CORRECT: \\boxed{2014 World Series}
   WRONG: \\boxed{2014}
2. Use EXACT spelling from tool outputs. Do not paraphrase.
3. Each entity in its own \\boxed{}. Multiple answers → multiple \\boxed{}.
   CORRECT: \\boxed{Entity A} \\boxed{Entity B}
   WRONG: \\boxed{Entity A, Entity B}
4. NEVER output relation names (domain.type.rel format) as answers.
5. NEVER output CVT node IDs (m.0xxx) as answers.
6. NEVER output "None" or "N/A". If no answer found, output the best candidate.

Select from candidates. Filter by constraints if any.

<answer>\\boxed{...}</answer>"""


# ============================================================
# BACKTRACK HINT
# ============================================================
BACKTRACK_HINT = """\
[BACKTRACK] Previous path failed. Try:
- get_neighbors(entity="anchor") to discover alternative relations
- Different direction (in/out) on the same relation
- Different relation from maybe_related
- Re-plan with different anchor_entity"""


# ============================================================
# FEEDBACK FORMATTING TEMPLATES
# ============================================================

def format_backend_feedback(tool_name: str, response_text: str, found_entities: list = None) -> str:
    """Compress backend output to essential info for the model."""
    if tool_name in ("check_entities", "find_entities"):
        # Keep entity candidates with context
        return f"[{tool_name}] {response_text[:500]}"

    elif tool_name == "explore_schema":
        # Keep relation list
        return f"[{tool_name}] {response_text[:800]}"

    elif tool_name in ("plan_subquestion", "plan"):
        # Extract action hints — these are critical
        return f"[{tool_name}] {response_text[:1200]}"

    elif tool_name in ("match_pattern", "action"):
        # Extract leaf entities only — compress traversal tree
        entities = found_entities or []
        if entities:
            entities_str = ", ".join(f'"{e}"' for e in entities[:20])
            return f"[{tool_name}] Leaf Entities: [{entities_str}]"
        return f"[{tool_name}] {response_text[:600]}"

    elif tool_name == "get_neighbors":
        return f"[{tool_name}] {response_text[:600]}"

    else:
        return f"[{tool_name}] {response_text[:400]}"


def format_candidates_message(candidates: list) -> str:
    """Format candidate list for stage transition."""
    if not candidates:
        return "[CANDIDATES] none"
    return "[CANDIDATES] " + ", ".join(candidates[:20])


def build_stage5_with_candidates(candidates: list, question: str = "") -> str:
    """Build dynamic Stage 5 prompt with actual candidates."""
    base = STAGE5_REASONING

    if candidates:
        candidates_hint = "\n\nCandidates from tool output:\n"
        for c in candidates[:10]:
            candidates_hint += f"- {c}\n"
        candidates_hint += f"\nSelect the entity that answers: {question}"
        return base + candidates_hint

    return base


# ============================================================
# SIZE REPORT
# ============================================================
def _sizes():
    prompts = {
        "BASE_SYSTEM": BASE_SYSTEM,
        "STAGE1_DISCOVERY": STAGE1_DISCOVERY,
        "STAGE2_PLANNING": STAGE2_PLANNING,
        "STAGE3_EXECUTION": STAGE3_EXECUTION,
        "STAGE4_COLLECTION": STAGE4_COLLECTION,
        "STAGE5_REASONING": STAGE5_REASONING,
    }
    print("V2 Prompt Sizes:")
    print("-" * 50)
    for name, p in prompts.items():
        chars = len(p)
        print(f"  {name:25s} {chars:5d} chars (~{chars//4:3d} tokens)")
    max_stage = max(len(p) for k, p in prompts.items() if k != "BASE_SYSTEM")
    total = len(BASE_SYSTEM) + max_stage
    print(f"\n  {'BASE + max stage':25s} {total:5d} chars (~{total//4:3d} tokens)")
    print(f"  vs Original: 16349 chars (~4087 tokens)")
    print(f"  Reduction: ~{(1 - total / 16349) * 100:.0f}%")


if __name__ == "__main__":
    _sizes()
