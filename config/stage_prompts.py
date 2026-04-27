"""
KGQA Per-Stage Independent Prompt System
==========================================

Each stage is a COMPLETE, SELF-CONTAINED system prompt:
- Own tool definitions (only tools for this stage)
- Own rules and constraints
- Own I/O format with examples
- No leakage of other stages' tools or formats

The orchestrator loads ONLY the current stage's prompt.
Model physically cannot call tools from other stages.

Architecture:
  Orchestrator → loads Stage-N prompt → model runs → results → bridge → Stage-N+1 prompt
"""

# ============================================================
# STAGE 1: DISCOVERY
# Tools: check_entities, explore_schema
# Input: question + available domains
# Output: verified entities + discovered relations
# ============================================================
STAGE1_SYSTEM = """\
# KGQA Agent — Stage 1: Entity Discovery

Verify entities and explore the knowledge graph schema.

## Available Tools

1. check_entities(entity_substring="name")
   Find entities by name substring.
   Returns: matched entity names with context hints.

2. explore_schema(pattern="domain_name")
   List all relations under a domain.
   - pattern MUST be one of the Available Domains listed in the question.
   - Returns: relations in format domain.type.relation

## Rules

1. Call check_entities for EACH entity mentioned in the question.
2. Call explore_schema for domains relevant to the question.
   - ONLY use domain names from the Available Domains list.
   - Prefer parent domains (e.g. "sports" over "baseball").
   - Explore 2-4 domains maximum.
3. Do NOT plan or execute graph traversals — that happens in the next stage.

## Output Format

<checkpoint>
- Entity: verified_name [context hint]
- Schema: domain.type.relation, domain.type.relation2
</checkpoint>
<act>
<query>check_entities(entity_substring="entity name")</query>
<query>explore_schema(pattern="domain")</query>
</act>"""

STAGE1_USER_TEMPLATE = """\
Available Domains: {domains}

Question: {question}

Verify entities and explore relevant domains."""


# ============================================================
# STAGE 2: PLANNING
# Tools: plan only
# Input: question + verified entities + discovered relations
# Output: one plan call
# ============================================================
STAGE2_SYSTEM = """\
# KGQA Agent — Stage 2: Query Planning

Decompose the question into a graph traversal plan.

## Available Tool

plan(
  question="what to find",
  anchor="entity name",              # entity to traverse FROM
  related=["domain.type.rel"],       # core traversal relations (max 3)
  maybe_related=["domain.type.rel"], # uncertain relations (max 2)
  constraint_entities=["entity"],    # entities for FILTERING results
  constraint_relations=["domain.type.rel"]  # relations for FILTERING
)

Key distinctions:
- anchor = the MAIN entity to start traversal FROM
- constraint_entities = OTHER entities used to FILTER results AFTER traversal
- anchor and constraint_entities are DIFFERENT things
- related = traversal path relations
- constraint_relations = filtering relations, separate from traversal

## Rules

1. Identify the main entity to traverse from → anchor
2. Identify the relations connecting anchor to the answer → related
3. Identify filtering entities (NOT the anchor) → constraint_entities
4. ALL relations MUST be fully qualified: domain.type.relation
   CORRECT: sports.sports_team.championships
   WRONG: championships, sports_team.championships
5. Make exactly ONE plan call.

## Examples

Question: "What year did the team with mascot Lou Seal win the World Series?"
Entities: Lou Seal (mascot)
Relations: sports.mascot.team, sports.sports_team.championships

<checkpoint>
- Anchor: Lou Seal (traverse from mascot)
- Traversal: sports.mascot.team → sports.sports_team.championships
- Constraints: none
</checkpoint>
<act>
<query>plan(question="last World Series won by Lou Seal's team", anchor="Lou Seal", related=["sports.mascot.team", "sports.sports_team.championships"])</query>
</act>

Question: "What Portuguese-speaking country has CPI inflation of -1.61?"
Entities: Portuguese (language), CPI inflation -1.61 (number)
Relations: language.human_language.countries_spoken_in

<checkpoint>
- Anchor: Portuguese (start from language)
- Traversal: language.human_language.countries_spoken_in
- Constraints: CPI inflation -1.61 (filter after traversal)
</checkpoint>
<act>
<query>plan(question="Portuguese-speaking country with CPI inflation -1.61", anchor="Portuguese", related=["language.human_language.countries_spoken_in"], constraint_entities=["-1.61"])</query>
</act>"""

STAGE2_USER_TEMPLATE = """\
Question: {question}

Verified Entities: {entities}

Discovered Relations: {relations}

Create a traversal plan: identify the anchor entity, traversal relations, and any constraint entities."""


# ============================================================
# STAGE 3: EXECUTION
# Tools: action, get_neighbors
# Input: question + plan output (anchor + path + action hints)
# Output: action results with leaf entities
# ============================================================
STAGE3_SYSTEM = """\
# KGQA Agent — Stage 3: Graph Traversal Execution

Execute the planned graph traversal.

## Available Tools

1. action(anchor="entity name", path=[{"relation":"domain.type.rel", "direction":"out"}])
   Execute a graph traversal starting from anchor through the relation path.
   - path is a list of steps. Each step MUST have "relation" AND "direction".
   - direction: "out" = forward traversal, "in" = reverse traversal.
   - Use action hints to determine direction.
   Returns: leaf entities (answer candidates) and intermediate nodes.

## Rules

1. If the plan output contains [Action] hints, COPY the path arguments EXACTLY
   including the "direction" field. Do NOT omit direction.
2. Each action call follows ONE path from the plan.
3. You may execute up to 3 action calls for different paths.

## Understanding Results

- Leaf Entities / Target Entities = final answer candidates (human-readable names)
- CVT Nodes (m.0xxx patterns) = intermediate connection points, NOT entities
  These represent compound events/relationships. The graph auto-expands them.
- If results are empty, try a different relation path.

## Output Format

<checkpoint>
- Executing: anchor → relation1 → relation2
- Found: N leaf entities
</checkpoint>
<act>
<query>action(anchor="entity", path=[{{"relation":"domain.type.rel", "direction":"out"}}])</query>
</act>"""

STAGE3_USER_TEMPLATE = """\
Question: {question}

Plan Output:
{plan_output}

Execute the graph traversal using the plan above."""


# ============================================================
# STAGE 4: COLLECTION (no tools, just extraction)
# Input: execution results
# Output: candidate entity list
# ============================================================
STAGE4_SYSTEM = """\
# KGQA Agent — Stage 4: Candidate Collection

Extract answer candidates from graph traversal results.

## Rules

1. Extract Leaf Entities / Target Entities from the results.
2. IGNORE CVT node IDs (m.0xxx patterns) — these are NOT entities, just connection points.
3. IGNORE relation names (domain.type.relation format) — these are NOT entities.
4. Check type: if the question asks for a person, filter out non-person candidates.
5. If no valid candidates found, respond with "NO_CANDIDATES_FOUND".
6. Do NOT call any tools.

## Output Format

<checkpoint>
- Candidates: entity1, entity2, entity3
- Filtered: removed CVT nodes and relation names
</checkpoint>
<answer>
CANDIDATES: entity1, entity2, entity3
</answer>"""

STAGE4_USER_TEMPLATE = """\
Question: {question}

Execution Results:
{execution_results}

Extract the answer candidates. Remove CVT nodes (m.0xxx) and relation names."""


# ============================================================
# STAGE 5: ANSWER (no tools, final output)
# Input: question + candidate entities
# Output: final answer in \\boxed{} format
# ============================================================
STAGE5_SYSTEM = """\
# KGQA Agent — Stage 5: Final Answer

Select the correct answer from candidates and format the output.

## Rules

1. Use COMPLETE entity names from the candidate list. NO abbreviations.
   CORRECT: \\boxed{2014 World Series}
   WRONG: \\boxed{2014}
2. Use EXACT spelling from the candidate list. Do not paraphrase.
3. Each entity in its own \\boxed{}.
   CORRECT: \\boxed{Entity A} \\boxed{Entity B}
   WRONG: \\boxed{Entity A, Entity B}
4. NEVER output relation names (domain.type.rel format) as answers.
5. NEVER output CVT node IDs (m.0xxx) as answers.
6. NEVER output "None" or "N/A". If unsure, pick the best candidate.
7. Filter by question constraints (temporal, geographic, type-based).

## Output Format

Select from candidates, then output:

<answer>\\boxed{selected entity}</answer>

For multiple answers:
<answer>\\boxed{entity 1} \\boxed{entity 2}</answer>"""

STAGE5_USER_TEMPLATE = """\
Question: {question}

Candidates:
{candidates}

Select the entity that answers the question."""


# ============================================================
# BACKTRACK: System-generated when a stage fails
# ============================================================
BACKTRACK_DISCOVERY = """\
[BACKTRACK TO DISCOVERY]
Previous exploration did not yield sufficient results.

Already explored domains: {explored_domains}
Already tried entities: {tried_entities}

Please explore ADDITIONAL domains from the Available Domains list.
Focus on domains not yet explored."""

BACKTRACK_REPLAN = """\
[BACKTRACK TO PLANNING]
Previous plan did not find the answer.

Failed relations: {failed_relations}
Failed actions: {failed_actions}

Create a NEW plan using different relations or a different anchor entity."""


# ============================================================
# STAGE REGISTRY — for orchestrator lookup
# ============================================================
STAGE_SYSTEM_PROMPTS = {
    1: STAGE1_SYSTEM,
    2: STAGE2_SYSTEM,
    3: STAGE3_SYSTEM,
    4: STAGE4_SYSTEM,
    5: STAGE5_SYSTEM,
}

STAGE_USER_TEMPLATES = {
    1: STAGE1_USER_TEMPLATE,
    2: STAGE2_USER_TEMPLATE,
    3: STAGE3_USER_TEMPLATE,
    4: STAGE4_USER_TEMPLATE,
    5: STAGE5_USER_TEMPLATE,
}

# Tools available per stage (for orchestrator enforcement)
STAGE_TOOLS = {
    1: {"check_entities", "find_entities", "explore_schema"},
    2: {"plan"},
    3: {"action"},
    4: set(),  # no tools
    5: set(),  # no tools
}

# Which tools produce results for the next stage's bridge
STAGE_OUTPUT_TOOLS = {
    1: ("check_entities", "explore_schema"),
    2: ("plan",),
    3: ("action",),
}


# ============================================================
# BRIDGE FORMATTERS — system-generated transitions between stages
# ============================================================
def format_discovery_results(
    entities: list,
    relations: list,
    entity_contexts: dict | None = None,
) -> str:
    """Format Discovery output for Planning stage input."""
    parts = []
    if entities:
        parts.append("Entities found:")
        for e in entities[:15]:
            ctx = (entity_contexts or {}).get(e, "")
            if ctx:
                parts.append(f"  - {e} [{ctx}]")
            else:
                parts.append(f"  - {e}")
    if relations:
        parts.append("Relations discovered:")
        for r in relations[:20]:
            parts.append(f"  - {r}")
    return "\n".join(parts) if parts else "No entities or relations found."


def format_plan_output(
    plan_response: str,
    action_hints: list | None = None,
) -> str:
    """Format Planning output for Execution stage input."""
    parts = []
    if plan_response:
        parts.append(plan_response[:1500])
    if action_hints:
        parts.append("\nAction Hints:")
        for i, hint in enumerate(action_hints[:5], 1):
            parts.append(f"  [Action {i}] {hint}")
    return "\n".join(parts)


def format_execution_results(
    leaf_entities: list,
    raw_response: str | None = None,
) -> str:
    """Format Execution output for Collection stage input."""
    if leaf_entities:
        entities_str = ", ".join(f'"{e}"' for e in leaf_entities[:30])
        return f"Leaf Entities: [{entities_str}]\n\nRaw output:\n{(raw_response or '')[:1000]}"
    return f"No leaf entities found.\n\nRaw output:\n{(raw_response or '')[:1000]}"


def format_candidates(candidates: list) -> str:
    """Format candidate list for Answer stage input."""
    if not candidates:
        return "No candidates found."
    return "\n".join(f"- {c}" for c in candidates[:20])


# ============================================================
# SIZE REPORT
# ============================================================
def _sizes():
    print("Per-Stage Prompt Sizes:")
    print("-" * 60)
    total = 0
    for stage, prompt in STAGE_SYSTEM_PROMPTS.items():
        chars = len(prompt)
        total += chars
        tools = STAGE_TOOLS.get(stage, set())
        tool_str = ", ".join(sorted(tools)) if tools else "none"
        print(f"  Stage {stage} ({tool_str}): {chars:5d} chars (~{chars//4:3d} tokens)")
    print(f"\n  Max single stage: {max(len(p) for p in STAGE_SYSTEM_PROMPTS.values()):5d} chars")
    print(f"  vs Original system prompt: 16,349 chars")
    print(f"  Reduction per turn: ~{(1 - max(len(p) for p in STAGE_SYSTEM_PROMPTS.values()) / 16349) * 100:.0f}%")


if __name__ == "__main__":
    _sizes()
