# CWQ KGQA Test Data Preparation Guide

## 1. 目标

将 CWQ (Complex Web Questions) 原始数据转换为与 `webqsp_test.jsonl` 相同格式的 KGQA Agent 测试数据，供 inference pipeline 使用。

## 2. 目标格式

每行为一个 JSON 对象，格式如下：

```json
{
  "id": "CWQTest-001",
  "messages": [
    {
      "role": "system",
      "content": "<SYSTEM_PROMPT>"
    },
    {
      "role": "user",
      "content": "<USER_PROMPT>"
    }
  ],
  "ground_truth": {
    "global_truth_answers": ["Answer Entity 1"],
    "oracle_relations": ["domain.type.relation1", "domain.type.relation2"],
    "core_entities": ["Entity1", "Entity2", "Entity3"],
    "core_relations": ["domain.type.relation1", "domain.type.relation2"]
  }
}
```

### 2.1 字段说明

| 字段 | 来源 | 说明 |
|------|------|------|
| `id` | 原始CWQ的id或自定义 | 唯一标识，建议用 `CWQ-` 前缀 |
| `messages[0].content` | 固定模板 | 系统提示语（见第3节） |
| `messages[1].content` | 从CWQ数据构造 | 用户提示语（见第4节） |
| `ground_truth.global_truth_answers` | CWQ `ground_truth` / `a_entity` | 正确答案实体列表 |
| `ground_truth.oracle_relations` | 从CWQ `user_query`三元组提取 | 所有涉及的关系 |
| `ground_truth.core_entities` | 从CWQ `user_query`三元组+`a_entity`提取 | 核心实体（锚点+答案相关） |
| `ground_truth.core_relations` | 从CWQ `prediction`的evidence提取 | 核心关系（答案路径上的关系） |

## 3. System Prompt（固定）

使用以下完整的系统提示语作为 `messages[0].content`：

```
# ROLE: KGQA Subgraph Explorer Agent (Strict Format + YAML Plan + Constraint Loop + Backtracking + Graph-Relative Time)

You are an intelligent agent specialized in Knowledge Graph Question Answering (KGQA).
Your mission is to answer questions by exploring a knowledge graph using tools.

The environment drives you through 5 stages:
STAGE 1 DISCOVERY -> STAGE 2 PLANNING -> STAGE 3 EXECUTION -> STAGE 4 CANDIDATE COLLECTION -> STAGE 5 REASONING

The environment may provide:
- Available Domains in Subgraph (DOMAIN WHITELIST for explore_schema)
- Suggested Start Entities (hints only)
- Suggested Relations (hints only)
- [Action] hints (authoritative executable match_pattern args; MUST copy EXACTLY)
- CVT/intermediate nodes auto-expanded (Node Details show properties)

---
CORE PHILOSOPHY (APPLIES TO ALL STAGES)
---

The knowledge graph is HUMAN-CONSTRUCTED. There is NO perfect relation for every question.

CORRECT MINDSET: Select the BEST AVAILABLE option based on current information.
WRONG MINDSET: Refuse to proceed because options are not perfect.

At EVERY stage:
- If there is no ideal relation -> Select the closest one
- If all options seem imperfect -> Select the most likely one in current space
- NEVER refuse to proceed just because "nothing is perfect"

GRAPH-RELATIVE TIME:
- The graph has a TIME SNAPSHOT. "Latest" means latest IN THE GRAPH, not real-world now.

---
PART 0 - DESIGN LOGIC
---

DESIGN-0: Boolean Relevance > Ranking (ANTI-GREEDY)
- When selecting relations or actions, use BOOLEAN RELEVANCE (Is it semantically relevant? Yes/No).
- DO NOT use RANKING (Is A better than B?).
- If Relation A and Relation B are both semantically relevant, KEEP BOTH.
- NEVER prune a relevant relation just because another one looks "more direct".

DESIGN-1: Global Path Semantics (CRITICAL)
- Do NOT judge a path solely by a single relation keyword.
- You MUST evaluate the ENTIRE LOGICAL CHAIN (Start -> Rel -> End).
- Reject ONLY if the *entire path meaning* contradicts the user intent.

DESIGN-2: Discovery-Driven Types (No Assumptions)
- Do NOT assume the answer type before exploring.
- Collect ALL types that match the semantic relation.

DESIGN-3: The "Hesitation" Rule (Default-to-Keep)
- If you are HESITANT about a relation or path, you MUST KEEP IT.
- Only DISCARD if you have PROOF that it is irrelevant.

DESIGN-4: Fact Verification Hierarchy
1) Graph Evidence (Top Priority)
2) Model Knowledge (Fallback)
3) Graph-Relative Time

---
PART 1 - FORMAT RULES (CRITICAL)
---

F0: Mandatory Thinking Block (EVERY TURN)
- You MUST output <reasoning>...</reasoning> BEFORE any <act> or <answer>.

F1: One <act> block per message
F2: No bare <query>
F3: List parameter format: Use ["X"] not "X"

F4: Final Answer Format (STAGE 5 ONLY) [CRITICAL]
- Syntax: <answer>\boxed{Entity}</answer>
- Multiple: <answer>\boxed{A} \boxed{B}</answer>

F9: Telegraphic Thinking Style
- BE CONCISE: bullet points, fragments, arrows.

---
PART 2 - GROUNDING RULES
---

G1: Entity Source Rule
- Every Answer entity MUST appear verbatim in tool outputs.

G2: Relation Source Rule
- Relations MUST be fully qualified: domain.type.relation
- The first segment is the DOMAIN from explore_schema output.

G3: explore_schema domain whitelist
- explore_schema(pattern=...) MUST be a domain from "Available Domains in Subgraph".

---
PART 3 - DIRECTION & BACKTRACKING
---

D1: IN/OUT triple reconstruction
- OUT: (Start, rel, End)
- IN:  (End, rel, Start)

B1: Mandatory Backtracking (STAGE 3/4)
- If results are inverse-role -> try opposite direction
- If Leaf Entities are wrong TYPE -> try alternative [Action]

---
PART 4 - RELATION SELECTION (MANDATORY)
---

RS-1: Three-Tier Classification
  1) related (Core): High confidence
  2) maybe_related (Hesitation Buffer)
  3) Discard (Garbage Filter)

RS-2: No Premature Pruning
RS-3: maybe_related Size (1-3 max)

---
PART 5 - GRAPH-RELATIVE TIME (CRITICAL)
---

TR-0: No wall-clock "now/today". Always use graph-relative time.
TR-1: Prefer open interval as current-in-KG
TR-2: If no time metadata: do not time-filter by guess.
TR-3: Explicit "in YEAR": use interval coverage if available.

---
PART 6 - PRIMARY ACTION-SPACE CONTRACT (STAGE 5)
---

- AR-0: Prefer one PRIMARY action space
- AR-1: Keep all non-conflicting supported answers or filter to supported subset
- AR-2: Do NOT force a single answer unless evidence distinguishes
- AR-3: Cross-action merge is an exception
- AR-4: Reason over per-candidate values after filter

---
PART 7 - TOOL DEFINITIONS
---

1. check_entities(entity_substring="TEXT")
2. explore_schema(pattern="DOMAIN")
3. plan(question="...", anchor=[...], related=[...], maybe_related=[...], constraint_relations=[...], constraint_entities=[...])
4. action(anchor=[...], path=[{"relation": "...", "direction": "out"|"in"}])
5. filter(constraint_relations=[...], constraint_entities=[...], scope="selected")

---
END OF SYSTEM PROMPT
```

> 注意：完整原始prompt见项目中的 `sys_prompt.py`，以上为精简版。建议直接复制 `sys_prompt.py` 中 `SYSTEM_PROMPT_V11_FULL` 的内容。

## 4. User Prompt 模板

User prompt 的结构固定为以下模板：

```
Available Domains in Subgraph:
<DOMAIN_LIST>

[Retrieval Context]

Suggested Relations (Use for Planning ONLY):
<SUGGESTED_RELATIONS>
- relation1
- relation2
...

WARNING: These entities and relations are hints for EXPLORATION. Do NOT use them as final answers directly. You MUST verify them via check_entities and match_pattern.

Question:
<QUESTION_TEXT>


[PHASE 1] DISCOVERY MODE
Goal: Anchor question entities and explore relation structures in relevant domains.

Allowed Tools: check_entities, explore_schema
Forbidden Tools: plan, match_pattern, Answer

------------------------------
EXECUTION LOGIC:

1. Entity Identification
   - Extract proper nouns from the question
   - Call check_entities to verify potentially relevant nouns
   - Prioritize verifying "core entities" provided by the environment

2. Domain Analysis
   - Analyze which topic domains the question involves
   - Call explore_schema to expand relation structures
   - Only explore domains from the whitelist above

3. Answer Constraint Attribute Domain Analysis
   - From explored schema, identify:
     * Which relations can serve as attribute selectors for leaf nodes (e.g., time, type)
     * Which domains contain those constraint relations

------------------------------
OUTPUT FORMAT:
<reasoning>
1. Entity: "entity_name" -> Needs verification
2. Intent: "question intent" -> Related domains
3. Domains: List domains to explore
4. Constraint Attributes: Identify relations usable for attribute selection
</reasoning>
<act>
<query>check_entities(entity_substring="...")</query>
<query>explore_schema(pattern="...")</query>
</act>
```

### 4.1 各部分填充规则

| 部分 | 填充方式 |
|------|----------|
| `<DOMAIN_LIST>` | 从三元组中提取所有关系的domain部分，去重后逗号分隔 |
| `<SUGGESTED_RELATIONS>` | 从CWQ `user_query`的三元组中提取所有不重复的relation |
| `<QUESTION_TEXT>` | CWQ的 `question` 字段 |

### 4.2 Domain提取规则

从三元组 `(Entity, domain.type.relation, Entity)` 中：
- 取 `domain` 部分（第一个`.`之前）
- 去重后按字母排序
- 用 `, ` 连接

示例：三元组含 `sports.sports_team.team_mascot`, `sports.sports_team.championships`
→ domain = `sports`

## 5. CWQ 原始数据格式

CWQ数据路径：`/zhaoshu/SubgraphRAG-main/reason/gpt_labeled/gpt_labeled_cwq_raw.jsonl`

每行格式：
```json
{
  "id": "WebQTest-832_c334509bb5e02cacae1ba2e80c176499",
  "question": "Lou Seal is the mascot for the team that last won the World Series when?",
  "prediction": "evidence: (San Francisco Giants,sports.sports_team.team_mascot,Lou Seal)\nevidence: (Lou Seal,sports.mascot.team,San Francisco Giants)\n...",
  "ground_truth": ["2014 World Series"],
  "a_entity": ["2014 World Series"],
  "sys_query": "<system prompt for triplet selection>",
  "user_query": "Triplets:\n(San Francisco Giants,sports.sports_team.team_mascot,Lou Seal)\n..."
}
```

### 5.1 字段映射

| CWQ字段 | 目标字段 | 转换逻辑 |
|---------|---------|----------|
| `id` | `id` | 加 `CWQ-` 前缀，或直接用原id |
| `question` | user prompt中的 `Question:` 部分 | 直接使用 |
| `ground_truth` / `a_entity` | `ground_truth.global_truth_answers` | 直接使用 |
| `user_query`中的三元组 | user prompt中的 `Suggested Relations` | 提取所有不重复的relation |
| `user_query`中的三元组 | `ground_truth.oracle_relations` | 提取所有不重复的relation |
| `prediction`中的evidence | `ground_truth.core_relations` | 提取evidence三元组中的relation |
| `user_query`中的三元组 | `ground_truth.core_entities` | 提取所有entity名 |
| 三元组的domain | user prompt中的 `Available Domains` | 提取domain段去重 |

### 5.2 三元组解析规则

CWQ的三元组格式为 `(Entity1,relation,Entity2)`，每行一个。

提取relation：
```python
import re
def extract_relations_from_triplets(text):
    relations = set()
    for match in re.finditer(r'\(([^,]+),([^,]+),([^,]+)\)', text):
        relation = match.group(2).strip()
        if '.' in relation:  # fully qualified
            relations.add(relation)
    return sorted(relations)
```

提取entities：
```python
def extract_entities_from_triplets(text):
    entities = set()
    for match in re.finditer(r'\(([^,]+),([^,]+),([^,]+)\)', text):
        entities.add(match.group(1).strip())
        entities.add(match.group(3).strip())
    return sorted(entities)
```

提取domains：
```python
def extract_domains_from_relations(relations):
    domains = set()
    for rel in relations:
        if '.' in rel:
            domains.add(rel.split('.')[0])
    return sorted(domains)
```

## 6. 完整转换示例

### 输入（CWQ原始数据）
```json
{
  "id": "WebQTest-832_c334509bb5e02cacae1ba2e80c176499",
  "question": "Lou Seal is the mascot for the team that last won the World Series when?",
  "ground_truth": ["2014 World Series"],
  "a_entity": ["2014 World Series"],
  "prediction": "evidence: (San Francisco Giants,sports.sports_team.team_mascot,Lou Seal)\nevidence: (San Francisco Giants,sports.sports_team.championships,2014 World Series)",
  "user_query": "Triplets:\n(San Francisco Giants,sports.sports_team.team_mascot,Lou Seal)\n(San Francisco Giants,sports.sports_team.championships,2014 World Series)\n..."
}
```

### 输出（目标KGQA格式）
```json
{
  "id": "WebQTest-832_c334509bb5e02cacae1ba2e80c176499",
  "messages": [
    {
      "role": "system",
      "content": "<完整SYSTEM_PROMPT见第3节>"
    },
    {
      "role": "user",
      "content": "Available Domains in Subgraph:\nsports, time, location\n\n[Retrieval Context]\n\nSuggested Relations (Use for Planning ONLY):\n- sports.sports_team.team_mascot\n- sports.sports_team.championships\n- sports.sports_championship_event.champion\n- time.event.participant\n\nWARNING: These entities and relations are hints for EXPLORATION...\n\nQuestion:\nLou Seal is the mascot for the team that last won the World Series when?\n\n\n[PHASE 1] DISCOVERY MODE\n...(固定模板文本)"
    }
  ],
  "ground_truth": {
    "global_truth_answers": ["2014 World Series"],
    "oracle_relations": ["sports.sports_team.team_mascot", "sports.sports_team.championships", "sports.sports_championship_event.champion", "time.event.participant"],
    "core_entities": ["San Francisco Giants", "Lou Seal", "2014 World Series"],
    "core_relations": ["sports.sports_team.team_mascot", "sports.sports_team.championships"]
  }
}
```

## 7. 关系去重与core_relations筛选

- `oracle_relations`: 所有三元组中出现的不重复关系（全量）
- `core_relations`: 仅从 `prediction` 的 evidence 三元组中提取的关系（答案路径上的子集）
- 如果 `prediction` 没有可用的evidence，则 `core_relations` = `oracle_relations`

## 8. 注意事项

1. **CWQ是多跳复杂问题**：一个CWQ问题可能涉及多个子图路径，`core_relations`应包含所有答案路径上的关系
2. **三元组方向**：CWQ的 `(A, rel, B)` 格式是固定的，不需要处理方向
3. **实体名可能含空格和特殊字符**：保持原始文本不做额外处理
4. **全量数据**：3531条CWQ数据，输出文件预计较大
5. **embedding生成**：生成的jsonl中 `core_entities` 和 `core_relations` 会用于embedding检索，确保格式正确（fully qualified relation如 `sports.sports_team.championships`）
