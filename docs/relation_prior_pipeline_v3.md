# Relation-Prior Constrained Subgraph Pipeline - Change Summary

## Modified File

**`scripts/test_chain_decompose.py`** (唯一修改的文件)

---

## Pipeline Overview

整个 pipeline 分为 6 个阶段：

```
Step 1: Question Decomposition (LLM)
    ↓
Step 2: Entity Resolution (GTE top-k → LLM selection with context)
    ↓
Step 3: Relation Resolution (GTE recall → single LLM prune call)
    ↓
Step 4: Graph Traversal (relation-prior bidirectional BFS)
    ↓
Step 5: Path Compression (causal tier scoring + pattern dedup)
    ↓
Step 6: LLM Reasoning (path selection → triple reasoning → answer)
```

---

## Stage Details

### Step 1: Question Decomposition (L1328)

- **函数**: `run_case()` → `parse_decomposition()`
- **输入**: CWQ 问题文本
- **输出**: anchor entity query + 多步 reasoning chain，每步包含 `relation_query`
- **LLM 调用**: 1次，XML 格式输出 `<anchor>`, `<steps>`

### Step 2: Entity Resolution — **本次新增** (L1346-1390)

- **函数**: `llm_resolve_entity()` (L118)
- **辅助函数**: `get_entity_contexts()` (L85)
- **流程**:
  1. GTE 检索 top-k 候选实体（k=5 for anchor, k=3 for endpoint）
  2. 对每个候选实体，从子图三元组中提取周围关系上下文（最多 2 个出边 + 1 个入边）
  3. LLM 根据上下文从 top-k 中选择正确实体（XML 格式 `<analysis>` + `<selected>`）
- **上下文格式示例**:
  ```
  Rise [mascot team→Baltimore Ravens; topic notable types→Mascot]
  Mascot [type expected by→Team Mascot; type hints included types→Topic]
  ```
  → LLM 能根据上下文区分实体 vs schema 类型

- **关键设计**: 每次都调用 LLM 选择（无先验实体类别信息），上下文仅在此阶段展示

### Step 3: Relation Resolution (L1393-1450)

- **函数**: `llm_prune_all_relations()` (L220)
- **流程**:
  1. **Phase A**: 对每步，GTE 用 relation_query 和 step question 分别检索 top-10，合并去重
  2. **Phase B**: 单次 LLM 调用，将所有步的候选关系一起送入，模型基于完整 chain 上下文选择
- **LLM 输出格式**: XML `<analysis>` + `<selected>` (YAML: `step_N: [1, 3]`)
- **重试**: 3次，失败 fallback 为 GTE top-3
- **提示语设计**: 分析部分要求 "One sentence per step"

### Step 4: Graph Traversal (L965)

- **函数**: `relation_prior_expand()` (L965)
- **策略**: Relation-prior guided bidirectional BFS
- **参数**: `max_hops=4`, `beam_width=80`, `per_branch_width=5`
- **核心逻辑**:
  1. **Phase 1**: 扫描子图三元组，找到所有通过 R_n（最后一步关系）连接的实体作为 backward targets
  2. **Phase 2**: Forward expansion from anchor，每一步仅沿着该步 LLM 选出的关系方向扩展
  3. **Phase 2**: Backward expansion from targets，沿 R_n 反向扩展
  4. 双向在中间节点 meeting 时合并路径
- **CVT passthrough**: 遇到 `m.xxx/g.xxx` 格式的 CVT 节点时透传，不计入 hop 限制
- **Bridge-through 修复**: 当中间节点同时是 backward target 时，仍创建 meeting 路径但不加入 frontier

### Step 5: Path Compression (L1175)

- **函数**: `compress_paths()`
- **排序函数**: `score_causal_tier()` (L199)
- **Causal tier scoring**: `(hit_count, max_layer_hit, -bridge_length)` 元组，字典序排序
  - `hit_count`: 路径覆盖的 decomposition 步数
  - `max_layer_hit`: 最深覆盖层
  - `-bridge_length`: 路径长度（短路径优先）
- **Pattern 去重**: 相同关系链的路径合并为 pattern，按 `(best_tier, support)` 排序

### Step 6: LLM Reasoning (L1500-1630)

分两个 LLM 阶段：

**6a. Path Selection** (L1530)
- 输入: top-15 logical paths（带候选实体列表）
- 输出: XML `<analysis>` + `<selected>` (索引列表)
- 支持多轮 rollback（模型可以请求更多路径，最多 3 轮）
- max_tokens=1500（防止分析截断丢失 `<selected>` 标签）

**6b. Final Reasoning** (L1590)
- 输入: 选中路径展开为 triples
- 输出: `ANSWER: <entity name>`
- 提示语要求 1-2 句简短推理，直接给出答案
- max_tokens=800

---

## Key Functions Index

| Function | Line | Type | Description |
|----------|------|------|-------------|
| `get_entity_contexts()` | L85 | new | 从子图提取实体周围关系上下文（max 3 relations） |
| `llm_resolve_entity()` | L118 | new | LLM 从 GTE top-k 候选中选择正确实体 |
| `candidate_hit()` | L161 | modified | GT 匹配检查（支持 substring match） |
| `call_llm()` | L171 | unchanged | LLM API 调用封装 |
| `gte_retrieve()` | L192 | unchanged | GTE embedding 检索封装 |
| `score_causal_tier()` | L199 | new | Causal tier 路径评分 |
| `extract_xml_tag()` | L214 | new | XML 标签解析工具 |
| `llm_prune_all_relations()` | L220 | new | 单次 LLM 调用剪枝所有步的关系 |
| `parse_decomposition()` | L327 | unchanged | 解析 LLM 分解输出 |
| `relation_prior_expand()` | L965 | new | Relation-prior 双向 BFS 游走 |
| `compress_paths()` | L1175 | modified | 路径压缩（使用 causal tier 排序） |
| `expand_to_triples()` | L1276 | unchanged | 路径展开为三元组（含 CVT bridge） |
| `run_case()` | L1328 | modified | 主流程编排 |

---

## Prompt Design

所有提示语使用英文，XML 格式输出：

1. **Entity Resolution** (L149): `<analysis>` + `<selected>` — 基于搜索词和关系上下文选择实体
2. **Relation Pruning** (L273): `<analysis>` + `<selected>` (YAML) — 每步一句分析，选择 2-3 个关系
3. **Path Selection** (L1548): `<analysis>` + `<selected>` + `<need_more>` — 答案类型 + 路径匹配
4. **Final Reasoning** (L1595): 简短推理 + `ANSWER: <entity>` — 1-2 句结论

---

## Results Progression

| Version | GT Recall | LLM Correct | Key Change |
|---------|-----------|-------------|------------|
| miss25_xml_v1 | 15/25 (60%) | 9/25 (36%) | XML format + single-call prune |
| miss25_entity_ctx | 15/25 (60%) | 10/25 (40%) | +entity context in reasoning |
| miss25_v2 | 16/25 (64%) | 11/25 (44%) | +simplified prompts, hybrid entity resolution |
| miss25_v3 | pending | pending | +always LLM entity resolution, context only in GTE step |

---

## External Dependencies

- **LLM API**: `localhost:8000/v1/chat/completions` (Qwen2.5-9B)
- **GTE API**: `localhost:8003/retrieve` (GTE embedding retrieval)
- **Data**: `/zhaoshu/SubgraphRAG-main/retrieve/data_files/cwq/processed/test.pkl` (CWQ test set with pre-extracted subgraphs)
- **Pilot**: `reports/stage_pipeline_test/find_check_plan_pilot_10cases/results.json` (default) or `miss25_pilot.json` (25 miss cases)

---

## Data Flow

```
CWQ question + pre-extracted subgraph (pkl)
  │
  ├── entity_list (text + non-text)
  ├── relation_list
  ├── h_id_list, r_id_list, t_id_list (三元组索引)
  │
  ↓ Decomposition
  anchor_query + [{step_N, question, relation_query, endpoint_query}]
  │
  ↓ Entity Resolution (GTE + LLM + context)
  anchor_idx + {step_N: endpoint_idx}
  │
  ↓ Relation Resolution (GTE + LLM)
  {step_N: [rel_idx, ...]}
  │
  ↓ Graph Traversal (bidirectional BFS)
  paths: [{nodes, relations, covered_steps, depth}]
  │
  ↓ Path Compression (causal tier)
  logical_paths: [{display, candidates, support, raw_paths}]
  │
  ↓ Path Selection (LLM)
  selected_paths → triples
  │
  ↓ Final Reasoning (LLM)
  ANSWER: entity_name
```
