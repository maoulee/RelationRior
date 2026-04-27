# Prompt Optimization v7 — Proposal (v3)

两个优化方向：(1) Filter Gate 提示语优化 (2) Hint 冗余精简

---

## 优化 1：Filter Gate 提示语优化

### 问题
当前 `_maybe_enforce_filter_gate()` 是拦截机制 — 后端检测到候选间差异关系后，
必须提醒模型调用 filter()，否则不予通过方向。

问题不在于"是否可选"（调用是强制的），而在于：
1. 当前提示语指定了具体关系列表，限制了模型自主选择
2. filter 调用失败后进入 repair 循环，浪费 turn
3. 提示语末尾的"分析"部分混淆了 filter 职责和推理职责

### 当前提示语 (runtime.py:784-793)

```
[FILTER GATE]
Multiple candidates remain in the selected action space: Colorado, United States of America, Weld County
The backend already exposed Suggested Filter Relations for these candidates: location.location.containedby
Before final selection, you MUST execute a graph-side filter step.
Next step:
- call `filter(constraint_relations=["location.location.containedby"], scope="selected")`
After filter(), analyze the displayed per-candidate values carefully.
A candidate 'passing' filter only means the relation/value is present; it does NOT automatically mean the candidate belongs in the final answer.
Keep only the candidate subset whose displayed filter values actually match the question target or constraint.
```

问题点：
- `filter(constraint_relations=["location.location.containedby"])` — 硬编码关系，模型无自主选择权
- "analyze the displayed per-candidate values" — 分析是推理阶段的事，不属于 filter 步骤
- "Keep only the candidate subset..." — 过滤逻辑不应在 filter 门控中出现

### 拟议修改

**关键设计决策：**
- 后端给出 filter 建议 → 门控拦截 → 模型必须调用 filter（强制）
- 但模型自己决定用什么关系参数来消歧（自主选择）
- filter 只负责获取属性值，不做分析判断
- 分析判断在推理阶段完成（三段式推理 STEP 3）

**修改后的 FILTER GATE 提示语：**
```
[FILTER GATE]
Multiple candidates remain: Colorado, United States of America, Weld County
Suggested relations where candidates differ: location.location.containedby

These candidates cannot proceed without disambiguation.
You MUST call filter() to retrieve per-candidate attribute values.

You decide which relation(s) to use as filter parameters.
Choose the relation(s) most relevant to the question's intent.
```

对比变化：
- 去掉了硬编码的 `constraint_relations=[...]`，改为让模型自己选择
- 去掉了 filter 后分析指令（分析在推理阶段）
- 保留了强制调用 filter 的要求（拦截不变）

### v5_filter_then_answer 合约修改

```
FINAL ORDERING CONTRACT:
1. Choose the primary action space
2. Disambiguation (when filter gate triggers):
   - You MUST execute filter() with your chosen relation(s)
   - You decide which relations are most relevant for disambiguation
   - filter() retrieves attribute values; analysis happens in reasoning (step 3)
3. Reference historical case experience (advisory, not mandatory) — combine with current context
4. Output the answer
```

### 推理模板 (v5 三段式)

```
<reasoning>
  [STEP 1: PRIMARY ACTION SPACE]
  - Which action space was selected and why.

  [STEP 2: REFERENCE EXPERIENCE]
  - Historical case experience for reference (advisory, not mandatory):
    {skill answer strategy content}
  - Combine this reference with the current question's specifics to form your analysis.

  [STEP 3: CANDIDATE EVALUATION]
  - Analyze filter() results (if executed): per-candidate attribute values.
  - Based on current information and reference experience, decide: keep all or select subset.
  - Use reference experience and current information to form your analysis.

  [SPELLING VERIFICATION]
  - Final string: ...
</reasoning>
```

### 中文对照

**修改后的 Filter Gate:**
```
[过滤门控]
多个候选存在：Colorado, United States of America, Weld County
候选间有差异的关系：location.location.containedby

这些候选必须经过消歧才能继续。
你必须调用 filter() 来获取每个候选的属性值。

你自己决定用什么关系作为过滤参数。
选择与问题意图最相关的关系。
```

**修改后的 v5 合约:**
```
最终排序合约：
1. 选择主要动作空间
2. 消歧（当 filter 门控触发时）：
   - 你必须用你选择的关系执行 filter()
   - 你决定哪些关系最相关
   - filter() 获取属性值；分析在推理步骤3中完成
3. 参考历史 case 的作答经验（仅供参考，非强制）— 结合当前题目信息自行分析
4. 输出答案
```

---

## 优化 2：Hint 冗余精简

### 修改原则
1. 保留细粒度的问题文本（有助于区分相似方向）
2. 去掉 `Reference question:` 重复行（Case 标题已包含问题）
3. 统一英文标签
4. 作答模式保留 skill 原始策略，不做粗粒度简化
5. 动作阶段（action 选择时）只注入一次
6. Plan 阶段不注入 hint
7. **所有 hint 内容为参考经验，非强制约束** — 模型自行决定参考哪些
8. **方向标题是聚合展示结构** — aggregator 按方向聚类 case，仅作展示组织用
9. **不要求模型"留在方向内"或"不放弃方向"** — skill 只是展示，模型自行分析作答

### 当前 hint 结构

```
## Direction: Merged
#### Case: "where is reggie bush from"
Reference question: where is reggie bush from
核心关系: `people.person.place_of_birth`
作答模式: 全量采纳
答题策略: 利用 xxx 属性进行过滤
过滤属性: xxx, yyy
```

### 拟议修改

**动作阶段 (Action 选择时) — 只注入一次，精简版：**

```
## Direction: Finding someone's place of birth
#### Case: "where is reggie bush from"
Key relations: `people.person.place_of_birth`

#### Case: "where is alex rodriguez from"
Key relations: `people.person.place_of_birth`
```

变化：
- 保留方向标题 `## Direction: ...`
- 去掉 `Reference question:` 重复行（Case 标题已包含问题文本）
- `核心关系:` → `Key relations:` (统一英文)
- 只保留关系路径（作答模式/策略在最终阶段才注入）
- 同方向的多个 case 聚在一起

**最终阶段 (Stage 4-5) — 完整版，保留 skill 原始策略（仅供参考）：**

```
## Direction: Finding someone's place of birth
#### Case: "where is reggie bush from"
Key relations: `people.person.place_of_birth`
Answer mode: {skill card 原始 action_space_mode}
Selection rule: {skill card 原始 selection_rule}
Filter attributes: {skill card 原始 filter_attrs}

#### Case: "where is alex rodriguez from"
Key relations: `people.person.place_of_birth`
Answer mode: {skill card 原始 action_space_mode}
Selection rule: {skill card 原始 selection_rule}
Filter attributes: {skill card 原始 filter_attrs}
```

变化：
- 保留方向标题（`## Direction: ...`），多 case 聚在同一方向下
- 去掉 `Reference question:` 重复行
- `核心关系:` → `Key relations:`
- `作答模式:` → `Answer mode:` — 保留 skill card 原始值，不做粗粒度简化
- `答题策略:` → `Selection rule:` — 保留 skill card 原始 selection_rule
- `过滤属性:` → `Filter attributes:`
- **所有内容均为参考经验，不是强制约束**

### Wrapper 标签更新

```
# 旧
[Historical Case Reference — Similar past cases for strategy analysis only. The current question has NOT changed. Do NOT answer the reference cases below.]

# 新
[Historical Case Reference — These are solved cases grouped by answer direction. Use their experience as REFERENCE ONLY, not as direct answers.]
```

### 额外优化：去重注入

- stage:2 和 stage:3 合并为一次注入（使用 `stage:2-3` key）
- Plan 阶段不注入 hint

---

## 影响范围

| 文件 | 优化1 | 优化2 |
|------|-------|-------|
| runtime.py `_maybe_enforce_filter_gate()` | 修改门控提示语（去硬编码关系，保留强制调用） | - |
| variants.py `v5_filter_then_answer` | 修改合约和推理模板 | - |
| aggregator.py `_synthesize_direction()` | - | 修改标签为英文，保留方向标题 |
| aggregator.py `extract_action_stage_guidance()` | - | 精简为只含 Key relations |
| aggregator.py `_action_space_mode_label()` | - | 保留 skill 原始值，不做粗粒度简化 |
| runtime.py `_maybe_stage_skill_hint()` | - | stage:2-3 合并一次注入 |
| runtime.py wrapper 标签 | - | 更新为纯参考标签（去掉方向约束） |

## 测试计划

修改完成后运行 100-case 测试，对比：
1. F1 和 Hit@1 vs 当前 agg_v6 (F1=0.799, Hit@1=0.820)
2. 关注之前回归的 case（WebQTest-80/165/116）是否修复
3. 关注之前改善的 case（WebQTest-109/163/86）是否保持
