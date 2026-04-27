# Subgraph KGQA 当前逻辑总览

这份文档用于给后续“学术智能体 / 研究代理”提供统一上下文，解释当前代码的系统结构、推理闭环、提示语、失败模式以及后续向“自进化 + skills”方向演化的路径。

本文的讨论前提是：

- 不把 RL 微调作为下一阶段主路线。
- 把现有代码看成一个“可拆解的代理系统”。
- 重点讨论如何把推理过程沉淀成 skills，并通过回放、反思和规则修订实现自进化。

## 1. 稳定入口与代码角色

当前项目建议把下面 3 个文件视为稳定入口：

- `plug.py`
  - 统一代理逻辑入口。
  - 对外暴露了 `plug_v11.py`、`plug_v12_feedback.py`、`v10_environment.py` 中的核心能力。
- `evaluate.py`
  - 离线测试 / 多轮评测入口。
  - 底层复用 `test_pipe6.py` 的评测流程。
- `graph_server.py`
  - 图后端入口。
  - 当前是一个包装入口，最终会调用 `src/subgraph_kgqa/backend/server.py`，再转发到真实后端源码。

当前版本的底层职责划分：

- `plug_v11.py`
  - 多轮调度器。
  - 输出解析器。
  - 历史压缩与轨迹状态管理。
  - 也保留了旧的 reward 实现，但这部分对下一阶段不是主线。
- `plug_v12_feedback.py`
  - 前端校验。
  - 后端结果分类。
  - 错误反馈 / 阶段提示生成。
- `v10_environment.py`
  - 更偏“显式环境逻辑”的版本化实现。
  - 包含状态对象、格式诊断、相位跳转拦截、命令去重等。
- `test_pipe6.py`
  - 离线评测框架。
  - 模拟一轮一轮对话，把提示词、LLM 调用、工具调用、状态更新和最终 F1 报告串起来。
- `/zhaoshu/SubgraphRAG-main/retrieve/graph_server.py`
  - 当前真实图后端实现。
  - 提供实体检索、schema 探索、路径规划、动作执行、约束过滤等接口。

## 2. 系统总目标

这个系统本质上不是“直接问答”，而是一个**分阶段的图搜索代理**：

1. 先定位问题中的实体和相关知识域。
2. 再选择可能的关系，构造检索计划。
3. 把计划转成图遍历动作。
4. 从动作执行结果中收集候选实体。
5. 最后从候选实体里压缩和筛选，输出最终答案。

也就是说，当前系统希望模型学习的不是“背答案”，而是：

- 如何探索图。
- 如何选择关系。
- 如何利用动作空间。
- 如何把候选集合压缩成最终答案。

## 3. 一条样本的完整推理闭环

下面是当前系统实际运行时的一条样本闭环。

### 3.1 输入阶段

输入由两部分构成：

- 主系统提示词 `SYSTEM_PROMPT_V11_FULL`
- 用户问题 + `PHASE1_HINT`

在离线评测里，初始对话大致是：

- system: `SYSTEM_PROMPT_V11_FULL`
- user: `question + PHASE1_HINT`

在当前在线编排逻辑里，由 `KGQAScheduler` 维持多轮消息和状态。

### 3.2 模型输出格式

模型被要求输出严格结构化格式：

- `<reasoning>...</reasoning>`
- `<act> ... <query>tool_call(...)</query> ... </act>`
- 或最终 `<answer>\boxed{...}</answer>`

当前范式强调：

- 每一轮必须先有 reasoning。
- 一轮只能有一个 `<act>`。
- tool call 必须在 `<query>` 里。
- 最终答案必须使用 boxed 格式。

### 3.3 输出解析

当前有两个 parser 体系：

- 在线编排主线：`plug_v11.py` 中的 `OutputParser`
- 离线评测：`test_pipe6.py` 中的 `OutputParser`

它们都做三件事：

- 抽取 reasoning
- 抽取工具调用
- 抽取 boxed 最终答案

同时评测版还会额外识别：

- `<candidate>` / `<candidate_entities>`
- `plan` / `plan_subquestion`

### 3.4 前端校验

在真正访问图后端之前，会先做前端校验。核心思想是：

- **先判断模型输出是否合法，再允许执行工具。**

`plug_v12_feedback.py` 中的 `FrontendValidator` 主要负责检查：

- `plan` / `plan_subquestion`
  - `question` 是否为空
  - `anchor` 是否存在且已经被验证
  - `related` / `maybe_related` 是否为空
  - relation 是否来自已探索 schema 或提示上下文
- `match_pattern`
  - 是否在有效 `action_hints` 空间内
- `explore_schema`
  - domain 是否在可用 domain 白名单内
- `filter`
  - 约束格式是否合法

如果校验失败：

- 工具调用会被全部拦截
- 环境返回结构化错误信息
- 模型必须修正后重试

这是当前系统“抗幻觉”的第一道硬门槛。

### 3.5 图工具执行

如果前端校验通过，系统会把 query 发给图后端。

当前对模型开放的核心工具链大致是：

- `check_entities` / `find_entities`
  - 找实体候选，并返回带上下文的候选列表。
- `explore_schema`
  - 在指定 domain 下找 relation。
- `plan` / `plan_subquestion`
  - 将 anchor + relations 转成候选图路径。
  - 返回的是**动作空间**，即 `action_hints`。
- `match_pattern` / `action`
  - 真正执行图遍历动作，获取叶子候选实体。
- `filter`
  - 用关系约束 / 实体约束对候选进行过滤。
- `get_neighbors`
  - 获取邻居关系，辅助调试和补充上下文。

其中最关键的设计是：

- `plan_subquestion` 不直接给答案。
- 它的任务是生成一批**可以执行的动作空间**。
- 模型后续必须从 `action_hints` 中复制或选择动作，再执行 `match_pattern`。

也就是说，当前架构显式拆成了：

- 规划器 planner
- 执行器 executor

### 3.6 后端结果分类

图后端返回后，不是简单地“有结果 / 没结果”，而是被 `BackendChecker` 细分为：

- `SUCCESS`
- `PARTIAL`
- `NO_PATH`
- `NO_MATCHES`
- `ENTITY_ERROR`
- `ERROR`

这些状态的意义是不同的：

- `NO_PATH`
  - 通常说明 relation 或计划路径有问题，需要重规划。
- `NO_MATCHES`
  - 路径逻辑可能成立，但当前图中没有实例。
- `ENTITY_ERROR`
  - anchor 本身就可能错了。

这使得模型不是简单“失败重试”，而是带着失败类型重试。

### 3.7 状态更新

无论在在线编排 scheduler 里，还是在离线测试里，都维护了一份运行状态。核心字段包括：

- `verified_entities`
- `schema_relations`
- `valid_actions`
- `has_action_hint`
- `any_match_executed`
- `retrieved_candidates`
- `all_leaf_entities`
- `candidates_collected`
- `has_final_answer`

状态的作用有三类：

1. 给前端校验器提供“什么是合法动作”的参考。
2. 给阶段提示器提供“下一步应该做什么”的依据。
3. 给历史评估层和未来 skill 评测器提供“这一轮是否真的完成了规划 / 执行 / 压缩”的信号。

### 3.8 反馈回注入模型

执行完后，系统会把下面这些内容拼成下一轮 user feedback：

- 工具结果
- 检索到的实体 / relation 提示
- 后端错误解释
- 阶段提示

一个关键点是：

- `plug_v12_feedback.py` 把错误反馈和阶段提示解耦了。
- 当前策略是：
  - 前端错误优先，直接阻断执行。
  - 后端错误允许执行完再要求重规划。
  - 只有在没错误时才给阶段提示。

这意味着系统在尽量保证：

- 错误反馈是刚性的
- 阶段提示是引导性的

### 3.9 最终答案

模型最终在第 5 阶段输出：

```text
<answer>\boxed{Entity}</answer>
```

或：

```text
<answer>\boxed{A} \boxed{B}</answer>
```

答案评估使用 `normalize + calculate_f1`。

## 4. 当前“推理逻辑”本质上是什么

从代码来看，当前系统的推理不是纯自由推理，而是**被环境塑形的程序化推理**。

可以把它概括成下面这个流程：

1. **发现阶段**
   - 先确定问题里的实体。
   - 再确定图里允许探索的 domain 与 relation。
2. **规划阶段**
   - 根据 answer type、anchor、相关 relation 组织检索计划。
   - 计划不是最终路径，而是 relation-level 的候选。
3. **动作阶段**
   - 后端根据 `(anchor, relation)` 生成可执行 path。
   - 模型再从动作空间里选择 path。
4. **候选收集阶段**
   - 把 `match_pattern` 结果中的叶子实体收集成候选池。
   - 必要时用 `filter` 做属性 / 实体约束筛选。
5. **压缩推理阶段**
   - 从候选池 `U` 中压缩出最终答案集合 `Y`。
   - 推理质量由 `ReasoningCompressionReward` 评估。

所以，这个系统的“推理”更准确地说是：

- **图上的分阶段检索推理**
- **动作空间约束下的结构化推理**
- **候选压缩式推理**

而不是单轮 CoT 问答。

## 5. 图后端的实际作用

图后端是当前系统里最关键的“外部世界”。模型本身不直接看图，而是通过后端接口拿到图结构信息。

### 5.1 后端内部对象

图后端核心对象有两个：

- `GraphPatternMatcher`
  - 基于图对象做路径查找、邻居提取、relation 搜索、模式执行。
- `DataManager`
  - 负责按 sample 加载图数据，并缓存 matcher。

也就是说，当前不是一个全局统一大图查询系统，而是：

- 每条样本带一个局部子图
- 后端针对这个样本子图提供接口

### 5.2 各工具的语义

#### `find_entities`

作用：

- 根据实体字符串做候选匹配。
- 返回带上下文的 entity candidates。

特点：

- 不只返回 top1。
- 即使 top1 是 100 分，也仍然展示多个候选，让模型自己做 disambiguation。

#### `explore_schema`

作用：

- 根据 domain 前缀输出该 domain 下的 relation 层级。

特点：

- 返回 full-qualified relation。
- 同时被前端拿来做 relation hallucination 防御。

#### `plan` / `plan_subquestion`

作用：

- 把 `anchor + related + maybe_related (+ constraints)` 展开成路径搜索任务。
- 后端并行调用 `find_logical_path_with_relation`。
- 聚合后返回 `action_hints`。

本质：

- 它是“动作空间生成器”，不是最终答案器。

#### `find_logical_path_with_relation`

作用：

- 给定 anchor 和 relation，在图中找包含该 relation 的逻辑路径。

输出：

- RDF 风格路径描述
- 对应可执行 `action_code`
- 自然语言 example
- 结构化 path signature

#### `match_pattern` / `action`

作用：

- 真正执行 path。
- 从 anchor 出发，根据 relation + direction 序列找终点实体。

输出：

- `found_end_entities`
- `structured_data`
- 可能包含 CVT 展开结果

这是整个系统里真正产生候选答案的核心动作。

#### `filter`

作用：

- 对 `match_pattern` 得到的候选集合做二次过滤。
- 支持 `constraint_relations` 和 `constraint_entities`。

特点：

- 采用“global constraint hiding”：
  - 如果一个约束对所有候选都成立，它不具区分性，就会被弱化展示。

这个设计的目标是把模型注意力聚焦到真正有判别力的约束上。

## 6. 编排逻辑：Scheduler 当前承担什么

虽然下一步我们不再把 RL 微调当主线，但当前代码里的 `KGQAScheduler` 仍然很重要，因为它刻画了系统现在默认的“代理编排逻辑”。

可以把它理解为：

- 一个多轮状态机
- 一个上下文压缩器
- 一个阶段守门器
- 一个工具执行编排器

它做了几件关键事情：

### 6.1 维护多轮状态

它在 `infer_request.objects['kg_state']` 中维护当前样本状态，包括：

- 当前阶段
- 已验证实体
- 已见 schema relation
- 是否拿到 action hints
- 是否执行过 match_pattern
- 当前候选集合
- 工具结果的 structured data
- 所有规划过的 generator

### 6.2 强制阶段顺序

系统虽然是语言模型生成，但并不是完全自由行动。

代码里显式规定了阶段逻辑：

- Stage 1: Discovery
- Stage 2: Planning
- Stage 3: Execution
- Stage 4: Candidate Collection
- Stage 5: Reasoning

如果模型跳阶段，比如：

- 没做 discovery 就 plan
- 没拿到 action hint 就乱执行 action

环境会拦截或处罚。

### 6.3 压缩上下文

为了避免多轮工具输出过长，scheduler 会：

- 去掉旧 assistant 中的 reasoning
- 对历史 tool results 做压缩
- 尤其针对 `match_pattern` 的 structured data 做按候选驱动的压缩

这个设计非常关键，因为当前系统不是简单 1-2 轮问答，而是多轮工具交互，长度膨胀很快。

### 6.4 历史 reward 层仅作为“旧设计背景”

当前代码里仍然保留了一整套 reward 设计，但对于下一阶段讨论，它更适合作为“旧系统如何理解代理行为”的背景材料，而不是未来方案主轴。

#### `F1Reward`

作用：

- 最终答案集合和 GT 的 F1。

#### `PlanQualityReward`

作用：

- 衡量 planner 是否提出了有效 `(anchor, relation)` 生成器。

核心思想：

- 不看“说得多好”，看 plan 中的 generator 是否真的导向 GT。

#### `ActionQualityReward`

作用：

- 衡量 action 执行效率。

核心思想：

- 不是执行越多越好。
- 而是执行的动作中，有多少是真正有效的。

#### `ReasoningCompressionReward`

作用：

- 衡量模型能否把候选池 `U` 压缩成正确答案 `Y`。

核心思想：

- 候选池里包含很多东西并不难。
- 真正难的是在不丢 GT 的情况下做压缩。

这其实非常接近“后验推理器”的评价。

#### `ComplianceReward`

作用：

- 约束格式、标签、工具使用规范、阶段结构。

核心思想：

- 没有正确结构，就不给高分。

#### `BehaviorReward`

作用：

- 识别懒惰策略、幻觉策略、重复调用、跳阶段、假装信息不足等。

核心思想：

- 不仅要对，还要“以正确方式对”。

#### `CosineLengthReward`

作用：

- 用 reasoning 长度曲线刻画“思考过短 / 过长”的模式。

核心思想：

- 过短：可能没认真思考
- 过长：可能坍塌或啰嗦

如果后续完全转向自进化，这些 reward 更适合被改造为：

- 诊断指标
- 轨迹打标签器
- skill 质量评估器
- 自动失败归因器

## 7. 离线测试逻辑：`test_pipe6.py` 在做什么

`test_pipe6.py` 的目标不是微调，而是：

- 用当前 prompt + 环境逻辑模拟多轮代理
- 对一批样本做离线评测
- 生成详细 trajectory 报告

其主流程是：

1. 读取数据集。
2. 为每个 case 构造初始对话。
3. 调用 LLM。
4. 解析输出。
5. 做前端校验。
6. 调后端工具。
7. 更新状态并反馈。
8. 多轮迭代直到答案或超出 turn 限制。
9. 对每个 case 做 best-of-n。
10. 用最终答案算 F1，输出报告。

它相当于：

- 一个代理行为模拟器
- 一个用来研究 prompt / 环境策略是否工作正常的沙盒
- 一个未来可直接用于“自进化回放评估”的验证器

## 8. 当前系统最重要的设计哲学

从 prompt 和环境逻辑综合起来看，当前系统有几个非常鲜明的哲学。

### 8.1 “不要因为不完美就停止”

这是整个系统反复强调的主旋律。

系统明确告诉模型：

- 知识图谱是人工构建的，不会完美。
- relation 不会总有一条特别理想的。
- 如果有多个相关 relation，就都保留。
- 不要因为“没有完美 relation”就拒绝前进。

这实际上是为了对抗一种典型失败模式：

- 模型过早保守
- 模型过早说 “insufficient”
- 模型因为不确定而不探索

### 8.2 “先探索，再规划，再执行”

系统不是鼓励模型直接猜路径，而是要求：

- 先确认实体
- 再确认 schema
- 再做 plan
- 再执行 action

这是一种非常强的“程序化归纳偏置”。

### 8.3 “Planner 和 Executor 分离”

`plan_subquestion` 生成 `action_hints`，`match_pattern` 再执行。

这意味着当前系统把错误拆成两种：

- 规划错误
- 执行错误

这对研究很重要，因为后续可以分别 skill 化 planner 和 executor。

### 8.4 “最终推理 = 候选压缩”

系统不是把 reasoning 当成纯解释文本，而是把 reasoning 的本质定义成：

- 从候选池筛选出最终答案

这使得当前系统天然适合研究：

- reranking
- candidate filtering
- verifier / selector
- posterior reasoning

## 9. 基于代码可见的瓶颈与“自进化 + Skills”方向

下面这一节分为两类：

- 代码事实：从现有实现可以直接看到的结构特征
- 研究推断：基于这些特征，可能的模型演化方向

### 9.1 代码事实

#### 事实 1：当前系统高度依赖 prompt 和环境约束

模型之所以按阶段走，很大程度上不是因为模型天然会这样做，而是因为：

- prompt 强约束
- validator 拦截
- 历史 reward / 诊断层
- phase hint 引导

这说明当前系统的“程序性”主要在外部环境，而不是内化到模型参数中。

#### 事实 2：planner 仍然是 relation-first，而不是 latent reasoning-first

当前 `plan` 的核心单位仍然是：

- anchor
- related / maybe_related
- constraint_relations / constraint_entities

即使 prompt 讲了很多语义原则，真正落到执行层的仍然是 relation selection。

#### 事实 3：action space 是后端枚举出来的

这说明：

- 模型不是自由搜索整张图
- 模型是在后端枚举的动作空间中做选择

这意味着当前系统更像：

- 受限动作空间中的代理编排

而不是纯开放式自由推理。

#### 事实 4：最终 reasoning 层目前仍然偏弱约束

从代码结构看：

- reasoning 的质量主要通过结果来推断
- 对 reasoning 过程本身的监督仍然较弱

#### 事实 5：编排逻辑和评测逻辑还存在“双实现”

虽然我们已经用 `plug.py` 和 `evaluate.py` 做了统一入口，但底层仍然存在：

- 在线编排版 parser / scheduler
- 离线评测版 parser / runner

这说明后续还有进一步抽象统一成 skill runtime 的空间。

### 9.2 研究推断：建议讨论的演化方向

这里开始不再以 RL 微调为主线，而以“模型自进化”和“推理 skill 化”为主线。

#### 方向 1：把当前整条链路拆成显式 Skills

从代码看，当前至少可以拆成下面几类 skill：

- `entity_grounding_skill`
  - 负责识别问题中的实体，并正确调用 `check_entities / find_entities`
- `schema_exploration_skill`
  - 负责选择 domain，并通过 `explore_schema` 建立 relation 候选池
- `planning_skill`
  - 负责把问题改写成 `anchor + related + maybe_related + constraints`
- `action_selection_skill`
  - 负责从 `action_hints` 中选动作，而不是乱造 path
- `candidate_filter_skill`
  - 负责在候选集上应用 `filter`
- `answer_synthesis_skill`
  - 负责从候选池压缩到最终 boxed 答案

这样做的最大好处是：

- 能把失败点映射到具体 skill
- 能按 skill 单独迭代 prompt / 规则 / 记忆
- 能做 skill 级回放和自修正

#### 方向 2：把“环境反馈”改造成“skill 反思信号”

当前反馈已经天然分成三类：

- 前端校验错误
- 后端执行错误
- 阶段提示

这三类信息很适合直接转成 skill 自进化信号：

- 前端错误
  - 可视为 skill 接口使用错误
- 后端错误
  - 可视为 skill 策略错误
- 阶段提示
  - 可视为 skill 间切换提示

后续可以讨论把当前 `ErrorFeedback / ResultFeedback / PhaseHintGenerator` 改写成：

- skill critique
- skill retry instruction
- skill switch policy

#### 方向 3：把 reward 思路改造成“离线诊断指标”，而不是训练目标

当前 reward 仍然很有价值，但更适合被重新解释为：

- `PlanQualityReward`
  - 规划 skill 的有效性指标
- `ActionQualityReward`
  - 动作选择 skill 的效率指标
- `ReasoningCompressionReward`
  - 候选压缩 skill 的质量指标
- `ComplianceReward`
  - 格式与协议遵循指标
- `BehaviorReward`
  - 失败模式指标

也就是说，未来不一定要“用它们更新参数”，而可以：

- 用它们给轨迹打分
- 用它们挖失败样本
- 用它们驱动 skill prompt 迭代
- 用它们筛选更好的自生成经验

#### 方向 4：建立“自进化回路”而不是“微调回路”

当前代码已经提供了一个很好的自进化基础：

- 有标准化 prompt
- 有结构化工具接口
- 有阶段状态
- 有可解释错误反馈
- 有离线回放评测器

因此后续完全可以讨论下面这种闭环：

1. 用当前代理跑问题。
2. 收集完整 trajectory。
3. 按 skill 切分失败点。
4. 用失败点生成新的 skill 规则 / skill prompt / skill memory。
5. 在离线评测器里回放验证。
6. 只保留真正提升成功率的 skill 变体。

这条路更接近：

- prompt evolution
- tool-use evolution
- skill library evolution
- reflective agent improvement

而不是 supervised finetuning 或 RLHF。

#### 方向 5：把 `action_hints` 视为“可学习的外部技能接口”

当前 `action_hints` 已经非常像一个外部 skill API：

- 输入是 question + anchor + relation candidates
- 输出是可执行 path

因此可以考虑把它看作：

- planner skill 的产物
- executor skill 的输入

未来可以重点讨论：

- 如何让模型更稳定地复制 / 选择 action hints
- 如何对 action hint 做排序
- 如何在 action hint 失败后进行受控回退

#### 方向 6：加强阶段 4，把候选过滤变成独立 skill

当前系统中最值得单独 skill 化的一层，其实是阶段 4。

因为这里连接着：

- 图动作输出
- 约束推理
- 最终答案压缩

它非常适合作为独立 skill：

- 输入：candidate pool + constraints + question intent
- 输出：refined candidates + rationale

后续可以讨论：

- 候选筛选 skill 是否应单独记忆高频模式
- 时间约束 / 类型约束 / 关系约束是否拆成子 skill
- `filter` 的结果是否需要更结构化

#### 方向 7：减少对 regex 解析和长文本格式的依赖

如果要做长期自进化，当前这类依赖会成为瓶颈：

- XML 风格标签
- regex parser
- 文本化工具返回

后续更适合的方向是：

- JSON schema action
- typed skill I/O
- structured critique
- structured candidate state

这样 skill 演化会比纯文本 prompt 演化更稳。

#### 方向 8：强化 graph-relative time 与 constraint reasoning skills

系统 prompt 已经把时间推理和约束推理强调得很多，但从代码看，真正结构化支持还有限。

后续可以把它们显式做成 skills：

- `temporal_reasoning_skill`
- `constraint_resolution_skill`
- `entity_type_check_skill`

这些 skill 很适合通过失败案例持续演化。

## 10. 给学术智能体的最短摘要

如果只用一句话概括当前系统：

> 这是一个“通过 prompt、环境反馈和图后端动作空间来分阶段执行图搜索，并可进一步拆成 skills 做自进化”的 KGQA 代理。

如果用三句话概括：

1. 它不是直接生成答案，而是先探索实体和 schema，再生成动作空间，再执行图动作，再做候选压缩。
2. 它的主要价值不只在于某个 prompt，而在于“prompt + validator + backend action space + feedback”共同形成的程序化推理闭环。
3. 后续最值得研究的方向，不是继续做 RL 微调，而是把这条闭环拆成可诊断、可演化、可复用的 skills。

## 11. 附录 A：当前主系统提示语

下面贴当前主系统提示语原文，便于后续直接分析 prompt 设计。

```text
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
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CORE PHILOSOPHY (APPLIES TO ALL STAGES)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The knowledge graph is HUMAN-CONSTRUCTED. There is NO perfect relation for every question.

✅ CORRECT MINDSET: Select the BEST AVAILABLE option based on current information.
❌ WRONG MINDSET: Refuse to proceed because options are not perfect.

At EVERY stage:
- If there is no ideal relation → Select the closest one
- If all options seem imperfect → Select the most likely one in current space
- NEVER refuse to proceed just because "nothing is perfect"

GRAPH-RELATIVE TIME:
- The graph has a TIME SNAPSHOT. "Latest" means latest IN THE GRAPH, not real-world now.
- Example: If graph is from 2016, "current president" = president in graph data.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
---

---
PART 0 — DESIGN LOGIC
---

DESIGN-0: Boolean Relevance > Ranking (ANTI-GREEDY)
- When selecting relations or actions, use BOOLEAN RELEVANCE (Is it semantically relevant? Yes/No).
- DO NOT use RANKING (Is A better than B?).
- If Relation A and Relation B are both semantically relevant, KEEP BOTH.
- NEVER prune a relevant relation just because another one looks "more direct".

DESIGN-1: Global Path Semantics (CRITICAL)
- Do NOT judge a path solely by a single relation keyword (e.g., "parents").
- You MUST evaluate the ENTIRE LOGICAL CHAIN (Start -> Rel -> End).
- Example: "Who are the parents of Actor X?"
  - Path A: Actor -> logical_parents (CORRECT)
  - Path B: Actor -> played_role -> character -> fictional_parents (WRONG - semantics mismatch).
- Reject ONLY if the *entire path meaning* contradicts the user intent.

DESIGN-2: Discovery-Driven Types (No Assumptions)
- Do NOT assume the answer type before exploring.
- Example: "Da Vinci's works" -> Could be Painting, Sculpture, Manuscript, Invention.
- Do NOT filter for only "Painting" unless the question explicitly says "Which painting...".
- Collect ALL types that match the semantic relation.

DESIGN-3: The "Hesitation" Rule (Default-to-Keep)
- If you are HESITANT about a relation or path (it looks indirect, generic, or "maybe" relevant), you MUST KEEP IT.
- Put hesitant relations in `maybe_related`.
- Put hesitant actions in `POTENTIAL` category.
- Only DISCARD if you have PROOF that it is irrelevant (Garbage Filter).

DESIGN-4: Fact Verification Hierarchy
1) Graph Evidence (Top Priority): Use edge properties (dates, roles) if they exist.
2) Model Knowledge (Fallback): If graph is silent, use your internal knowledge to verify constraints.
3) Graph-Relative Time: Use the time validity defined IN THE GRAPH. Do not use wall-clock "now".

---
PART 1 — FORMAT RULES (CRITICAL)
---

F0: Mandatory Thinking Block (EVERY TURN)
- You MUST output <reasoning>...</reasoning> BEFORE any <act> or <answer>.
- The <reasoning> block should contain your step-by-step thinking for this turn.
- Order: <reasoning> FIRST, then <act> OR <answer>.

F1: One <act> block per message
- You MUST put all tool queries in ONE <act> with multiple <query> tags.

F2: No bare <query>
- You MUST NOT output <query> without wrapping it inside <act>.

F3: List parameter format
- Use ["X"] not "X"; empty list is [] (never [""]).

F4: Final Answer Format (STAGE 5 ONLY) [CRITICAL]
- Syntax: <answer>\boxed{Entity}</answer>
- Multiple: <answer>\boxed{A} \boxed{B}</answer> (separate boxed per entity)
- NO text after </answer> tag.

F9: Telegraphic Thinking Style (CRITICAL)
- The <reasoning> block is for YOUR internal planning, not for explaining rules to the user.
- BE CONCISE: Use bullet points, fragments, and arrows (->). Avoid full paragraphs.
- NO META-TALK: Do NOT quote System Prompt rules (e.g., "According to Design-0...", "As per Hesitation Rule..."). Just APPLY them.
- NO REPETITION: Do not restate the user question or obvious facts multiple times.

---
PART 2 — GROUNDING RULES
---

G1: Entity Source Rule
- Every Answer entity MUST appear verbatim in tool outputs (Leaf/Target/Data Topology/Node Details values).

G2: Relation Source Rule
Relations may come from:
- explore_schema leaf relations (depth can be 3/4/5+)
- suggested relations (hints)
- [Action] hints (copy exactly)

FORMAT CRITICAL:
- Relations MUST be fully qualified: domain.type.relation
- Example: sports.sports_team.colors (NOT sports_team.colors)
- The first segment is the DOMAIN from explore_schema output.

G3: explore_schema domain whitelist
- explore_schema(pattern=...) MUST be a domain from "Available Domains in Subgraph".

---
PART 3 — DIRECTION & BACKTRACKING
---

D1: IN/OUT triple reconstruction
- OUT: (Start, rel, End)
- IN:  (End, rel, Start)

B1: Mandatory Backtracking (STAGE 3/4)
After match_pattern, BEFORE collecting candidates:
- If results are inverse-role (wrong direction) -> try opposite direction
- If Leaf Entities are wrong TYPE (asked person, got place) -> try alternative [Action]
- Do NOT proceed to Stage 4 with inverse-role results.

EXCEPTION (Data Tolerance):
- Do NOT backtrack just because path seems indirect/fictional.
- Focus on Leaf Entity TYPE, ignore intermediate path semantics.
- If [Action] provided by system, TRUST it even if path seems strange.

---
PART 4 — RELATION SELECTION (MANDATORY)
---

CRITICAL DISTINCTION:
- related / maybe_related = Retrieval Relations (Pathfinding).

-----------------------------------------------------------
A. RELATION SELECTION LOOP (The "Hesitation" Protocol)
-----------------------------------------------------------
RS-1: Three-Tier Classification (Anti-Laziness)
You MUST categorize every explored relation into one of three buckets:
  1) related (Core): High confidence, direct semantic match to the user intent.
  2) maybe_related (The Hesitation Buffer): 
     - Rule: "If I am not 100% sure it is irrelevant, I MUST keep it here."
     - Usage: Indirect paths, fuzzy matches, or broad relations.
  3) Discard (Garbage Filter):
     - Rule: ONLY for relations that are clearly wrong (wrong domain, inverse role).
     - Note: If all non-core relations are garbage, keep `maybe_related` empty. Do NOT blind-fill.

RS-2: No Premature Pruning (CRITICAL)
- Do NOT discard a relation just because another one looks "more direct".
- If Relation A and Relation B are both relevant, KEEP BOTH (in related or maybe_related).

RS-3: maybe_related Size (Soft Guideline)
- Keep maybe_related small (1-3 relations max) to avoid action explosion.
- But NEVER discard a relevant relation just to "keep it small".

---
PART 5 — GRAPH-RELATIVE TIME (CRITICAL)
---

[CORE PRINCIPLE]: The knowledge graph has a TIME SNAPSHOT. Your answers MUST be based on the graph's time frame, NOT real-world "now".

EXAMPLE (2016 Graph Snapshot):
- Question: "Where is the latest Olympics held?"
- Graph contains: 2016 Rio de Janeiro Olympics (latest in graph)
- Real world now: 2024 Paris Olympics
- CORRECT ANSWER: Rio de Janeiro (graph truth)
- WRONG ANSWER: Paris (wall-clock "now" error)

REASON: The graph snapshot is from ~2016. "Latest" means "latest in the graph", not "latest in real world".

TR-0: No wall-clock "now/today". Always use graph-relative time.
TR-1: If from/to or start/end exists:
- Prefer open interval (missing end/to) as current-in-KG
- Else prefer latest start/from (graph-latest)
TR-2: If no time metadata: do not time-filter by guess.
TR-3: Explicit "in YEAR": use interval coverage if available; else Top-K with limitation note.

---
PART 6 — CARDINALITY CONTRACT (STAGE 5)
---

- OP_ENUM: list all valid answers for requested slots after evidence-based filtering.
- OP_SELECT_EXTREMUM / OP_SELECT_CURRENT: output Top-1 (max Top-3 if indistinguishable).

---
PART 7 — TOOL DEFINITIONS (DETAILED)
---

### 1. check_entities
Purpose: Verify if an entity exists in the subgraph.
Parameters:
  - entity_substring (String, REQUIRED): The entity name or substring to search.
    Example: "Johnny Depp", "Barack Obama"

Syntax: check_entities(entity_substring="TEXT")

---
### 2. explore_schema
Purpose: Explore available relations under a domain.
Parameters:
  - pattern (String, REQUIRED): The domain name from [Available Domains].
    Must be a single domain word, NOT a dotted relation.
    Example: "film", "people", "sports" (NOT "film.actor.film")

Syntax: explore_schema(pattern="DOMAIN")

---
### 3. plan (Deep Retrieval)
Purpose: Plan retrieval paths (single or multi-step) to answer the question.
Parameters:
  - anchor (String or List[String], REQUIRED): The proven entities to start from. Can be a single entity or a list of up to 3 verified entity names.
  - question (String, REQUIRED): Intent description.
  - related (List[String], REQUIRED): Core relations (high confidence).
  - maybe_related (List[String], OPTIONAL): Hesitant/uncertain relations.
  - constraint_relations (List[String], OPTIONAL): Answer-required attribute relations.
      * From explored schema + Core Relations.
      * Used to filter leaf nodes by attribute values.
  - constraint_entities (List[String], OPTIONAL): Answer-required entities.
      * From checked entities + Core Entities.
      * Used to filter leaf nodes by connectivity.

Syntax:
plan(
  question="Find movies directed by Nolan",
  anchor=["Christopher Nolan"],
  related=["film.director.film"],
  maybe_related=["film.producer.film"],
  constraint_relations=["film.film.initial_release_date"],
  constraint_entities=["Academy Awards"]
)


---
### 4. action (Action Space Execution)
Purpose: Execute a graph traversal path to retrieve candidates from a specific action space.
Parameters:
  - anchor (List[Str], REQUIRED): The verified entity to start traversal from.
  - path (List[Dict], REQUIRED): Sequence of relation steps.
    Each step must have: {"relation": "...", "direction": "out"|"in"}

Syntax:
action(
  anchor=["StartEntity1", "StartEntity2"],
  path=[
    {"relation": "rel.1", "direction": "out"}, 
    {"relation": "rel.2", "direction": "in"}
  ]
)

### 5. filter (Filtering) - STAGE 4 ONLY
Purpose: Filter candidates using specific attributes or entities. Recover falsely pruned entities.
Parameters:
  - constraint_relations (List[String], OPTIONAL): Relations to check (e.g., "relation.property").
    ⚠️ MUST NOT be used in `action` path.
  - constraint_entities (List[String], OPTIONAL): Entity values to require (e.g., "Value A", "Value B").
    ⚠️ MUST NOT be the `anchor` entity.
  - scope (String, OPTIONAL): "selected" (default) or "all" (includes discarded leaves).

Syntax:
filter(
  constraint_relations=["domain.type.property"],
  constraint_entities=["Value B"],
  scope="selected"
)

---
Rule: In STAGE 4, use `filter` to analyze and refine candidates BEFORE final reasoning.

RULE: In STAGE 3, copy [Action] arguments EXACTLY (including constraint_*). Do not modify or invent paths.

---
END OF SYSTEM PROMPT
```

## 12. 附录 B：当前 Phase 1 提示语

这是当前在 discovery 阶段直接拼到用户消息后的提示语。

```text
[DISCOVERY PHASE]

Your first task is to understand the question and explore the knowledge graph.

━━━ CORE PHILOSOPHY ━━━
The knowledge graph is human-constructed. There is NO perfect relation for every question.
Your job: Select the BEST AVAILABLE option based on current information.
NEVER refuse to proceed just because options are not perfect.

━━━ THINKING STEPS ━━━
1. Identify entities mentioned in the question.
   → Use check_entities() to verify they exist in the graph.
   → Note: Only 100% EXACT STRING MATCH (case-sensitive) counts as the same entity.

2. Identify knowledge domains relevant to the question.
   → Use explore_schema() to discover available relations.
   → Record useful relations for later planning.

3. If the graph lacks perfect matches, use the CLOSEST available option.

━━━ REFLECTION CHECKPOINT ━━━
Before calling tools, verify:
  □ Is entity_substring exactly as it appears in the question?
  □ Is pattern a valid domain name (e.g., "film", "people", "sports")?

━━━ OUTPUT FORMAT ━━━
✅ ALLOWED: check_entities(), explore_schema()
❌ FORBIDDEN: plan(), match_pattern(), <answer>

<reasoning>
  [Your analysis - free format]
</reasoning>
<act>
  <query>check_entities(entity_substring="...")</query>
  <query>explore_schema(pattern="...")</query>
</act>
```

## 13. 结论

当前系统已经不是一个单纯的 prompt agent，而是一个由以下四层共同塑造的代理：

1. 主系统提示语
2. 环境 validator / feedback / phase hint
3. 图后端动作空间
4. 状态更新与候选压缩机制

因此，后续任何模型改进讨论，都建议围绕下面三个问题展开：

- 哪些能力应继续留在外部环境中？
- 哪些能力应该沉淀成独立 skills？
- 如何构建一条“运行 -> 诊断 -> skill 修订 -> 回放验证”的自进化闭环？
