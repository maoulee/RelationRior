# Inference Runtime

这份说明对应新的独立推理代码：

- [src/subgraph_kgqa/inference/runtime.py](/zhaoshu/subgraph/src/subgraph_kgqa/inference/runtime.py)
- [src/subgraph_kgqa/inference/state.py](/zhaoshu/subgraph/src/subgraph_kgqa/inference/state.py)
- [src/subgraph_kgqa/inference/history.py](/zhaoshu/subgraph/src/subgraph_kgqa/inference/history.py)
- [src/subgraph_kgqa/inference/hints.py](/zhaoshu/subgraph/src/subgraph_kgqa/inference/hints.py)
- [src/subgraph_kgqa/inference/backend.py](/zhaoshu/subgraph/src/subgraph_kgqa/inference/backend.py)
- [src/subgraph_kgqa/inference/parser.py](/zhaoshu/subgraph/src/subgraph_kgqa/inference/parser.py)

## 目标

这套 runtime 的目标不是替代 `plug` 里的完整历史环境，也不是替代 `test_pipe6.py` 的离线评测器。

它只承担推理主线里的几个核心职责：

1. 状态维护
2. 历史压缩
3. 提示注入
4. 输出解析
5. 图后端交互

## 边界

它刻意不负责：

1. 数据集批量评测
2. 报告聚合
3. best-of-n
4. 训练/RL 相关逻辑

## 设计原则

### 1. 复用 `plug` 的成熟逻辑

新的 runtime 没有重新发明状态机，而是尽量复用了：

- `PhaseHintGenerator`
- `RepairPolicy`
- `FrontendValidator`
- `BackendChecker`
- `strip_reason_blocks`
- `compress_with_structured_data`

### 2. 与 `test_pipe6.py` 解耦

`test_pipe6.py` 里仍然保留完整的离线评测壳。

新的 `inference/` 目录只保留“单条样本推理闭环”必需的能力，不再依赖 `test_pipe6.TestRunner`。

### 3. 阶段提示只发一次

新的 [hints.py](/zhaoshu/subgraph/src/subgraph_kgqa/inference/hints.py) 做了两层控制：

- 普通阶段提示：按 stage key 只发一次
- repair 提示：只抑制连续重复

这样可以避免同一阶段提示在相邻多轮里反复出现。
