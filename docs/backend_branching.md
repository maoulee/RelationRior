# Backend Branching

当前目录不是 git 仓库，所以图后端实验采用“文件级双分支工作区”。

## 目录

- [original graph_server.py](/zhaoshu/subgraph/backend_branches/original/retrieve/graph_server.py)
- [experiment graph_server.py](/zhaoshu/subgraph/backend_branches/experiment/retrieve/graph_server.py)

两个分支都复用了原始数据目录：

- `/zhaoshu/SubgraphRAG-main/retrieve/data_files`

通过符号链接挂到：

- [original data_files](/zhaoshu/subgraph/backend_branches/original/retrieve/data_files)
- [experiment data_files](/zhaoshu/subgraph/backend_branches/experiment/retrieve/data_files)

## 用法

启动原始后端：

```bash
bash /zhaoshu/subgraph/scripts/start_graph_server_branch.sh original --dataset webqsp --split train --port 8001
```

启动实验后端：

```bash
bash /zhaoshu/subgraph/scripts/start_graph_server_branch.sh experiment --dataset webqsp --split train --port 8001
```

查看两个分支的代码差异：

```bash
bash /zhaoshu/subgraph/scripts/diff_backend_branches.sh
```

## 设计目的

这套结构的目的不是替代 git branch，而是提供一个稳定的实验位点：

1. `original` 始终保留当前可回归的基线版本
2. `experiment` 专门用于图后端关系表示、图结构展开、CVT canonical candidate 等实验
3. 不需要复制 12G 的原始 `retrieve/` 目录
