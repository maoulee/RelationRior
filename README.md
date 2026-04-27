# Subgraph KGQA

This repository now exposes the KGQA reinforcement-learning environment, offline evaluation pipeline, and graph backend as one project instead of a group of standalone scripts.

## Layout

- `src/subgraph_kgqa/`: unified Python package and CLI.
- `scripts/train_grpo.sh`: GRPO training launcher with graph-server lifecycle management.
- `scripts/start_graph_server.sh`: graph backend launcher.
- `scripts/evaluate_pipeline.sh`: offline evaluation launcher.
- `plug.py`: stable unified Swift plugin entrypoint.
- `evaluate.py`: stable offline evaluation entrypoint.
- `graph_server.py`: stable graph-backend entrypoint.

The legacy research files are still preserved in the repository root:

- `plug_v11.py`: legacy reward / scheduler implementation.
- `plug_v12_feedback.py`: legacy feedback / validation implementation.
- `v10_environment.py`: legacy standalone environment feedback implementation.
- `test_pipe6.py`: legacy offline evaluation implementation, now imported through `evaluate.py`.
- `sys_prompt.py`: system prompt.

## Quick Start

Start the graph backend:

```bash
bash scripts/start_graph_server.sh
```

Run the offline evaluation pipeline:

```bash
python evaluate.py
```

Run GRPO training:

```bash
bash scripts/train_grpo.sh
```

## Stable Entry Points

For day-to-day use, prefer these stable files instead of versioned research files:

- `plug.py`: training plugin entrypoint
- `evaluate.py`: offline test / evaluation entrypoint
- `graph_server.py`: graph backend entrypoint

The versioned files remain as implementation backends so we can refactor incrementally without breaking training commands.

## Config

The shell launchers accept environment-variable overrides. Common ones:

- `MODEL_PATH`
- `DATASET_PATH`
- `OUTPUT_DIR`
- `GRAPH_SERVER_PORT`
- `GRAPH_SERVER_DATASET`
- `GRAPH_SERVER_SPLIT`
- `PYTHON_BIN`
- `SWIFT_BIN`

The graph backend source defaults to `/zhaoshu/SubgraphRAG-main/retrieve/graph_server.py`. Override with `SUBGRAPH_GRAPH_SERVER_SOURCE` if you migrate that file again later.
