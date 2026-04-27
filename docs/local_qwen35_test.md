# Local Qwen3.5-9B Test

This project can now run the existing KGQA evaluation loop against the local model at `/zhaoshu/llm/Qwen3.5-9B`.

## Start only the model server

```bash
cd /zhaoshu/subgraph
./scripts/start_local_qwen35_server.sh
```

Defaults:

- model path: `/zhaoshu/llm/Qwen3.5-9B`
- served model name: `qwen35-9b-local`
- port: `8000`
- GPU: `CUDA_VISIBLE_DEVICES=0,1`
- default tensor parallel size: `2`
- default context length: `20480`
- default startup mode: `--enforce-eager` to avoid the current `vLLM 0.18.0` compile init crash in this environment
- default modality mode: `--language-model-only --skip-mm-profiling` so the multimodal encoder does not block text-only KGQA tests
- default parser mode: `--reasoning-parser qwen3`

## Run WebQSP with graph backend + local model

```bash
cd /zhaoshu/subgraph
./scripts/run_webqsp_qwen35_local.sh
```

This script will:

1. start the local Qwen3.5-9B OpenAI-compatible server if `8000` is not ready
2. start the graph backend on `webqsp/train` if `8001` is not ready
3. run `test_pipe6.py` with the local model and current graph backend

Default request sampling for this local runner:

- `max_tokens=4096`
- `temperature=0.7`
- `top_p=0.8`
- `top_k=20`
- `min_p=0.0`
- `presence_penalty=1.5`
- `repetition_penalty=1.0`
- `chat_template_kwargs.enable_thinking=false`

## Useful overrides

```bash
export CUDA_VISIBLE_DEVICES=0,1
export KGQA_LIMIT_CASES=1
export KGQA_TARGET_CASE_IDS=WebQTrn-42
export KGQA_MAX_TURNS=6
export VLLM_MAX_MODEL_LEN=20480
export VLLM_GPU_MEMORY_UTILIZATION=0.85
export KGQA_REPORT_FILE=/zhaoshu/subgraph/reports/test_qwen35_9b_local.md
```
