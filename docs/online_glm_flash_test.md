# Online GLM-4.7-Flash Test

This project now supports switching `test_pipe6.py` from local vLLM to an OpenAI-compatible online API by environment variables.

If plain `http_proxy/https_proxy` is not enough in this environment, use the bundled `graftcp` path below. This environment has already shown that `graftcp` works more reliably than direct TLS through the proxy.

## 1. Start the graph backend

```bash
cd /zhaoshu/subgraph
./scripts/start_graph_server.sh --dataset webqsp --split train --port 8001
```

## 2. Run WebQSP with GLM-4.7-Flash

Do not hardcode the API key into source files. Export it in the shell:

```bash
export KGQA_LLM_API_KEY='your-secret-key'
cd /zhaoshu/subgraph
./scripts/run_webqsp_glm_flash.sh
```

## 2.1 Verify BigModel connectivity first

```bash
export KGQA_LLM_API_KEY='your-secret-key'
cd /zhaoshu/subgraph
./scripts/test_bigmodel_proxy.sh
```

## 3. Useful overrides

```bash
export KGQA_LIMIT_CASES=1
export KGQA_MAX_PARALLEL_CASES=1
export KGQA_BEST_OF_N=1
export KGQA_MAX_TURNS=6
export KGQA_TARGET_CASE_IDS=WebQTrn-42
export KGQA_TEST_DATA_PATH=/zhaoshu/subgraph/data/webqsp/webqsp_train.jsonl
export KGQA_REPORT_FILE=/zhaoshu/subgraph/reports/test_glm47_flash_webqsp.md
```

## 4. Optional: auto-start graph server inside the runner

```bash
export KGQA_LLM_API_KEY='your-secret-key'
export KGQA_START_GRAPH_SERVER=1
cd /zhaoshu/subgraph
./scripts/run_webqsp_glm_flash.sh
```

## Notes

- Default online API base: `https://open.bigmodel.cn/api/paas/v4`
- Default model name: `glm-4.7-flash`
- Default dataset: `/zhaoshu/subgraph/data/webqsp/webqsp_train.jsonl`
- If you do not set any of the new env vars, `test_pipe6.py` keeps its old local defaults.
- `run_webqsp_glm_flash.sh` defaults to `KGQA_USE_GRAFTCP=1` in this environment.
