#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRAFTCP_BIN="${KGQA_GRAFTCP_BIN:-/root/.graftcp-antigravity/graftcp/graftcp}"
GRAFTCP_LOCAL_BIN="${KGQA_GRAFTCP_LOCAL_BIN:-/root/.graftcp-antigravity/graftcp/local/graftcp-local}"
GRAFTCP_LOCAL_PORT="${KGQA_GRAFTCP_LOCAL_PORT:-2233}"
GRAFTCP_LOCAL_FIFO="${KGQA_GRAFTCP_LOCAL_FIFO:-/tmp/graftcplocal-2233.fifo}"
PROXY_HTTP_ADDR="${KGQA_PROXY_HTTP_ADDR:-127.0.0.1:8888}"
PROXY_SOCKS5_ADDR="${KGQA_PROXY_SOCKS5_ADDR:-127.0.0.1:8888}"

if [[ -z "${KGQA_LLM_API_KEY:-}" ]]; then
  echo "KGQA_LLM_API_KEY is not set." >&2
  exit 1
fi

"${GRAFTCP_LOCAL_BIN}" \
  -listen ":${GRAFTCP_LOCAL_PORT}" \
  -http_proxy "${PROXY_HTTP_ADDR}" \
  -socks5 "${PROXY_SOCKS5_ADDR}" \
  -select_proxy_mode auto \
  -pipepath "${GRAFTCP_LOCAL_FIFO}" \
  -logfile /tmp/graftcp-local.log \
  -logfile-truncate &
GRAFTCP_LOCAL_PID=$!
trap 'kill "${GRAFTCP_LOCAL_PID}" >/dev/null 2>&1 || true' EXIT
sleep 2

unset http_proxy https_proxy all_proxy no_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY

exec "${GRAFTCP_BIN}" \
  -p "${GRAFTCP_LOCAL_PORT}" \
  -f "${GRAFTCP_LOCAL_FIFO}" \
  curl --max-time 60 -sS "https://open.bigmodel.cn/api/paas/v4/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${KGQA_LLM_API_KEY}" \
    -d "{\"model\":\"${KGQA_MODEL_NAME:-glm-4.7-flash}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK only.\"}],\"max_tokens\":8,\"temperature\":0.1}"
